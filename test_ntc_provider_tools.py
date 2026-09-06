import asyncio
import datetime
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI, HTTPException
from starlette.background import BackgroundTasks

import main
from call_context import create_call_context
from ntc_upcoming import (
    CancelBookingRequest,
    UpcomingBookingsRequest,
    register_ntc_upcoming_tool,
)


class FakeQuery:
    def __init__(self, database, operation, payload=None):
        self.database = database
        self.operation = operation
        self.payload = payload
        self.filters = []
        self.start_after = ""

    def select(self, _columns):
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = payload
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def neq(self, column, value):
        return self

    def gte(self, column, value):
        self.filters.append((column, value))
        self.start_after = value
        return self

    def lte(self, column, value):
        return self

    def order(self, _column):
        return self

    def limit(self, _limit):
        return self

    def execute(self):
        filters = dict(self.filters)
        if self.operation == "update":
            self.database.last_update = {
                "payload": self.payload,
                "filters": filters,
            }
            return SimpleNamespace(data=[{"id": filters.get("id")}])

        rows = [
            row
            for row in self.database.rows
            if all(
                column in {"start_at", "end_at"} or row.get(column) == value
                for column, value in self.filters
            )
        ]
        return SimpleNamespace(data=rows)


class FakeRPC:
    def __init__(self, database, fn_name, payload):
        self.database = database
        self.fn_name = fn_name
        self.payload = payload

    def execute(self):
        self.database.rpc_calls.append({"fn_name": self.fn_name, "payload": self.payload})
        return SimpleNamespace(
            data=[
                {
                    "booking_id": "booking-member-42",
                    "charged_eur": 15.0,
                    "balance_eur": 45.0,
                }
            ]
        )


class FakeSupabase:
    def __init__(self, rows):
        self.rows = rows
        self.last_update = None
        self.rpc_calls = []

    def table(self, table_name):
        if table_name not in {"bookings", "wallet_transactions"}:
            raise AssertionError(f"Unexpected table: {table_name}")
        return FakeQuery(self, "select")

    def rpc(self, fn_name, payload):
        return FakeRPC(self, fn_name, payload)


class ProviderNeutralBookingToolsTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                "VOICE_CALL_CONTEXT_SECRET": "provider-tool-test-secret",
                "CORE_SUPABASE_SERVICE_ROLE_KEY": "confirmation-test-secret",
            },
            clear=False,
        )
        self.environment.start()
        future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=2)
        self.booking = {
            "id": "booking-1",
            "tenant_id": "tenant-ntc",
            "user_id": "user-1",
            "status": "confirmed",
            "sport": "tennis",
            "court_id": "tennis-2",
            "start_at": future.isoformat(),
            "end_at": (future + datetime.timedelta(hours=1)).isoformat(),
            "notes": {},
        }
        self.database = FakeSupabase([self.booking])
        self.main_module = SimpleNamespace(
            CALL_CONTEXT={},
            CONVERSATION_CONTEXT={},
            supabase=self.database,
            NTC_TENANT_ID="tenant-ntc",
            find_user_name_and_id_by_phone=lambda phone: ("Ján Novák", "user-1"),
            _parse_iso_to_utc=self.parse_iso_to_utc,
        )
        self.app = FastAPI()
        register_ntc_upcoming_tool(self.app, self.main_module)
        self.upcoming = self.route_endpoint("/api/ntc-upcoming-bookings")
        self.cancel = self.route_endpoint("/api/ntc-cancel-booking")
        self.call_context = create_call_context(
            caller_phone="+421900000001",
            call_id="telnyx-call-control-id",
            conversation_id="conv-provider-test",
            provider="sip",
        )

    def tearDown(self):
        self.environment.stop()

    @staticmethod
    def parse_iso_to_utc(value):
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(datetime.timezone.utc)

    def route_endpoint(self, path):
        return next(route.endpoint for route in self.app.routes if route.path == path)

    def test_signed_sip_context_lists_and_cancels_owned_booking(self):
        upcoming = asyncio.run(
            self.upcoming(UpcomingBookingsRequest(call_context=self.call_context))
        )
        self.assertEqual(upcoming["status"], "success")
        self.assertEqual(upcoming["count"], 1)
        self.assertEqual(upcoming["bookings"][0]["booking_reference"], "booking-1")

        prepared = asyncio.run(
            self.cancel(
                CancelBookingRequest(
                    call_context=self.call_context,
                    booking_reference="booking-1",
                    action="prepare",
                )
            )
        )
        self.assertEqual(prepared["status"], "awaiting_confirmation")
        self.assertFalse(prepared["cancelled"])

        confirmed = asyncio.run(
            self.cancel(
                CancelBookingRequest(
                    call_context=self.call_context,
                    booking_reference="booking-1",
                    action="confirm",
                    confirmation_token=prepared["confirmation_token"],
                )
            )
        )
        self.assertEqual(confirmed["status"], "cancelled")
        self.assertTrue(confirmed["cancelled"])
        self.assertTrue(
            any(
                call["fn_name"] == "wallet_refund_ntc_booking"
                and call["payload"] == {"p_booking_id": "booking-1"}
                for call in self.database.rpc_calls
            )
        )

    def test_signed_telnyx_context_creates_member_booking_with_idempotent_retries(self):
        req = main.CreateBookingRequest(
            sport="tennis",
            court_id="tennis-1",
            customer_name="Ján Novák",
            start_time_iso="2026-09-10T10:00:00",
            call_context=self.call_context,
        )

        with patch.object(main, "supabase", self.database), patch.object(
            main, "find_user_name_and_id_by_phone", return_value=("Ján Novák", "user-1")
        ):
            first_response = asyncio.run(main.ntc_create_booking(req, BackgroundTasks()))
            second_response = asyncio.run(main.ntc_create_booking(req, BackgroundTasks()))

        self.assertEqual(first_response["status"], "success")
        self.assertEqual(first_response["booking_id"], "booking-member-42")
        self.assertEqual(first_response["charged_eur"], 15.0)
        self.assertEqual(first_response["balance_eur"], 45.0)

        self.assertEqual(len(self.database.rpc_calls), 2)
        first_call = self.database.rpc_calls[0]
        second_call = self.database.rpc_calls[1]

        self.assertEqual(first_call["fn_name"], "wallet_create_ntc_booking")
        self.assertEqual(first_call["payload"]["p_user_id"], "user-1")
        self.assertEqual(first_call["payload"]["p_customer_phone"], "+421900000001")
        self.assertEqual(first_call["payload"]["p_sport"], "tennis")
        self.assertTrue(first_call["payload"]["p_idempotency_key"].startswith("voice-"))

        # Overenie, že oba retry requesty odoslali rovnaký idempotency kľúč
        self.assertEqual(
            first_call["payload"]["p_idempotency_key"],
            second_call["payload"]["p_idempotency_key"],
        )

    def test_member_booking_without_verified_context_is_rejected(self):
        req = main.CreateBookingRequest(
            sport="tennis",
            court_id="tennis-1",
            customer_name="Ján Novák",
            start_time_iso="2026-09-10T10:00:00",
            call_context="",
            customer_phone="+421900000001",
        )

        with patch.object(main, "supabase", self.database), patch.object(
            main, "find_user_name_and_id_by_phone", return_value=("Ján Novák", "user-1")
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(main.ntc_create_booking(req, BackgroundTasks()))
        self.assertEqual(raised.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
