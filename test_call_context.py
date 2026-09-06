import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from call_context import booking_operation_id, create_call_context, verify_call_context
from ntc_upcoming import UpcomingBookingsRequest, _resolve_verified_call


class CallContextTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {"VOICE_CALL_CONTEXT_SECRET": "test-secret-with-sufficient-entropy"},
            clear=False,
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()

    def test_telnyx_sip_context_round_trip(self):
        token = create_call_context(
            caller_phone="+421900000001",
            call_id="telnyx-call-control-id",
            conversation_id="conv_test",
            provider="sip",
            now=1_000,
        )
        context = verify_call_context(token, now=1_001)

        self.assertEqual(context["caller_phone"], "+421900000001")
        self.assertEqual(context["call_id"], "telnyx-call-control-id")
        self.assertEqual(context["conversation_id"], "conv_test")
        self.assertEqual(context["provider"], "sip")

    def test_tampered_and_expired_contexts_are_rejected(self):
        token = create_call_context(
            caller_phone="+421900000001",
            call_id="sip-call-id",
            now=1_000,
            ttl_seconds=60,
        )

        self.assertIsNone(verify_call_context(token + "x", now=1_001))
        self.assertIsNone(verify_call_context(token, now=1_061))

    def test_booking_operation_id_is_stable_and_call_scoped(self):
        booking = {
            "user_id": "user-1",
            "court_id": "tennis-1",
            "sport": "tennis",
            "start_at": "2026-09-10T08:00:00+00:00",
            "end_at": "2026-09-10T09:00:00+00:00",
        }
        first_context = {"call_id": "call-1", "conversation_id": "conv-1"}
        second_context = {"call_id": "call-2", "conversation_id": "conv-2"}

        first = booking_operation_id(first_context, booking)
        self.assertEqual(first, booking_operation_id(first_context, booking))
        self.assertNotEqual(first, booking_operation_id(second_context, booking))

    def test_resolver_accepts_signed_sip_context_without_process_memory(self):
        token = create_call_context(
            caller_phone="+421900000001",
            call_id="sip-call-id",
        )
        request = UpcomingBookingsRequest(call_context=token)
        main_module = SimpleNamespace(CALL_CONTEXT={})

        call_id, caller_phone = _resolve_verified_call(request, main_module)

        self.assertEqual(call_id, "sip-call-id")
        self.assertEqual(caller_phone, "+421900000001")

    def test_resolver_preserves_twilio_legacy_context(self):
        call_sid = "CA" + "a" * 32
        request = UpcomingBookingsRequest(call_sid=call_sid)
        main_module = SimpleNamespace(CALL_CONTEXT={call_sid: "+421900000002"})

        self.assertEqual(
            _resolve_verified_call(request, main_module),
            (call_sid, "+421900000002"),
        )

    def test_resolver_rejects_untrusted_provider_id(self):
        request = UpcomingBookingsRequest(call_sid="telnyx-call-id")
        with self.assertRaises(HTTPException) as raised:
            _resolve_verified_call(request, SimpleNamespace(CALL_CONTEXT={}))
        self.assertEqual(raised.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
