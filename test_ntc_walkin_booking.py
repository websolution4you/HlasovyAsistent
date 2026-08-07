import datetime as dt
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ntc_walkin_booking import BRATISLAVA, create_booking, validate_slot

NOW = dt.datetime(2026, 8, 7, 12, 0, tzinfo=BRATISLAVA)


class WalkinBookingTests(unittest.TestCase):
    def test_slot_rules_enforce_horizon_hours_and_duration(self):
        validate_slot("tennis", "2026-08-10T20:00:00", 120, NOW)
        with self.assertRaisesRegex(ValueError, "14 dní"):
            validate_slot("tennis", "2026-08-22T10:00:00", 60, NOW)
        with self.assertRaisesRegex(ValueError, "otváracích"):
            validate_slot("squash", "2026-08-08T20:30:00", 60, NOW)
        with self.assertRaisesRegex(ValueError, "30-minútových"):
            validate_slot("badminton", "2026-08-10T10:00:00", 45, NOW)

    def test_walkin_booking_uses_existing_calendar_without_member_id(self):
        table = Mock()
        table.insert.return_value.execute.return_value.data = [{"id": "booking-1"}]
        supabase = Mock()
        supabase.table.return_value = table
        main = SimpleNamespace(
            NTC_TENANT_ID="595cbb6c-1019-41ae-b1c2-a60c13c8dcdf",
            supabase=supabase,
            _normalize_phone=lambda value: value,
            _get_ntc_busy_courts=lambda start, end: set(),
        )
        future = dt.datetime.now(BRATISLAVA) + dt.timedelta(days=1)
        future = future.replace(hour=10, minute=0, second=0, microsecond=0)
        result = create_booking(
            main, "badminton", "badminton-1", "Ján Novák", "+421900000000", future.isoformat(), 60,
        )

        self.assertEqual(result["status"], "success")
        supabase.table.assert_called_once_with("bookings")
        payload = table.insert.call_args.args[0]
        self.assertNotIn("user_id", payload)
        self.assertEqual(payload["tenant_id"], main.NTC_TENANT_ID)
        self.assertEqual(payload["customer_name"], "Ján Novák")
        self.assertEqual(json.loads(payload["notes"])["source"], "voice-assistant-walkin")


if __name__ == "__main__":
    unittest.main()
