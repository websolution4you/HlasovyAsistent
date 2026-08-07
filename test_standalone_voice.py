import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from standalone_voice import VoiceSession, _sign, _valid_token


class StandaloneVoiceSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_requires_explicit_confirmation(self):
        session = VoiceSession(AsyncMock(), "CA123456789012345678", "+421900000000", SimpleNamespace())
        session.last_availability = {
            "status": "available",
            "sport": "badminton",
            "start_time_iso": "2026-08-10T10:00:00",
            "duration_minutes": 60,
            "free_courts": ["badminton-1"],
        }
        arguments = {
            "sport": "badminton",
            "court_id": "badminton-1",
            "customer_name": "Ján Novák",
            "start_time_iso": "2026-08-10T10:00:00",
            "duration_minutes": 60,
        }
        session.last_user_text = "Nie, zmeňme čas."
        with patch("standalone_voice.create_booking") as create:
            result = await session.run_tool("create_booking", arguments)
        self.assertEqual(result["status"], "confirmation_required")
        create.assert_not_called()

    async def test_create_rejects_changed_verified_slot(self):
        session = VoiceSession(AsyncMock(), "CA123456789012345678", "+421900000000", SimpleNamespace())
        session.last_user_text = "Áno, potvrdzujem."
        session.last_availability = {
            "status": "available",
            "sport": "badminton",
            "start_time_iso": "2026-08-10T10:00:00",
            "duration_minutes": 60,
            "free_courts": ["badminton-1"],
        }
        arguments = {
            "sport": "badminton",
            "court_id": "badminton-1",
            "customer_name": "Ján Novák",
            "start_time_iso": "2026-08-10T11:00:00",
            "duration_minutes": 60,
        }
        with patch("standalone_voice.create_booking") as create:
            result = await session.run_tool("create_booking", arguments)
        self.assertEqual(result["status"], "availability_required")
        create.assert_not_called()

    def test_media_token_is_short_lived_and_call_bound(self):
        expires = int(time.time()) + 300
        signature = _sign("CA123456789012345678", expires, "secret")
        self.assertTrue(_valid_token("CA123456789012345678", str(expires), signature, "secret"))
        self.assertFalse(_valid_token("CAother1234567890123", str(expires), signature, "secret"))
        self.assertFalse(_valid_token("CA123456789012345678", str(int(time.time()) - 1), signature, "secret"))


if __name__ == "__main__":
    unittest.main()
