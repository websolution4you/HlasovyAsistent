import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from standalone_voice import VoiceSession, _sign, _valid_token
from voice_providers import elevenlabs_key


class StandaloneVoiceSafetyTests(unittest.IsolatedAsyncioTestCase):
    def test_standalone_key_has_priority_over_legacy_agent_key(self):
        environment = {
            "ELEVENLABS_API_KEY": "standalone-key",
            "ELEVENLABS_NTC_API_KEY": "legacy-agent-key",
        }
        with patch.dict("os.environ", environment, clear=True):
            self.assertEqual(elevenlabs_key(), "standalone-key")

    async def test_barge_in_cancels_speech_and_clears_twilio_audio(self):
        websocket = AsyncMock()
        session = VoiceSession(websocket, "CA123456789012345678", "+421900000000", SimpleNamespace())
        session.stream_sid = "MZstream"
        session.speech_task = asyncio.create_task(asyncio.sleep(10))

        await session.barge_in()
        await asyncio.gather(session.speech_task, return_exceptions=True)

        self.assertTrue(session.interrupted)
        self.assertTrue(session.speech_task.cancelled())
        websocket.send_json.assert_awaited_once_with({"event": "clear", "streamSid": "MZstream"})

    async def test_new_turn_cancels_previous_turn_instead_of_queueing(self):
        session = VoiceSession(AsyncMock(), "CA123456789012345678", "+421900000000", SimpleNamespace())
        started = asyncio.Event()

        async def slow_turn(_):
            started.set()
            await asyncio.sleep(10)

        with patch.object(session, "process_turn", side_effect=slow_turn):
            await session.start_turn("prvý vstup")
            first = session.turn_task
            await started.wait()
            await session.start_turn("nový vstup")
            self.assertTrue(first.cancelled())
            await session.shutdown()

    async def test_short_yes_confirms_pending_booking_without_llm(self):
        websocket = AsyncMock()
        session = VoiceSession(websocket, "CA123456789012345678", "+421900000000", SimpleNamespace())
        session.stream_sid = "MZstream"
        session.pending_booking = {
            "sport": "badminton",
            "court_id": "badminton-1",
            "customer_name": "Ján Novák",
            "start_time_iso": "2026-08-10T10:00:00",
            "duration_minutes": 60,
        }
        with (
            patch.object(session, "run_tool", AsyncMock(return_value={"status": "success"})) as run_tool,
            patch.object(session, "speak", AsyncMock()) as speak,
            patch("standalone_voice.llm_client") as llm,
            patch("standalone_voice.asyncio.sleep", AsyncMock()),
        ):
            await session.process_turn("Áno.")

        run_tool.assert_awaited_once_with("create_booking", session.pending_booking)
        speak.assert_awaited_once()
        llm.assert_not_called()
        websocket.close.assert_awaited_once_with(code=1000)

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
