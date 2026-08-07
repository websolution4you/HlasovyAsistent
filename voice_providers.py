import base64
import json
import os
from collections.abc import AsyncIterator

import httpx
import websockets
from fastapi import WebSocket
from openai import AsyncAzureOpenAI, AsyncOpenAI


def elevenlabs_key() -> str:
    return os.getenv("ELEVENLABS_NTC_API_KEY") or os.environ["ELEVENLABS_API_KEY"]


def llm_client():
    if os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        ), os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]), os.getenv("OPENAI_MODEL", "gpt-4o-mini")


async def synthesize(text: str, websocket: WebSocket, stream_sid: str) -> None:
    voice_id = os.environ["ELEVENLABS_VOICE_ID"]
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    params = {
        "model_id": os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5"),
        "output_format": "ulaw_8000",
        "optimize_streaming_latency": "3",
    }
    body = {
        "text": text,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.8,
            "style": 0.2,
            "speed": 0.92,
        },
    }
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream(
            "POST",
            url,
            params=params,
            headers={"xi-api-key": elevenlabs_key()},
            json=body,
        ) as response:
            response.raise_for_status()
            buffer = b""
            async for chunk in response.aiter_bytes():
                buffer += chunk
                complete = len(buffer) - (len(buffer) % 160)
                for offset in range(0, complete, 160):
                    await websocket.send_json({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": base64.b64encode(buffer[offset:offset + 160]).decode()},
                    })
                buffer = buffer[complete:]


async def scribe_session() -> AsyncIterator:
    query = (
        "model_id=scribe_v2_realtime&language_code=sk&audio_format=ulaw_8000"
        "&commit_strategy=vad&vad_silence_threshold_secs=0.8&vad_threshold=0.4"
        "&min_speech_duration_ms=100&min_silence_duration_ms=100"
    )
    return websockets.connect(
        f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{query}",
        additional_headers={"xi-api-key": elevenlabs_key()},
        max_size=10 * 1024 * 1024,
    )


async def send_scribe_audio(scribe, payload: str) -> None:
    await scribe.send(json.dumps({
        "message_type": "input_audio_chunk",
        "audio_base_64": payload,
    }))
