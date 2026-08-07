import asyncio
import hashlib
import hmac
import json
import os
import time
from datetime import datetime
from html import escape as xml_escape
from typing import Any

from fastapi import Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.request_validator import RequestValidator

from ntc_walkin_booking import BRATISLAVA, check_availability, create_booking
from voice_config import AFFIRMATIVE, GREETING, SYSTEM_PROMPT, TOOLS
from voice_providers import llm_client, send_scribe_audio, scribe_session, synthesize


def _sign(call_sid: str, expires: int, secret: str) -> str:
    return hmac.new(secret.encode(), f"{call_sid}.{expires}".encode(), hashlib.sha256).hexdigest()


def _valid_token(call_sid: str, expires: str, signature: str, secret: str) -> bool:
    now = int(time.time())
    return (
        expires.isdigit()
        and now <= int(expires) <= now + 360
        and hmac.compare_digest(signature, _sign(call_sid, int(expires), secret))
    )


def _public_url(request: Request) -> str:
    configured = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if configured:
        return f"{configured}{request.url.path}"
    proto = request.headers.get("x-forwarded-proto", "https").split(",")[0]
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    return f"{proto}://{host}{request.url.path}"


def _ws_base(request: Request) -> str:
    configured = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if configured:
        return configured.replace("https://", "wss://").replace("http://", "ws://")
    return f"wss://{request.headers.get('x-forwarded-host') or request.headers.get('host', '')}"


async def _form(request: Request) -> dict[str, str]:
    source = request.query_params if request.method == "GET" else await request.form()
    return {key: str(value) for key, value in dict(source).items()}


class VoiceSession:
    def __init__(self, websocket: WebSocket, call_sid: str, phone: str, main_module):
        self.websocket = websocket
        self.call_sid = call_sid
        self.phone = phone
        self.main = main_module
        self.stream_sid = ""
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.turn_lock = asyncio.Lock()
        self.speech_task: asyncio.Task | None = None
        self.last_user_text = ""
        self.last_availability: dict[str, Any] | None = None
        self.should_end = False

    async def speak(self, text: str) -> None:
        if not text or not self.stream_sid:
            return
        self.speech_task = asyncio.create_task(synthesize(text, self.websocket, self.stream_sid))
        try:
            await self.speech_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"[standalone/tts] synthesis failed: {exc}")
        finally:
            self.speech_task = None

    async def barge_in(self) -> None:
        if self.speech_task and not self.speech_task.done():
            self.speech_task.cancel()
            await self.websocket.send_json({"event": "clear", "streamSid": self.stream_sid})

    async def run_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        if name == "check_availability":
            result = await asyncio.to_thread(
                check_availability,
                self.main,
                arguments["sport"],
                arguments["start_time_iso"],
                int(arguments["duration_minutes"]),
            )
            self.last_availability = {**arguments, **result}
            return result
        if name == "create_booking":
            if not AFFIRMATIVE.search(self.last_user_text):
                return {"status": "confirmation_required", "message": "Zákazník rezerváciu výslovne nepotvrdil."}
            if not self.last_availability or self.last_availability.get("status") != "available":
                return {"status": "availability_required", "message": "Termín nebol bezpečne overený."}
            if arguments.get("court_id") not in self.last_availability.get("free_courts", []):
                return {"status": "availability_required", "message": "Kurt nebol medzi overenými voľnými kurtmi."}
            for field in ("sport", "start_time_iso", "duration_minutes"):
                if str(arguments.get(field)) != str(self.last_availability.get(field)):
                    return {"status": "availability_required", "message": "Potvrdené údaje sa nezhodujú s overeným termínom."}
            result = await asyncio.to_thread(
                create_booking,
                self.main,
                arguments["sport"],
                arguments["court_id"],
                arguments["customer_name"],
                self.phone,
                arguments["start_time_iso"],
                int(arguments["duration_minutes"]),
            )
            self.last_availability = None
            return result
        if name == "end_call":
            self.should_end = True
            return {"status": "ok"}
        return {"status": "error", "message": "Neznámy nástroj."}

    async def process_turn(self, transcript: str) -> None:
        async with self.turn_lock:
            self.last_user_text = transcript
            now = datetime.now(BRATISLAVA)
            self.messages.append({
                "role": "user",
                "content": f"Aktuálny lokálny dátum a čas: {now.isoformat()}. Zákazník povedal: {transcript}",
            })
            client, model = llm_client()
            for _ in range(4):
                response = await client.chat.completions.create(
                    model=model,
                    messages=self.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                )
                message = response.choices[0].message
                assistant: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
                if message.tool_calls:
                    assistant["tool_calls"] = [
                        {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
                        for call in message.tool_calls
                    ]
                self.messages.append(assistant)
                if message.content:
                    await self.speak(message.content)
                if not message.tool_calls:
                    break
                for call in message.tool_calls:
                    try:
                        result = await self.run_tool(call.function.name, json.loads(call.function.arguments or "{}"))
                    except Exception as exc:
                        result = {"status": "error", "message": str(exc)}
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
            if self.should_end:
                await self.websocket.close(code=1000)


async def _voice_socket(websocket: WebSocket, session: VoiceSession) -> None:
    connector = await scribe_session()
    async with connector as scribe:
        async def inbound():
            while True:
                try:
                    message = json.loads(await websocket.receive_text())
                except WebSocketDisconnect:
                    return
                event = message.get("event")
                if event == "start":
                    start = message["start"]
                    session.stream_sid = start["streamSid"]
                    session.phone = session.main._normalize_phone(
                        start.get("customParameters", {}).get("phone", "") or session.phone,
                    )
                    asyncio.create_task(session.speak(GREETING))
                elif event == "media":
                    await send_scribe_audio(scribe, message["media"]["payload"])
                elif event == "stop":
                    return

        async def transcripts():
            async for raw in scribe:
                message = json.loads(raw)
                kind = message.get("message_type") or message.get("type")
                text = str(message.get("text") or message.get("transcript") or "").strip()
                if kind == "partial_transcript" and text:
                    await session.barge_in()
                elif kind == "committed_transcript" and text:
                    asyncio.create_task(session.process_turn(text))
                elif kind in {"auth_error", "quota_exceeded", "transcriber_error", "error"}:
                    raise RuntimeError(f"ElevenLabs Scribe error: {kind}")

        tasks = {asyncio.create_task(inbound()), asyncio.create_task(transcripts())}
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()


def register_standalone_voice(app, main_module) -> None:
    @app.api_route("/twilio/standalone", methods=["GET", "POST"])
    async def standalone_webhook(request: Request):
        form = await _form(request)
        token = os.getenv("TWILIO_AUTH_TOKEN", "")
        signature = request.headers.get("x-twilio-signature", "")
        if not token or not signature or not RequestValidator(token).validate(_public_url(request), form, signature):
            return Response(status_code=403)
        has_elevenlabs = bool(os.getenv("ELEVENLABS_NTC_API_KEY") or os.getenv("ELEVENLABS_API_KEY"))
        if not has_elevenlabs or not os.getenv("ELEVENLABS_VOICE_ID") or not (os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")):
            return Response(status_code=503)
        call_sid = form.get("CallSid", "")
        phone = main_module._normalize_phone(form.get("From", ""))
        if not re_call_sid(call_sid):
            return Response(status_code=400)
        expires = int(time.time()) + 300
        signature = _sign(call_sid, expires, token)
        url = f"{_ws_base(request)}/ws/standalone/{call_sid}/{expires}/{signature}"
        xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="{xml_escape(url, quote=True)}"><Parameter name="phone" value="{xml_escape(phone, quote=True)}"/></Stream></Connect></Response>'
        return Response(content=xml, media_type="application/xml")

    @app.websocket("/ws/standalone/{call_sid}/{expires}/{signature}")
    async def standalone_socket(websocket: WebSocket, call_sid: str, expires: str, signature: str):
        token = os.getenv("TWILIO_AUTH_TOKEN", "")
        if not token or not _valid_token(call_sid, expires, signature, token):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await _voice_socket(websocket, VoiceSession(websocket, call_sid, "", main_module))


def re_call_sid(value: str) -> bool:
    return len(value) >= 20 and value.startswith("CA") and value.isalnum()
