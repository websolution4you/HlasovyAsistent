import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any


_DEFAULT_TTL_SECONDS = 4 * 60 * 60


def _secret() -> bytes:
    value = (
        os.getenv("VOICE_CALL_CONTEXT_SECRET", "").strip()
        or os.getenv("CORE_SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )
    if not value:
        raise RuntimeError("VOICE_CALL_CONTEXT_SECRET is not configured")
    return value.encode("utf-8")


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_call_context(
    *,
    caller_phone: str,
    call_id: str,
    conversation_id: str = "",
    provider: str = "sip",
    now: int | None = None,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> str:
    if not caller_phone or not call_id:
        return ""
    issued_at = int(time.time() if now is None else now)
    payload = {
        "caller_phone": caller_phone,
        "call_id": call_id,
        "conversation_id": conversation_id,
        "provider": provider,
        "iat": issued_at,
        "exp": issued_at + ttl_seconds,
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(_secret(), payload_bytes, hashlib.sha256).digest()
    return f"{_encode(payload_bytes)}.{_encode(signature)}"


def verify_call_context(token: str, *, now: int | None = None) -> dict[str, Any] | None:
    try:
        encoded_payload, encoded_signature = str(token or "").split(".", 1)
        payload_bytes = _decode(encoded_payload)
        signature = _decode(encoded_signature)
        expected = hmac.new(_secret(), payload_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(payload_bytes.decode("utf-8"))
        current_time = int(time.time() if now is None else now)
        if not isinstance(payload, dict) or payload.get("exp", 0) < current_time:
            return None
        if payload.get("iat", current_time + 1) > current_time + 60:
            return None
        if not payload.get("caller_phone") or not payload.get("call_id"):
            return None
        return payload
    except (ValueError, TypeError, json.JSONDecodeError, RuntimeError):
        return None


def booking_operation_id(context: dict[str, Any], booking_data: dict[str, Any]) -> str:
    canonical = json.dumps(
        {
            "call_id": context.get("call_id", ""),
            "conversation_id": context.get("conversation_id", ""),
            **booking_data,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "voice-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
