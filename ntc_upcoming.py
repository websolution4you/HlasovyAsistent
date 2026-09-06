import base64
import datetime
import hashlib
import hmac
import json
import os
from zoneinfo import ZoneInfo

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from call_context import verify_call_context


class UpcomingBookingsRequest(BaseModel):
    call_sid: str = ""
    call_context: str = ""
    dynamic_variables: dict | None = None


class CancelBookingRequest(BaseModel):
    call_sid: str = ""
    call_context: str = ""
    dynamic_variables: dict | None = None

    booking_reference: str = Field(min_length=1, max_length=128)
    action: str
    confirmation_token: str | None = Field(default=None, max_length=2048)


_PENDING_CANCELLATION_TTL = datetime.timedelta(minutes=5)


_SLOVAK_WEEKDAYS = (
    "pondelok",
    "utorok",
    "streda",
    "štvrtok",
    "piatok",
    "sobota",
    "nedeľa",
)
_SLOVAK_MONTHS = (
    "januára",
    "februára",
    "marca",
    "apríla",
    "mája",
    "júna",
    "júla",
    "augusta",
    "septembra",
    "októbra",
    "novembra",
    "decembra",
)
_NTC_TIMEZONE = ZoneInfo("Europe/Bratislava")


def _valid_twilio_call_sid(call_sid: str) -> bool:
    return (
        len(call_sid) == 34
        and call_sid.startswith("CA")
        and all(character in "0123456789abcdefABCDEF" for character in call_sid[2:])
    )


def _resolve_verified_call(req, main_module) -> tuple[str, str]:
    context_token = str(getattr(req, "call_context", "") or "").strip()
    if not context_token:
        dyn = getattr(req, "dynamic_variables", None) or {}
        if isinstance(dyn, dict):
            context_token = str(dyn.get("call_context", "") or "").strip()

    if context_token:
        context = verify_call_context(context_token)
        if not context:
            raise HTTPException(
                status_code=403,
                detail="Kontext hovoru je neplatný alebo vypršal.",
            )
        return str(context["call_id"]), str(context["caller_phone"])

    call_sid = str(getattr(req, "call_sid", "") or "").strip()
    if not call_sid:
        dyn = getattr(req, "dynamic_variables", None) or {}
        if isinstance(dyn, dict):
            call_sid = str(dyn.get("call_sid", "") or "").strip()

    if not _valid_twilio_call_sid(call_sid):
        raise HTTPException(status_code=400, detail="Neplatný kontext hovoru.")
    caller_phone = main_module.CALL_CONTEXT.get(call_sid)
    if not caller_phone:
        raise HTTPException(
            status_code=403,
            detail="Hovor sa nepodarilo bezpečne overiť.",
        )
    return call_sid, caller_phone


def _confirmation_secret() -> bytes:
    secret = os.getenv("CORE_SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="Bezpečné potvrdenie zrušenia momentálne nie je dostupné.",
        )
    return secret.encode("utf-8")


def _encode_confirmation_token(payload: dict) -> str:
    payload_bytes = json.dumps(
        payload, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    signature = hmac.new(
        _confirmation_secret(), payload_bytes, hashlib.sha256
    ).digest()
    encoded_payload = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
    encoded_signature = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{encoded_payload}.{encoded_signature}"


def _decode_confirmation_token(token: str) -> dict | None:
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        payload_bytes = base64.urlsafe_b64decode(
            encoded_payload + "=" * (-len(encoded_payload) % 4)
        )
        signature = base64.urlsafe_b64decode(
            encoded_signature + "=" * (-len(encoded_signature) % 4)
        )
        expected_signature = hmac.new(
            _confirmation_secret(), payload_bytes, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(signature, expected_signature):
            return None
        payload = json.loads(payload_bytes.decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except HTTPException:
        raise
    except Exception:
        return None


def _sport_name(sport: str, court_id: str) -> str:
    value = str(sport or court_id or "").lower().strip()
    if "badminton" in value or "bedminton" in value:
        return "bedminton"
    if "squash" in value:
        return "squash"
    if "tennis-clay" in value or "antuka" in value:
        return "tenis na antuke"
    if "tennis" in value or "tenis" in value:
        return "tenis"
    return "šport"


def _court_name(court_id: str) -> str:
    value = str(court_id or "").lower().strip()
    number = value.rsplit("-", 1)[-1] if "-" in value else ""
    if not number.isdigit():
        return ""
    if value.startswith("tennis-clay-"):
        return f"dvorec číslo {int(number)}"
    return f"kurt číslo {int(number)}"


def _duration_text(total_minutes: int) -> str:
    if total_minutes <= 0:
        return ""
    if total_minutes % 60:
        return f"na {total_minutes} minút"
    hours = total_minutes // 60
    if hours == 1:
        return "na jednu hodinu"
    if 2 <= hours <= 4:
        return f"na {hours} hodiny"
    return f"na {hours} hodín"


def _court_id_from_row(row: dict) -> str:
    court_id = str(row.get("court_id") or "").strip()
    if court_id:
        return court_id

    try:
        notes = row.get("notes") or {}
        notes_object = json.loads(notes) if isinstance(notes, str) else notes
        if isinstance(notes_object, dict):
            return str(notes_object.get("courtId") or "").strip()
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return ""


def _format_booking(row: dict, parse_iso_to_utc) -> tuple[str, dict] | None:
    try:
        start_utc = parse_iso_to_utc(str(row.get("start_at") or ""))
        end_utc = parse_iso_to_utc(str(row.get("end_at") or ""))
    except Exception:
        return None
    if end_utc <= start_utc:
        return None

    start_local = start_utc.astimezone(_NTC_TIMEZONE)
    end_local = end_utc.astimezone(_NTC_TIMEZONE)
    duration_minutes = round((end_utc - start_utc).total_seconds() / 60)
    court_id = _court_id_from_row(row)
    sport = _sport_name(str(row.get("sport") or ""), court_id)
    court = _court_name(court_id)
    weekday = _SLOVAK_WEEKDAYS[start_local.weekday()]
    date_text = f"{start_local.day}. {_SLOVAK_MONTHS[start_local.month - 1]}"
    court_text = ""
    if court.startswith("kurt číslo "):
        court_text = f" na kurte číslo {court.rsplit(' ', 1)[-1]}"
    elif court:
        court_text = f" na {court}"

    spoken_text = (
        f"V {weekday} {date_text} od {start_local.strftime('%H:%M')} "
        f"do {end_local.strftime('%H:%M')} máte rezervovaný {sport}"
        f"{court_text} {_duration_text(duration_minutes)}."
    )
    public_booking = {
        "booking_reference": str(row.get("id") or ""),
        "sport": sport,
        "court": court,
        "start_at": start_local.isoformat(),
        "end_at": end_local.isoformat(),
        "duration_minutes": duration_minutes,
    }
    return spoken_text, public_booking


def _format_bookings(rows: list[dict], parse_iso_to_utc) -> tuple[str, list[dict]]:
    spoken_bookings = []
    public_bookings = []

    for row in rows:
        formatted = _format_booking(row, parse_iso_to_utc)
        if not formatted:
            continue
        spoken_text, public_booking = formatted
        if not public_booking["booking_reference"]:
            continue
        spoken_bookings.append(spoken_text)
        public_bookings.append(public_booking)

    count = len(public_bookings)
    if count == 0:
        return "V najbližšom čase u nás nemáte žiadnu potvrdenú rezerváciu.", []
    if count == 1:
        introduction = "Máte jednu nadchádzajúcu rezerváciu. "
    elif count <= 4:
        introduction = f"Máte {count} nadchádzajúce rezervácie. "
    else:
        introduction = f"Máte {count} nadchádzajúcich rezervácií. "
    return introduction + " ".join(spoken_bookings), public_bookings


class CallContextCleanupMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, main_module):
        super().__init__(app)
        self.main_module = main_module

    async def dispatch(self, request: Request, call_next):
        call_sid = ""
        conversation_id = ""
        if request.url.path == "/api/end_call" and request.method == "POST":
            try:
                body = await request.json()
                call_sid = str(body.get("call_sid") or body.get("CallSid") or "").strip()
                conversation_id = str(
                    body.get("conversation_id") or body.get("conversationId") or ""
                ).strip()
            except Exception:
                pass

        response = await call_next(request)

        if call_sid:
            removed_phone = self.main_module.CALL_CONTEXT.pop(call_sid, None)
            if removed_phone:
                print(f"[ntc-context] CALL_CONTEXT cleaned for call_sid={call_sid}")
        if conversation_id:
            self.main_module.CONVERSATION_CONTEXT.pop(conversation_id, None)
        return response


def register_ntc_upcoming_tool(app, main_module) -> None:
    app.add_middleware(CallContextCleanupMiddleware, main_module=main_module)

    @app.post("/api/ntc-upcoming-bookings", name="ntc_upcoming_bookings")
    async def ntc_upcoming_bookings(req: UpcomingBookingsRequest):
        call_id, caller_phone = _resolve_verified_call(req, main_module)

        if not main_module.supabase:
            raise HTTPException(
                status_code=503,
                detail="Rezervácie momentálne nie je možné overiť.",
            )

        customer_name, user_id = main_module.find_user_name_and_id_by_phone(caller_phone)
        if not user_id:
            return {
                "status": "customer_not_found",
                "has_bookings": False,
                "count": 0,
                "message": (
                    "Vaše rezervácie sa mi nepodarilo bezpečne overiť. "
                    "Obráťte sa, prosím, na recepciu."
                ),
                "bookings": [],
            }

        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            query = main_module.supabase.table("bookings").select(
                "id, sport, court_id, start_at, end_at, notes"
            )
            query = query.eq("tenant_id", main_module.NTC_TENANT_ID)
            query = query.eq("user_id", user_id)
            query = query.eq("status", "confirmed")
            query = query.gte("start_at", now_utc)
            result = query.order("start_at").limit(5).execute()
        except Exception as exc:
            print(
                f"[ntc-upcoming] Supabase query failed for verified "
                f"call_id={call_id}: {exc}"
            )
            raise HTTPException(
                status_code=503,
                detail="Rezervácie momentálne nie je možné overiť.",
            ) from exc

        message, bookings = _format_bookings(
            result.data or [], main_module._parse_iso_to_utc
        )
        print(
            f"[ntc-upcoming] Verified call_id={call_id}, "
            f"customer='{customer_name or ''}', returned_bookings={len(bookings)}"
        )
        return {
            "status": "success",
            "has_bookings": bool(bookings),
            "count": len(bookings),
            "message": message,
            "bookings": bookings,
        }

    @app.post("/api/ntc-cancel-booking", name="ntc_cancel_booking")
    async def ntc_cancel_booking(req: CancelBookingRequest):
        print(
            f"[ntc-cancel] Incoming request: action='{req.action}', "
            f"booking_reference='{req.booking_reference}', "
            f"has_call_context={bool(req.call_context)}, "
            f"has_dyn_context={bool((req.dynamic_variables or {}).get('call_context'))}"
        )
        call_id, caller_phone = _resolve_verified_call(req, main_module)
        booking_reference = str(req.booking_reference or "").strip()
        action = str(req.action or "").strip().lower()
        confirmation_token = str(req.confirmation_token or "").strip()

        if action not in {"prepare", "confirm"}:
            raise HTTPException(
                status_code=400,
                detail="Action musí byť prepare alebo confirm.",
            )

        if not main_module.supabase:
            raise HTTPException(
                status_code=503,
                detail="Rezerváciu momentálne nie je možné zrušiť.",
            )

        customer_name, user_id = main_module.find_user_name_and_id_by_phone(caller_phone)
        if not user_id:
            return {
                "status": "customer_not_found",
                "cancelled": False,
                "message": (
                    "Rezerváciu sa nepodarilo bezpečne overiť. "
                    "Obráťte sa, prosím, na recepciu."
                ),
            }

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now_utc.isoformat()

        if action == "prepare":
            try:
                query = main_module.supabase.table("bookings").select(
                    "id, sport, court_id, start_at, end_at, notes"
                )
                query = query.eq("id", booking_reference)
                query = query.eq("tenant_id", main_module.NTC_TENANT_ID)
                query = query.eq("user_id", user_id)
                query = query.eq("status", "confirmed")
                result = query.gte("start_at", now_iso).limit(1).execute()
            except Exception as exc:
                print(
                    f"[ntc-cancel] Prepare query failed for verified "
                    f"call_id={call_id}: {exc}"
                )
                raise HTTPException(
                    status_code=503,
                    detail="Rezerváciu momentálne nie je možné overiť.",
                ) from exc

            rows = result.data or []
            if not rows:
                return {
                    "status": "not_cancellable",
                    "cancelled": False,
                    "message": (
                        "Táto rezervácia nepatrí aktuálnemu volajúcemu, "
                        "už bola zrušená alebo ju nie je možné zrušiť."
                    ),
                }

            formatted = _format_booking(rows[0], main_module._parse_iso_to_utc)
            if not formatted:
                return {
                    "status": "not_cancellable",
                    "cancelled": False,
                    "message": "Túto rezerváciu nie je možné bezpečne zrušiť.",
                }
            spoken_text, booking = formatted
            expires_at = now_utc + _PENDING_CANCELLATION_TTL
            confirmation_token = _encode_confirmation_token(
                {
                    "call_id": call_id,
                    "booking_reference": booking_reference,
                    "user_id": str(user_id),
                    "expires_at": int(expires_at.timestamp()),
                }
            )

            confirmation_message = (
                f"{spoken_text} Naozaj chcete túto rezerváciu zrušiť?"
            )
            print(
                f"[ntc-cancel] Prepared call_id={call_id}, "
                f"customer='{customer_name or ''}', booking={booking_reference}"
            )
            return {
                "status": "awaiting_confirmation",
                "cancelled": False,
                "expires_in_seconds": int(_PENDING_CANCELLATION_TTL.total_seconds()),
                "confirmation_token": confirmation_token,
                "message": confirmation_message,
                "booking": booking,
            }

        token_payload = _decode_confirmation_token(confirmation_token)
        token_is_valid = bool(
            token_payload
            and token_payload.get("call_id") == call_id
            and token_payload.get("booking_reference") == booking_reference
            and token_payload.get("user_id") == str(user_id)
            and isinstance(token_payload.get("expires_at"), int)
            and token_payload["expires_at"] >= int(now_utc.timestamp())
        )

        if not token_is_valid:
            return {
                "status": "confirmation_required",
                "cancelled": False,
                "message": (
                    "Zrušenie nebolo pripravené alebo potvrdenie vypršalo. "
                    "Najprv je potrebné znova overiť konkrétnu rezerváciu."
                ),
            }

        try:
            query = main_module.supabase.table("bookings").update(
                {"status": "cancelled"}
            )
            query = query.eq("id", booking_reference)
            query = query.eq("tenant_id", main_module.NTC_TENANT_ID)
            query = query.eq("user_id", user_id)
            query = query.eq("status", "confirmed")
            result = query.gte("start_at", now_iso).execute()
        except Exception as exc:
            print(
                f"[ntc-cancel] Confirm update failed for verified "
                f"call_id={call_id}: {exc}"
            )
            raise HTTPException(
                status_code=503,
                detail="Rezerváciu momentálne nie je možné zrušiť.",
            ) from exc

        if not (result.data or []):
            return {
                "status": "not_cancellable",
                "cancelled": False,
                "message": (
                    "Rezervácia už bola zrušená alebo ju už nie je možné zrušiť."
                ),
            }

        print(
            f"[ntc-cancel] Cancelled call_id={call_id}, "
            f"customer='{customer_name or ''}', booking={booking_reference}"
        )
        return {
            "status": "cancelled",
            "cancelled": True,
            "message": "Rezervácia bola úspešne zrušená.",
        }
