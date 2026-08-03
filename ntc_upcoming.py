import datetime
import json
from zoneinfo import ZoneInfo

from fastapi import HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware


class UpcomingBookingsRequest(BaseModel):
    call_sid: str


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


def _format_bookings(rows: list[dict], parse_iso_to_utc) -> tuple[str, list[dict]]:
    spoken_bookings = []
    public_bookings = []

    for row in rows:
        try:
            start_utc = parse_iso_to_utc(str(row.get("start_at") or ""))
            end_utc = parse_iso_to_utc(str(row.get("end_at") or ""))
        except Exception:
            continue
        if end_utc <= start_utc:
            continue

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

        spoken_bookings.append(
            f"V {weekday} {date_text} od {start_local.strftime('%H:%M')} "
            f"do {end_local.strftime('%H:%M')} máte rezervovaný {sport}"
            f"{court_text} {_duration_text(duration_minutes)}."
        )
        public_bookings.append(
            {
                "sport": sport,
                "court": court,
                "start_at": start_local.isoformat(),
                "end_at": end_local.isoformat(),
                "duration_minutes": duration_minutes,
            }
        )

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
        call_sid = str(req.call_sid or "").strip()
        if not _valid_twilio_call_sid(call_sid):
            raise HTTPException(status_code=400, detail="Neplatný kontext hovoru.")

        caller_phone = main_module.CALL_CONTEXT.get(call_sid)
        if not caller_phone:
            raise HTTPException(
                status_code=403,
                detail="Hovor sa nepodarilo bezpečne overiť.",
            )

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
                "sport, court_id, start_at, end_at, notes"
            )
            query = query.eq("tenant_id", main_module.NTC_TENANT_ID)
            query = query.eq("user_id", user_id)
            query = query.eq("status", "confirmed")
            query = query.gte("start_at", now_utc)
            result = query.order("start_at").limit(5).execute()
        except Exception as exc:
            print(
                f"[ntc-upcoming] Supabase query failed for verified "
                f"call_sid={call_sid}: {exc}"
            )
            raise HTTPException(
                status_code=503,
                detail="Rezervácie momentálne nie je možné overiť.",
            ) from exc

        message, bookings = _format_bookings(
            result.data or [], main_module._parse_iso_to_utc
        )
        print(
            f"[ntc-upcoming] Verified call_sid={call_sid}, "
            f"customer='{customer_name or ''}', returned_bookings={len(bookings)}"
        )
        return {
            "status": "success",
            "has_bookings": bool(bookings),
            "count": len(bookings),
            "message": message,
            "bookings": bookings,
        }
