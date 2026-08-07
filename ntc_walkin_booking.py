import datetime as dt
import json
from typing import Optional
from zoneinfo import ZoneInfo

BRATISLAVA = ZoneInfo("Europe/Bratislava")
SPORT_LIMITS = {"badminton": 10, "squash": 4, "tennis": 8, "tennis-clay": 2}


def parse_local_start(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(str(value or "").strip().replace("Z", "+00:00"))
    return parsed.replace(tzinfo=BRATISLAVA) if parsed.tzinfo is None else parsed.astimezone(BRATISLAVA)


def validate_slot(
    sport: str,
    start_iso: str,
    duration: int,
    now: Optional[dt.datetime] = None,
) -> tuple[dt.datetime, dt.datetime]:
    if sport not in SPORT_LIMITS:
        raise ValueError("Nepodporovaný šport.")
    if duration < 30 or duration > 180 or duration % 30:
        raise ValueError("Trvanie musí byť 30 až 180 minút po 30-minútových blokoch.")
    start = parse_local_start(start_iso)
    current = (now or dt.datetime.now(BRATISLAVA)).astimezone(BRATISLAVA)
    if start < current:
        raise ValueError("Termín je v minulosti.")
    if start.date() > (current + dt.timedelta(days=14)).date():
        raise ValueError("Rezervácia kurtov je možná len na najbližších 14 dní.")
    end = start + dt.timedelta(minutes=duration)
    closing_hour = 22 if start.weekday() < 5 else 21
    opening = start.replace(hour=7, minute=0, second=0, microsecond=0)
    closing = start.replace(hour=closing_hour, minute=0, second=0, microsecond=0)
    if start < opening or end > closing or end.date() != start.date():
        raise ValueError(f"Termín je mimo otváracích hodín 07:00-{closing_hour}:00.")
    return start, end


def court_name(court_id: str) -> str:
    number = court_id.rsplit("-", 1)[-1]
    return f"Vonkajší kurt {number}" if court_id.startswith("tennis-clay-") else f"Kurt {number}"


def _busy_courts(main_module, start: dt.datetime, end: dt.datetime) -> set[str]:
    return main_module._get_ntc_busy_courts(
        start.astimezone(dt.timezone.utc),
        end.astimezone(dt.timezone.utc),
    )


def check_availability(main_module, sport: str, start_iso: str, duration: int) -> dict:
    start, end = validate_slot(sport, start_iso, duration)
    busy = _busy_courts(main_module, start, end)
    free = [
        f"{sport}-{index}"
        for index in range(1, SPORT_LIMITS[sport] + 1)
        if f"{sport}-{index}" not in busy
    ]
    return {
        "status": "available" if free else "busy",
        "free_courts": free,
        "free_court_names": [court_name(item) for item in free],
    }


def create_booking(
    main_module,
    sport: str,
    court_id: str,
    customer_name: str,
    customer_phone: str,
    start_iso: str,
    duration: int,
) -> dict:
    start, end = validate_slot(sport, start_iso, duration)
    valid_courts = {f"{sport}-{index}" for index in range(1, SPORT_LIMITS[sport] + 1)}
    if court_id not in valid_courts:
        raise ValueError("Neplatný kurt.")
    if len(customer_name.strip()) < 2:
        raise ValueError("Chýba meno zákazníka.")
    if court_id in _busy_courts(main_module, start, end):
        raise ValueError("Vybraný kurt už nie je voľný.")

    booking = {
        "tenant_id": main_module.NTC_TENANT_ID,
        "court_id": court_id,
        "sport": sport,
        "customer_name": customer_name.strip(),
        "customer_phone": main_module._normalize_phone(customer_phone or ""),
        "start_at": start.astimezone(dt.timezone.utc).isoformat(),
        "end_at": end.astimezone(dt.timezone.utc).isoformat(),
        "status": "confirmed",
        "notes": json.dumps(
            {"courtId": court_id, "source": "voice-assistant-walkin", "notes": "Telefonická rezervácia návštevníka"},
            ensure_ascii=False,
        ),
    }
    result = main_module.supabase.table("bookings").insert(booking).execute()
    if not result.data:
        raise RuntimeError("Rezerváciu sa nepodarilo zapísať.")
    return {"status": "success", "court_id": court_id, "court_name": court_name(court_id)}
