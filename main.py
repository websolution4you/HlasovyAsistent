import os
import time
import asyncio
import unicodedata
import json
import base64
import struct
from dataclasses import dataclass, field
from typing import Dict, Optional
from rapidfuzz import fuzz, process
import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

try:
    import audioop
except ImportError:
    try:
        import audioop_lts as audioop  # type: ignore
    except ImportError:
        audioop = None  # type: ignore

# Nacitanie environment premennych (uzitocne pre lokalny vyvoj)
load_dotenv()

app = FastAPI(title="Pizza Sicilia Voice Assistant")


def _parse_cors_origins() -> list[str]:
    """
    Nacita povolene originy pre web frontend.
    Defaultne povoli localhosty pre lokalny vyvoj.
    V Renderi nastav CORS_ALLOW_ORIGINS ako ciarkou oddeleny zoznam.
    """
    raw = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    ).strip()
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


CORS_ALLOW_ORIGINS = _parse_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_TWILIO_VOICE_MESSAGE = (
    "Dobry den, dovolali ste sa do Pizza Sicilia. "
    "Nasa hlasova objednavkova linka sa prave pripravuje. "
    "Skuste prosim zavolat o chvilu neskor."
)

# --- KONFIGURACIA SUPABASE ---
SUPABASE_URL = os.getenv("CORE_SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("CORE_SUPABASE_SERVICE_ROLE_KEY", "").strip()

TENANT_ID = os.getenv("TENANT_ID", "").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
FINALIZE_DELAY_SECONDS = float(os.getenv("FINALIZE_DELAY_SECONDS", "2.0"))

# --- AZURE VOICE LIVE KONFIG ---
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "swedencentral").strip()
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "").strip()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://telio-openai-sk-01.openai.azure.com/").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip()
AZURE_VOICE_LIVE_WS_URL = (
    f"wss://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/voice-live/realtime"
    f"?api-version=2025-10-01"
    f"&model={AZURE_OPENAI_DEPLOYMENT}"
    f"&api-key={AZURE_SPEECH_KEY}"
)

PIZZA_SYSTEM_PROMPT = """Si priateľský hlasový asistent Pizza Sicilia v Bratislave. Prijímaš telefonické objednávky pizze.
Tvoj postup:
1. Privítaj zákazníka
2. Spýtaj sa čo si želá objednať (ponúkni aj nápoje/dezerty ako upsell)
3. Spýtaj sa na adresu doručenia
4. Zopakuj objednávku a potvrd ju
5. Po potvrdení zavolaj funkciu uloz_objednavku

Buď stručný, hovor krátke vety (max 2-3 vety naraz). Ceny: Margherita 6.90€, Šunková 7.50€, Diavola 8.20€, Quattro Formaggi 8.90€. Dovoz zadarmo nad 15€, inak 2€."""

PIZZA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "uloz_objednavku",
            "description": "Uloží potvrdenú objednávku pizze do databázy. Volaj až po potvrdení zákazníkom.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pizza_type": {"type": "string", "description": "Typ pizze (napr. Margherita, Diavola)"},
                    "upsell": {"type": "string", "description": "Doplnkový produkt alebo 'ziadny'"},
                    "address": {"type": "string", "description": "Adresa doručenia"},
                    "phone_number": {"type": "string", "description": "Telefónne číslo zákazníka"},
                },
                "required": ["pizza_type", "upsell", "address", "phone_number"],
            },
        },
    }
]

if not SUPABASE_URL or not SUPABASE_KEY:
    print("--- CHYBA KONFIGURACIE ---")
    if not SUPABASE_URL:
        print("Chyba premenna: CORE_SUPABASE_URL")
    if not SUPABASE_KEY:
        print("Chyba premenna: CORE_SUPABASE_SERVICE_ROLE_KEY")
    print("--------------------------")
if not TENANT_ID:
    print("VAROVANIE: TENANT_ID nie je nastavene")

print("--- STARTUP KONFIG ---")
print(f"ELEVENLABS_API_KEY nastavene: {'ano' if bool(ELEVENLABS_API_KEY) else 'nie'}")
print(f"FINALIZE_DELAY_SECONDS: {FINALIZE_DELAY_SECONDS}")
print(f"PORT: {os.getenv('PORT', 'nenastaveny')}")
print(f"CORE_SUPABASE_URL nastavene: {'ano' if bool(SUPABASE_URL) else 'nie'}")
print(f"CORE_SUPABASE_SERVICE_ROLE_KEY nastavene: {'ano' if bool(SUPABASE_KEY) else 'nie'}")
print(f"TENANT_ID nastavene: {'ano' if bool(TENANT_ID) else 'nie'}")
print(f"CORS_ALLOW_ORIGINS: {CORS_ALLOW_ORIGINS}")
print(f"AZURE_SPEECH_KEY nastavene: {'ano' if bool(AZURE_SPEECH_KEY) else 'nie'}")
print(f"AZURE_OPENAI_KEY nastavene: {'ano' if bool(AZURE_OPENAI_KEY) else 'nie'}")
print(f"AZURE_SPEECH_REGION: {AZURE_SPEECH_REGION}")
print(f"AZURE_OPENAI_DEPLOYMENT: {AZURE_OPENAI_DEPLOYMENT}")
print(f"AZURE_VOICE_LIVE_WS_URL: wss://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/voice-live/realtime?api-version=2025-10-01&model={AZURE_OPENAI_DEPLOYMENT}&api-key=***")
print(f"audioop dostupny: {'ano' if audioop else 'nie'}")
print("----------------------")

try:
    # Inicializacia Supabase klienta
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Chyba pri inicializacii Supabase: {e}")
    supabase = None


# ---------------------------------------------------------------------------
# SESSION STATE — in-memory, keyed by ElevenLabs conversation_id
# ---------------------------------------------------------------------------

@dataclass
class CallSession:
    conversation_id: str
    order_finalized: bool = False       # manage_order úspešne vykonaný (DB insert)
    call_ending: bool = False           # end_call bol spustený (hovor sa ukončuje)
    finalize_task: Optional[asyncio.Task] = None  # pending auto-hangup timer
    created_at: float = field(default_factory=time.monotonic)


_SESSIONS: Dict[str, CallSession] = {}
_SESSION_TTL = 3600  # 1 hodina — po nej sa session vyčistí (lazy)


def _cleanup_old_sessions() -> None:
    """Vyčistí sessions staršie ako SESSION_TTL. Volá sa lazy pri každom prístupe."""
    now = time.monotonic()
    stale = [cid for cid, s in _SESSIONS.items() if now - s.created_at > _SESSION_TTL]
    for cid in stale:
        _SESSIONS.pop(cid, None)


def get_or_create_session(conversation_id: str) -> CallSession:
    _cleanup_old_sessions()
    if conversation_id not in _SESSIONS:
        _SESSIONS[conversation_id] = CallSession(conversation_id=conversation_id)
    return _SESSIONS[conversation_id]


def extract_conversation_id(request: Request) -> Optional[str]:
    """Extrahuje conversation_id z ElevenLabs HTTP hlavičky alebo query parametra."""
    # ElevenLabs posiela conversation_id v hlavičke
    cid = request.headers.get("ElevenLabs-Conversation-Id") or request.headers.get("elevenlabs-conversation-id")
    if cid:
        return cid.strip()
    return None


def cancel_post_summary_finalize(session: CallSession) -> None:
    """Zruší pending auto-hangup timer ak existuje."""
    if session.finalize_task and not session.finalize_task.done():
        session.finalize_task.cancel()
        print(f"[session:{session.conversation_id}] finalize timer zruseny")
    session.finalize_task = None


async def elevenlabs_end_call_api(conversation_id: str) -> None:
    """Zavolá ElevenLabs REST API na ukončenie konverzácie."""
    if not ELEVENLABS_API_KEY:
        print(f"[session:{conversation_id}] ELEVENLABS_API_KEY nie je nastaveny — auto-hangup preskoceny")
        return
    url = f"https://api.elevenlabs.io/v1/convai/conversations/{conversation_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(url, headers={"xi-api-key": ELEVENLABS_API_KEY})
        print(f"[session:{conversation_id}] ElevenLabs end_call API -> {resp.status_code}")
    except Exception as e:
        print(f"[session:{conversation_id}] ElevenLabs end_call API chyba: {e}")


def schedule_auto_end_call(session: CallSession, delay: float = FINALIZE_DELAY_SECONDS) -> None:
    """Naplánuje asyncio task ktorý po 'delay' sekundách zavolá ElevenLabs API na hangup."""
    cancel_post_summary_finalize(session)  # zruší prípadný predchádzajúci timer

    async def _auto_hangup():
        await asyncio.sleep(delay)
        session_now = _SESSIONS.get(session.conversation_id)
        if session_now and not session_now.call_ending:
            session_now.call_ending = True
            print(f"[session:{session.conversation_id}] auto-hangup timer dobehol — volam ElevenLabs API")
            await elevenlabs_end_call_api(session.conversation_id)

    session.finalize_task = asyncio.create_task(_auto_hangup())
    print(f"[session:{session.conversation_id}] auto-hangup timer nastaveny na {delay}s")


# ---------------------------------------------------------------------------


class SearchStreetRequest(BaseModel):
    query: str


class ManageOrder(BaseModel):
    pizza_type: str
    total_price: float
    delivery_address: str
    customer_phone: Optional[str] = None
    customer_name: Optional[str] = None
    upsell_item: Optional[str] = None
    upsell_accepted: bool = False
    transcript: Optional[str] = None


ALLERGEN_MAP = {
    "1": "lepok", "2": "kôrovce", "3": "vajcia", "4": "ryby",
    "5": "arašidy", "6": "sója", "7": "mlieko", "8": "orechy",
    "9": "zeler", "10": "horčica", "11": "sezam", "12": "oxid siričitý",
    "13": "vlčí bôb", "14": "mäkkýše",
}


_STREETS_CACHE: dict = {"data": [], "tenant_id": "", "timestamp": 0.0}
_CACHE_TTL = 300  # 5 minút


def _normalize(s: str) -> str:
    """Lowercase + odstránenie diakritiky."""
    nfkd = unicodedata.normalize("NFD", s.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _street_score(query: str, street: str) -> int:
    """Vráti skóre 0–100 (max z ratio a partial_ratio cez rapidfuzz)."""
    q = _normalize(query)
    s = _normalize(street)
    return round(max(fuzz.ratio(q, s), fuzz.partial_ratio(q, s)))


def _get_streets_cached(tenant_id: str) -> list[str]:
    """Načíta ulice z DB, výsledok cachuje na 5 minút."""
    now = time.monotonic()
    if (
        _STREETS_CACHE["tenant_id"] == tenant_id
        and _STREETS_CACHE["data"]
        and now - _STREETS_CACHE["timestamp"] < _CACHE_TTL
    ):
        return _STREETS_CACHE["data"]

    result = supabase.table("streets").select("name").execute()
    streets = [row["name"] for row in result.data] if result.data else []
    _STREETS_CACHE.update({"data": streets, "tenant_id": tenant_id, "timestamp": now})
    return streets


def format_menu_from_db(tenant_id: str) -> str:
    """Načíta menu_items z DB a naformátuje ako text pre system prompt."""
    if not supabase or not tenant_id:
        return ""

    try:
        result = (
            supabase.table("menu_items")
            .select("name, price, ingredients, allergens")
            .eq("tenant_id", tenant_id)
            .order("price")
            .execute()
        )
        if not result.data:
            return ""

        pizzas = []
        drinks = []
        for item in result.data:
            name = item["name"]
            price = item["price"]
            ingredients = item.get("ingredients", "")
            allergens_raw = item.get("allergens") or []

            allergen_names = ", ".join(
                ALLERGEN_MAP.get(str(a), str(a)) for a in allergens_raw
            ) if allergens_raw else ""

            # Jednoduché rozlíšenie: ak nemá ingrediencie typické pre pizzu, je to nápoj
            if item.get("ingredients") and "salsa" not in (ingredients or "").lower() and "základ" not in (ingredients or "").lower() and price <= 5:
                drinks.append(f"| {name} | {price:.2f}€ | {ingredients} |")
            else:
                allergen_str = f" | {allergen_names}" if allergen_names else " |"
                pizzas.append(f"| {name} | {price:.2f}€ | {ingredients}{allergen_str}")

        lines = ["MENU:", "| Pizza | Cena | Ingrediencie | Alergény |", "|-------|------|-------------|----------|"]
        lines.extend(pizzas)

        if drinks:
            lines.append("")
            lines.append("NÁPOJE:")
            lines.append("| Nápoj | Cena | Popis |")
            lines.append("|-------|------|-------|")
            lines.extend(drinks)

        return "\n".join(lines)
    except Exception as e:
        print(f"Chyba pri nacitani menu: {e}")
        return ""


def match_street(raw_address: str, tenant_id: str) -> tuple[Optional[str], int]:
    """Fuzzy match adresy voči tabuľke ulíc. Vracia (matched_address, confidence 0-1).
    Podporuje čísla ako cifry aj slová (napr. 'Dlhá tri', 'Levočská dvanásť').
    """
    if not supabase or not tenant_id:
        return raw_address, 0

    try:
        result = supabase.table("streets").select("name").execute()
        if not result.data:
            return raw_address, 0

        street_names = [s["name"] for s in result.data]
        street_names_lower = {name.lower(): name for name in street_names}

        address = raw_address.strip()
        parts = address.rsplit(maxsplit=1)

        # Kandidáti na street_part + house_number:
        # 1. Posledné slovo je číslica (napr. "Dlhá 5")
        # 2. Posledné slovo nie je číslica ale adresa má viac slov (napr. "Dlhá tri")
        # 3. Celá adresa je len jeden výraz (napr. "Rozvoj")
        candidates = []
        if len(parts) == 2:
            candidates.append((parts[0], parts[1]))  # bez posledného slova
        candidates.append((address, ""))  # celá adresa bez čísla

        for street_part, house_number in candidates:
            results = process.extractOne(
                street_part.lower(), street_names_lower.keys(), score_cutoff=60
            )
            matches = [results[0]] if results else []
            if matches:
                matched_name = street_names_lower[matches[0]]
                matched_address = f"{matched_name} {house_number}".strip() if house_number else matched_name
                print(f"Address match: '{address}' -> '{matched_address}'")
                return matched_address, 1

        print(f"Address no match: '{address}'")
        return raw_address, 0
    except Exception as e:
        print(f"Chyba pri matchovani adresy: {e}")
        return raw_address, 0


@app.get("/")
async def health_check():
    """
    Jednoduchy health check endpoint pre Render.
    """
    return {"status": "online", "message": "Server bezi."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/config")
def health_config():
    """
    Bezpecny endpoint na rychlu kontrolu Render konfiguracie.
    Nevracia tajne hodnoty, iba ich dostupnost.
    """
    return {
        "status": "ok",
        "config": {
            "port_present": bool(os.getenv("PORT")),
            "supabase_url_present": bool(SUPABASE_URL),
            "supabase_key_present": bool(SUPABASE_KEY),
            "tenant_id_present": bool(TENANT_ID),
            "supabase_client_ready": supabase is not None,
            "cors_allow_origins": CORS_ALLOW_ORIGINS,
        },
    }


@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice_webhook(request: Request):
    """
    Twilio voice webhook — ak sú Azure kľúče k dispozícii, spustí Media Stream
    na náš /ws/voice WebSocket. Inak vráti fallback správu.
    """
    if AZURE_SPEECH_KEY and AZURE_OPENAI_KEY:
        host = request.headers.get("host", "")
        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/ws/voice" />
    </Connect>
</Response>'''
    else:
        message = os.getenv("TWILIO_VOICE_MESSAGE", DEFAULT_TWILIO_VOICE_MESSAGE).strip()
        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Vlasta" language="sk-SK">{message}</Say>
    <Pause length="1"/>
    <Hangup/>
</Response>'''
    return Response(content=twiml, media_type="application/xml")


@app.api_route("/twilio/fallback", methods=["GET", "POST"])
async def twilio_fallback_webhook():
    """
    Zalozny Twilio webhook, ak primarny handler zlyha.
    """
    twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Vlasta" language="sk-SK">Ospravedlnujeme sa, linka je docasne nedostupna. Skuste prosim zavolat neskor.</Say>
    <Hangup/>
</Response>'''
    return Response(content=twiml, media_type="application/xml")


@app.api_route("/twilio/status", methods=["GET", "POST"])
async def twilio_status_webhook(request: Request):
    """
    Prijima stavove zmeny hovorov od Twilia.
    """
    form_data = await request.form()
    payload = dict(form_data)
    print(f"Twilio status callback: {payload}")
    return {"status": "ok"}


@app.post("/api/prompt-config")
async def prompt_config():
    """
    ElevenLabs Server URL endpoint — volá sa pred každým hovorom.
    Vracia dynamic_variables s aktuálnym menu z DB.
    """
    menu_text = format_menu_from_db(TENANT_ID)
    return {
        "dynamic_variables": {
            "menu": menu_text if menu_text else "Menu nie je momentálne dostupné.",
        }
    }


@app.post("/api/search-street")
async def search_street(request: Request, body: SearchStreetRequest):
    """
    Fuzzy vyhľadávanie ulice podľa časti názvu (STT výstup z ElevenLabs).
    Cachuje ulice z DB na 5 minút. Vracia max 2 zhody so skóre ≥ 55.
    Ak prebieha finalizačný timer (user koriguje adresu), zruší ho.
    """
    # Ak user hovorí o adrese, zruší prípadný pending finalize timer
    cid = extract_conversation_id(request)
    if cid and cid in _SESSIONS:
        cancel_post_summary_finalize(_SESSIONS[cid])

    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase klient nie je inicializovany.")

    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Parameter query nesmie byť prázdny.")

    try:
        streets = _get_streets_cached(TENANT_ID)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chyba pri načítaní ulíc: {e}")

    if not streets:
        return {"found": False, "message": "Na túto adresu momentálne nevieme doručiť.", "suggestions": []}

    print(f"[search-street] query='{query}' streets_count={len(streets)}")

    scored = sorted(
        ({"street": s, "score": _street_score(query, s)} for s in streets),
        key=lambda x: x["score"],
        reverse=True,
    )
    top = [item for item in scored[:2] if item["score"] >= 55]

    print(f"[search-street] top_results={top}")

    if not top:
        return {"found": False, "message": "Na túto adresu momentálne nevieme doručiť.", "suggestions": []}

    return {
        "found": True,
        "best_match": top[0]["street"],
        "confidence": top[0]["score"],
        "suggestions": top,
    }


@app.post("/api/vytvor-objednavku")
async def vytvor_objednavku(request: Request):
    """
    Endpoint pre ElevenLabs agenta (manage_order tool) na ulozenie objednavky do DB.
    Idempotentný: druhý call s rovnakým conversation_id nevytvorí duplicitný záznam.
    Po úspešnom uložení spustí auto-hangup timer (FINALIZE_DELAY_SECONDS).
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase klient nie je inicializovany.")

    try:
        body = await request.json()
        print(f"[vytvor-objednavku] raw body: {body}")
        order = ManageOrder(**body)
    except Exception as e:
        print(f"[vytvor-objednavku] validacna chyba: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    # --- IDEMPOTENCIA: ak sme túto objednávku už uložili, vrátime cached success ---
    cid = extract_conversation_id(request)
    if cid:
        session = get_or_create_session(cid)
        if session.order_finalized:
            print(f"[vytvor-objednavku] session:{cid} uz finalizovana — preskakujem DB insert")
            return {"status": "success", "message": "Objednavka uspesne zapisana."}
        # Nastavíme flag PRED asyncom, aby súbežný request videl finalized=True
        session.order_finalized = True
    else:
        session = None
        print("[vytvor-objednavku] conversation_id chyba — session tracking preskoceny")

    try:
        matched_address, confidence = match_street(order.delivery_address, TENANT_ID)

        order_data = {
            "tenant_id": TENANT_ID,
            "customer_phone": order.customer_phone or "",
            "phone_raw": order.customer_phone or "",
            "customer_name": order.customer_name or "Zákazník",
            "pizza_type": order.pizza_type,
            "total_price": order.total_price,
            "delivery_address": matched_address,
            "address_raw": order.delivery_address,
            "address_confidence": confidence,
            "upsell_offered": order.upsell_item is not None,
            "upsell_item": order.upsell_item,
            "upsell_accepted": order.upsell_accepted,
            "notes": order.transcript,
            "status": "NEW",
        }

        supabase.table("pizza_orders").insert(order_data).execute()

        # --- AUTO-HANGUP: po úspešnom uložení naplánuj ukončenie hovoru ---
        if session and cid:
            schedule_auto_end_call(session, delay=FINALIZE_DELAY_SECONDS)

        return {
            "status": "success",
            "message": "Objednavka uspesne zapisana.",
        }

    except Exception as e:
        # Ak DB zápis zlyhal, resetujeme flag aby mohol byť retry
        if session:
            session.order_finalized = False
        error_msg = str(e)
        print(f"DEBUG CHYBA: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Chyba Supabase: {error_msg}")


@app.post("/api/end-call")
async def end_call(request: Request):
    """
    ElevenLabs agent tool — zavolá sa keď agent chce ukončiť hovor podľa promptu.
    Idempotentný: druhý call je no-op (guard cez call_ending flag).
    Zruší prípadný pending auto-hangup timer (aby nedošlo k dvojitému volaniu API).
    """
    cid = extract_conversation_id(request)
    if cid:
        session = get_or_create_session(cid)
        if session.call_ending:
            print(f"[end-call] session:{cid} uz ukonculavana — preskakujem")
            return {"status": "ok"}
        session.call_ending = True
        cancel_post_summary_finalize(session)
        print(f"[end-call] session:{cid} hovor sa ukonculava (agent-triggered)")
    else:
        print("[end-call] conversation_id chyba — pokracujem bez session")

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# AUDIO KONVERZIA: mulaw 8kHz (Twilio) ↔ PCM 16kHz (Azure Voice Live)
# ---------------------------------------------------------------------------

def mulaw8k_to_pcm16k(mulaw_bytes: bytes) -> bytes:
    """Konvertuje mulaw 8kHz na PCM 16kHz (linear16, little-endian)."""
    if audioop is None:
        # Bez audioop: len upsamplujeme bez ulaw dekódovania (núdzový fallback)
        return mulaw_bytes * 2
    pcm8k = audioop.ulaw2lin(mulaw_bytes, 2)        # mulaw -> PCM 16-bit 8kHz
    pcm16k, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)  # 8kHz -> 16kHz
    return pcm16k


def pcm16k_to_mulaw8k(pcm16k: bytes) -> bytes:
    """Konvertuje PCM 16kHz na mulaw 8kHz pre Twilio."""
    if audioop is None:
        return pcm16k[:len(pcm16k)//2]
    pcm8k, _ = audioop.ratecv(pcm16k, 2, 1, 16000, 8000, None)  # 16kHz -> 8kHz
    return audioop.lin2ulaw(pcm8k, 2)               # PCM -> mulaw


def build_azure_session_config(phone_number: str = "") -> dict:
    """Zostaví konfiguračnú správu pre Azure Voice Live API.
    Kľúče NIE sú tu — idú len v WS headers pri pripojení.
    """
    return {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "azure-speech",
                "language": "sk-SK",
            },
            "voice": {
                "name": "sk-SK-LukasNeural",
                "type": "azure-standard",
            },
            "instructions": PIZZA_SYSTEM_PROMPT,
            "tools": PIZZA_TOOLS,
            "tool_choice": "auto",
        },
    }


async def handle_tool_call(tool_name: str, tool_args: dict, phone_number: str) -> str:
    """Vykoná tool call od Azure agenta — insertne objednávku priamo cez Supabase."""
    if tool_name == "uloz_objednavku":
        if not tool_args.get("phone_number"):
            tool_args["phone_number"] = phone_number
        if not supabase:
            return json.dumps({"status": "error", "message": "Supabase nie je dostupný"})
        try:
            raw_address = tool_args.get("address", "")
            matched_address, confidence = match_street(raw_address, TENANT_ID)
            order_data = {
                "tenant_id": TENANT_ID,
                "customer_phone": tool_args.get("phone_number", ""),
                "phone_raw": tool_args.get("phone_number", ""),
                "customer_name": "Zákazník",
                "pizza_type": tool_args.get("pizza_type", ""),
                "total_price": 0,
                "delivery_address": matched_address,
                "address_raw": raw_address,
                "address_confidence": confidence,
                "upsell_offered": tool_args.get("upsell", "ziadny") != "ziadny",
                "upsell_item": tool_args.get("upsell") or None,
                "upsell_accepted": tool_args.get("upsell", "ziadny") != "ziadny",
                "notes": None,
                "status": "NEW",
            }
            supabase.table("pizza_orders").insert(order_data).execute()
            print(f"[tool] uloz_objednavku OK: {order_data}")
            return json.dumps({"status": "success", "message": "Objednávka uložená."})
        except Exception as e:
            print(f"[tool] uloz_objednavku chyba: {e}")
            return json.dumps({"status": "error", "message": str(e)})
    return json.dumps({"status": "error", "message": f"Neznámy tool: {tool_name}"})


# ---------------------------------------------------------------------------
# WebSocket endpoint: Twilio Media Streams ↔ Azure Voice Live API
# ---------------------------------------------------------------------------

@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    """
    Prijíma Twilio Media Stream (mulaw 8kHz) a prepája ho na Azure Voice Live API.
    Konvertuje audio oboma smermi a spracúva function calling (uloz_objednavku).
    """
    await websocket.accept()
    print("[ws/voice] Twilio pripojeny")

    stream_sid: Optional[str] = None
    phone_number: str = ""
    azure_ws = None
    pending_tool_calls: dict = {}  # call_id -> {name, args_acc}

    try:
        # Otvor spojenie s Azure Voice Live API
        azure_headers = {
            "X-OpenAI-Api-Key": AZURE_OPENAI_KEY,
        }
        # Logujeme URL bez api-key
        log_url = (
            f"wss://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/voice-live/realtime"
            f"?api-version=2025-10-01&model={AZURE_OPENAI_DEPLOYMENT}&api-key=***"
        )
        print(f"[ws/voice] Pripájam sa na Azure: {log_url}")
        try:
            azure_ws = await websockets.connect(
                AZURE_VOICE_LIVE_WS_URL,
                additional_headers=azure_headers,
                max_size=10 * 1024 * 1024,
            )
        except Exception as conn_err:
            import traceback
            print(f"[ws/voice] Azure WS pripojenie zlyhalo: {conn_err}")
            traceback.print_exc()
            raise
        print("[ws/voice] Azure Voice Live pripojeny")

        # Pošli konfiguráciu
        session_cfg = build_azure_session_config(phone_number)
        print(f"[ws/voice] Posielam session.update: {json.dumps(session_cfg)}")
        await azure_ws.send(json.dumps(session_cfg))

        async def twilio_to_azure():
            """Číta správy od Twilia, konvertuje audio a posiela do Azure."""
            nonlocal stream_sid, phone_number
            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    print("[ws/voice] Twilio odpojil")
                    return
                msg = json.loads(raw)
                event = msg.get("event")

                if event == "start":
                    stream_sid = msg["start"].get("streamSid", "")
                    phone_number = (
                        msg["start"]
                        .get("customParameters", {})
                        .get("phone_number", "")
                        or msg["start"].get("from", "")
                    )
                    print(f"[ws/voice] stream started sid={stream_sid} phone={phone_number}")
                    # Aktualizuj session s telefónom ak ho máme
                    if phone_number:
                        await azure_ws.send(json.dumps(build_azure_session_config(phone_number)))

                elif event == "media":
                    mulaw_b64 = msg["media"]["payload"]
                    mulaw_bytes = base64.b64decode(mulaw_b64)
                    pcm16k = mulaw8k_to_pcm16k(mulaw_bytes)
                    pcm_b64 = base64.b64encode(pcm16k).decode()
                    await azure_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": pcm_b64,
                    }))

                elif event == "stop":
                    print("[ws/voice] Twilio stream stop")
                    return

        async def azure_to_twilio():
            """Číta správy od Azure, konvertuje audio a posiela späť do Twilia."""
            nonlocal pending_tool_calls
            while True:
                try:
                    raw = await azure_ws.recv()
                except Exception as e:
                    print(f"[ws/voice] Azure WS ukončil: {e}")
                    return

                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "response.audio.delta":
                    # Azure posiela PCM 16kHz audio delta
                    pcm16k = base64.b64decode(msg.get("delta", ""))
                    if pcm16k:
                        mulaw8k = pcm16k_to_mulaw8k(pcm16k)
                        mulaw_b64 = base64.b64encode(mulaw8k).decode()
                        if stream_sid:
                            await websocket.send_text(json.dumps({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": mulaw_b64},
                            }))

                elif msg_type == "response.function_call_arguments.delta":
                    call_id = msg.get("call_id", "")
                    if call_id not in pending_tool_calls:
                        pending_tool_calls[call_id] = {
                            "name": msg.get("name", ""),
                            "args_acc": "",
                        }
                    pending_tool_calls[call_id]["args_acc"] += msg.get("delta", "")

                elif msg_type == "response.function_call_arguments.done":
                    call_id = msg.get("call_id", "")
                    tool_name = msg.get("name", "") or pending_tool_calls.get(call_id, {}).get("name", "")
                    args_str = msg.get("arguments", "") or pending_tool_calls.get(call_id, {}).get("args_acc", "{}")
                    pending_tool_calls.pop(call_id, None)

                    try:
                        tool_args = json.loads(args_str)
                    except Exception:
                        tool_args = {}

                    print(f"[ws/voice] tool_call name={tool_name} args={tool_args}")
                    result_str = await handle_tool_call(tool_name, tool_args, phone_number)

                    # Pošli výsledok toolcallu späť do Azure
                    await azure_ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": result_str,
                        },
                    }))
                    await azure_ws.send(json.dumps({"type": "response.create"}))

                elif msg_type == "error":
                    print(f"[ws/voice] Azure chyba: {msg}")

        # Spusti oba smery súbežne
        await asyncio.gather(twilio_to_azure(), azure_to_twilio())

    except Exception as e:
        print(f"[ws/voice] chyba: {e}")
    finally:
        if azure_ws:
            try:
                await azure_ws.close()
            except Exception:
                pass
        print("[ws/voice] spojenie ukončené")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
