import os
import re
import time
import asyncio
import unicodedata
import json
import base64
from typing import Optional
import httpx
from rapidfuzz import fuzz, process
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from openai import AzureOpenAI

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

# --- CUSTOM PIPELINE (Deepgram + GPT-4.1 + Google TTS) ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY", "").strip()
GOOGLE_TTS_VOICE = os.getenv("GOOGLE_TTS_VOICE", "sk-SK-Wavenet-A").strip()
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "azure").strip()  # "azure" alebo "custom"

DEEPGRAM_KEYWORDS = [
    "Margherita", "Hawaii", "Salami", "Quattro+Formaggi", "Bresaola",
    "Prosciutto", "Panchetta", "Gorgonzola", "Pepperoni", "Carbonara",
    "Ventricina", "Mortadella", "Pistacio", "Cola", "Kofola", "pizza", "pizzu",
]

# --- AZURE VOICE LIVE KONFIG ---
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "swedencentral").strip()
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "").strip()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://telio-openai-sk-01.openai.azure.com/").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip()
AZURE_AI_SERVICES_ENDPOINT = os.getenv("AZURE_AI_SERVICES_ENDPOINT", "").strip()
_ws_base = AZURE_AI_SERVICES_ENDPOINT.rstrip("/").replace("https://", "wss://") if AZURE_AI_SERVICES_ENDPOINT else f"wss://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com"
AZURE_VOICE_LIVE_WS_URL = f"{_ws_base}/voice-live/realtime?api-version=2025-10-01&model={AZURE_OPENAI_DEPLOYMENT}&api-key={AZURE_SPEECH_KEY}"

_prompt_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
PIZZA_SYSTEM_PROMPT_BASE = open(_prompt_path, encoding="utf-8").read().strip()

PIZZA_MENU_FALLBACK = """
MENU (fallback):
| Pizza | Cena |
|-------|------|
| Margherita | 6.50€ |
| Salami | 7.50€ |
| Hawaii | 7.90€ |
| Pepperoni | 9.90€ |
| Quattro Formaggi | 9.50€ |"""

PIZZA_TOOLS = [
    {
        "type": "function",
        "name": "over_adresu",
        "description": "Overí či adresa je v zóne doručenia. Volaj pred uložením objednávky.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Iba názov ulice BEZ čísla domu"},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "uloz_objednavku",
        "description": "Uloží potvrdenú objednávku pizze do databázy. Volaj až po potvrdení zákazníkom.",
        "parameters": {
            "type": "object",
            "properties": {
                "pizza_type": {"type": "string", "description": "Typ pizze (napr. Margherita, Diavola)"},
                "upsell": {"type": "string", "description": "Doplnkový produkt alebo 'ziadny'"},
                "address": {"type": "string", "description": "Adresa doručenia"},
                "phone_number": {"type": "string", "description": "Telefónne číslo zákazníka — vyplní sa automaticky, netreba pýtať"},
                "total_price": {"type": "number", "description": "Celková cena vrátane dovozu"},
            },
            "required": ["pizza_type", "upsell", "address", "total_price"],
        },
    },
    {
        "type": "function",
        "name": "ukonci_hovor",
        "description": "Ukončí telefonický hovor. Volaj po rozlúčke so zákazníkom.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
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
print(f"PORT: {os.getenv('PORT', 'nenastaveny')}")
print(f"CORE_SUPABASE_URL nastavene: {'ano' if bool(SUPABASE_URL) else 'nie'}")
print(f"CORE_SUPABASE_SERVICE_ROLE_KEY nastavene: {'ano' if bool(SUPABASE_KEY) else 'nie'}")
print(f"TENANT_ID nastavene: {'ano' if bool(TENANT_ID) else 'nie'}")
print(f"CORS_ALLOW_ORIGINS: {CORS_ALLOW_ORIGINS}")
print(f"AZURE_SPEECH_KEY nastavene: {'ano' if bool(AZURE_SPEECH_KEY) else 'nie'}")
print(f"AZURE_OPENAI_KEY nastavene: {'ano' if bool(AZURE_OPENAI_KEY) else 'nie'}")
print(f"AZURE_SPEECH_REGION: {AZURE_SPEECH_REGION}")
print(f"AZURE_OPENAI_DEPLOYMENT: {AZURE_OPENAI_DEPLOYMENT}")
print(f"AZURE_AI_SERVICES_ENDPOINT: {AZURE_AI_SERVICES_ENDPOINT or '(nenastaveny, pouziva sa regionalny)'}")
print(f"AZURE_VOICE_LIVE_WS_URL: {_ws_base}/voice-live/realtime?api-version=2025-10-01&model={AZURE_OPENAI_DEPLOYMENT}&api-key=***")
print(f"AZURE_SPEECH_KEY dlzka: {len(AZURE_SPEECH_KEY)} znakov")
print(f"AZURE_OPENAI_KEY dlzka: {len(AZURE_OPENAI_KEY)} znakov")
print(f"audioop dostupny: {'ano' if audioop else 'nie'}")
print(f"PIPELINE_VERSION: {PIPELINE_VERSION}")
print(f"DEEPGRAM_API_KEY nastavene: {'ano' if bool(DEEPGRAM_API_KEY) else 'nie'}")
print(f"GOOGLE_TTS_API_KEY nastavene: {'ano' if bool(GOOGLE_TTS_API_KEY) else 'nie'}")
print(f"GOOGLE_TTS_VOICE: {GOOGLE_TTS_VOICE}")
print("----------------------")

try:
    # Inicializacia Supabase klienta
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Chyba pri inicializacii Supabase: {e}")
    supabase = None


class SearchStreetRequest(BaseModel):
    query: str


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
    if not supabase:
        return ""

    try:
        result = (
            supabase.table("menu_items")
            .select("name, price, ingredients, allergens")
            .order("price")
            .execute()
        )
        print(f"[menu] načítané položky ({len(result.data)}): {[i['name'] for i in result.data]}")
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
    if not supabase:
        return raw_address, 0

    try:
        result = supabase.table("streets").select("name").execute()
        if not result.data:
            return raw_address, 0

        street_names = [s["name"] for s in result.data]
        street_names_lower = {name.lower(): name for name in street_names}

        address = raw_address.strip()
        parts = address.rsplit(maxsplit=1)

        candidates = []
        if len(parts) == 2:
            candidates.append((parts[0], parts[1]))  # bez posledného slova (číslo domu)
        candidates.append((address, ""))  # celá adresa

        for street_part, house_number in candidates:
            results = process.extractOne(
                street_part.lower(), street_names_lower.keys(), score_cutoff=45
            )
            matches = [results[0]] if results else []
            if matches:
                matched_name = street_names_lower[matches[0]]
                matched_address = f"{matched_name} {house_number}".strip() if house_number else matched_name
                print(f"Address match: '{address}' -> '{matched_address}'")
                return matched_address, 1

        # Žiadna zhoda — vráť top výsledok ak aspoň niečo nájde (pre debug)
        top = process.extractOne(address.lower(), street_names_lower.keys())
        print(f"Address no match: '{address}' | best candidate: {top}")
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
    Twilio voice webhook — vyberá pipeline podľa PIPELINE_VERSION env premennej.
    'azure'  → /ws/voice  (Azure Voice Live)
    'custom' → /ws/voice-v2 (Deepgram + GPT-4.1 + Google TTS)
    """
    host = request.headers.get("host", "")

    if PIPELINE_VERSION == "custom" and DEEPGRAM_API_KEY and AZURE_OPENAI_KEY and GOOGLE_TTS_API_KEY:
        ws_endpoint = f"wss://{host}/ws/voice-v2"
        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_endpoint}">
            <Parameter name="phone_number" value="{{{{From}}}}" />
        </Stream>
    </Connect>
</Response>'''
    elif AZURE_SPEECH_KEY and AZURE_OPENAI_KEY:
        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/ws/voice">
            <Parameter name="phone_number" value="{{{{From}}}}" />
        </Stream>
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



@app.post("/api/search-street")
async def search_street(body: SearchStreetRequest):
    """Fuzzy vyhľadávanie ulice — cachuje ulice z DB na 5 minút, vracia max 2 zhody so skóre ≥ 55."""
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
    Načíta menu z DB pri každom hovore. Kľúče idú len v WS headers.
    """
    menu_text = format_menu_from_db(TENANT_ID)
    instructions = PIZZA_SYSTEM_PROMPT_BASE + "\n\n" + (menu_text if menu_text else PIZZA_MENU_FALLBACK)
    return {
        "type": "session.update",
        "session": {
            "turn_detection": {
                "type": "azure_semantic_vad",
                "threshold": 0.7,           # vyšší = menej citlivý na šum (default ~0.5)
                "silence_duration_ms": 800, # čaká dlhšie než ukončí turn
                "prefix_padding_ms": 300,   # ignoruje kratšie zvuky pred rečou
            },
            "input_audio_format": "pcm16",
            "output_audio_format": "g711_ulaw",  # 8kHz mulaw — priamo Twilio formát, bez konverzie
            "input_audio_transcription": {
                "model": "azure-speech",
                "language": "sk-SK",
                "phrase_list": "Margherita, Hawaii, Salami, Quattro Formaggi, Bresaola, Prosciutto Parmigiano, Panchetta Gorgonzola, Pepperoni, Milano Napoli, Ventricina, Carbonara, Mortadella, Pistacio, pizza, pizzu, objednávka, adresa, ulica, Cola, Kofola, Levoča, doručenie, číslo",
            },
            "input_audio_noise_reduction": {
                "type": "azure_deep_noise_suppression"
            },
            # Aktívny hlas: natívny slovenský (ženský)
            "voice": {
                "name": "sk-SK-ViktoriaNeural",
                "type": "azure-standard",
                "rate": "1.1",
            },
            # Záložný hlas: český (test)
            # "voice": {
            #     "name": "cs-CZ-VlastaNeural",
            #     "type": "azure-standard",
            #     "rate": "1.1",
            # },
            # Záložný hlas: natívny slovenský (mužský)
            # "voice": {
            #     "name": "sk-SK-LukasNeural",
            #     "type": "azure-standard",
            #     "rate": "1.1",
            # },
            # Záložný hlas: multilingual HD (americký prízvuk)
            # "voice": {
            #     "name": "en-US-Andrew:DragonHDLatestNeural",
            #     "type": "azure-standard",
            #     "temperature": 0.8,
            # },
            "instructions": instructions,
            "tools": PIZZA_TOOLS,
            "tool_choice": "auto",
        },
    }


async def handle_tool_call(tool_name: str, tool_args: dict, phone_number: str, transcript_lines: list = []) -> str:
    """Vykoná tool call od Azure agenta."""
    if tool_name == "over_adresu":
        raw_address = tool_args.get("query", tool_args.get("address", ""))
        matched_address, confidence = match_street(raw_address, TENANT_ID)
        if confidence > 0:
            print(f"[tool] over_adresu OK: '{raw_address}' -> '{matched_address}'")
            return json.dumps({"found": True, "best_match": matched_address})
        else:
            print(f"[tool] over_adresu NOT FOUND: '{raw_address}'")
            return json.dumps({"found": False, "message": "Adresa nie je v zóne doručenia."})

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
                "total_price": float(tool_args.get("total_price", 0)),
                "delivery_address": matched_address,
                "address_raw": raw_address,
                "address_confidence": confidence,
                "upsell_offered": tool_args.get("upsell", "ziadny") != "ziadny",
                "upsell_item": tool_args.get("upsell") or None,
                "upsell_accepted": tool_args.get("upsell", "ziadny") != "ziadny",
                "notes": "\n".join(transcript_lines) or None,
                "status": "NEW",
            }
            supabase.table("pizza_orders").insert(order_data).execute()
            print(f"[tool] uloz_objednavku OK: {order_data}")
            return json.dumps({"status": "success", "message": "Objednávka uložená."})
        except Exception as e:
            print(f"[tool] uloz_objednavku chyba: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    if tool_name == "ukonci_hovor":
        return json.dumps({"status": "ok"})

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
    transcript_lines: list = []    # akumulovaný prepis hovoru

    try:
        # Otvor spojenie s Azure Voice Live API
        azure_headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "X-OpenAI-Api-Key": AZURE_OPENAI_KEY,
        }
        print(f"[ws/voice] Pripájam sa na Azure: {_ws_base}/voice-live/realtime?api-version=2025-10-01&model={AZURE_OPENAI_DEPLOYMENT}&api-key=***")
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
            greeting_sent = False
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
                    # Pošli pozdrav až keď je stream_sid nastavený — audio bude mať kam ísť
                    if not greeting_sent:
                        await azure_ws.send(json.dumps({"type": "response.create"}))
                        print("[ws/voice] response.create odoslany po start evente")
                        greeting_sent = True

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
            nonlocal pending_tool_calls, transcript_lines
            while True:
                try:
                    raw = await azure_ws.recv()
                except Exception as e:
                    print(f"[ws/voice] Azure WS ukončil: {e}")
                    return

                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "response.audio.delta":
                    # Azure posiela g711_ulaw 8kHz — priamo Twilio formát, bez konverzie
                    delta_b64 = msg.get("delta", "")
                    if delta_b64 and stream_sid:
                        await websocket.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": delta_b64},
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
                    result_str = await handle_tool_call(tool_name, tool_args, phone_number, transcript_lines)

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

                    if tool_name == "ukonci_hovor":
                        await asyncio.sleep(3)  # nechaj agenta dohovoriť rozlúčku
                        print("[ws/voice] ukonci_hovor — zatvaram spojenie")
                        await websocket.close()
                        return

                elif msg_type == "response.audio_transcript.done":
                    text = msg.get("transcript", "").strip()
                    if text:
                        transcript_lines.append(f"Agent: {text}")
                        print(f"[transcript] Agent: {text}")

                elif msg_type == "conversation.item.input_audio_transcription.completed":
                    text = msg.get("transcript", "").strip()
                    if text and len(text.split()) >= 2:  # ignoruj jednoslovné halucinécie
                        transcript_lines.append(f"Zákazník: {text}")
                        print(f"[transcript] Zákazník: {text}")
                    elif text:
                        print(f"[transcript/skip] Zákazník (prilis kratke): {text}")

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


# ---------------------------------------------------------------------------
# /ws/voice-v2: Custom pipeline — Deepgram STT + GPT-4.1 (Azure) + ElevenLabs TTS
# ---------------------------------------------------------------------------

# GPT-4.1 tools vo formáte openai SDK (type=function, function={name, description, parameters})
PIZZA_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        },
    }
    for t in PIZZA_TOOLS
]


def _build_deepgram_url() -> str:
    keywords_qs = "&".join(f"keyterm={k}" for k in DEEPGRAM_KEYWORDS)
    return (
        f"wss://api.deepgram.com/v1/listen"
        f"?model=nova-3"
        f"&language=sk"
        f"&encoding=mulaw"
        f"&sample_rate=8000"
        f"&channels=1"
        f"&punctuate=true"
        f"&interim_results=true"
        f"&endpointing=300"
        f"&utterance_end_ms=1500"
        f"&{keywords_qs}"
    )


# Regex pre delenie textu na vety (Google TTS — paralelné generovanie)
_SENTENCE_RE = re.compile(r'(?<=[.?!;])\s+')


async def _google_tts(text: str) -> bytes:
    """Zavolá Google Cloud TTS REST API, vráti mulaw 8kHz audio bytes."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}",
            json={
                "input": {"text": text},
                "voice": {"languageCode": "sk-SK", "name": GOOGLE_TTS_VOICE},
                "audioConfig": {
                    "audioEncoding": "MULAW",
                    "sampleRateHertz": 8000,
                    "speakingRate": 1.05,
                    "pitch": 2.0,
                },
            },
        )
    if resp.status_code != 200:
        print(f"[google-tts] HTTP {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
    return base64.b64decode(resp.json().get("audioContent", ""))


async def _tts_to_twilio(text: str, twilio_ws: WebSocket, stream_sid: str) -> None:
    """Rozdelí text na vety, vygeneruje audio paralelne, pošle do Twilia v poradí."""
    if not text.strip() or not stream_sid:
        return
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        sentences = [text.strip()]
    print(f"[google-tts] {len(sentences)} viet → TTS")
    audio_chunks = await asyncio.gather(*[_google_tts(s) for s in sentences])
    for audio_bytes in audio_chunks:
        if audio_bytes:
            await twilio_ws.send_text(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": base64.b64encode(audio_bytes).decode()},
            }))


async def _gpt_stream_and_tts(
    messages: list,
    twilio_ws: WebSocket,
    stream_sid: str,
    phone_number: str,
    transcript_lines: list,
    gpt_lock: asyncio.Lock,
) -> list:
    """
    Zavolá GPT-4.1 (Azure) streaming, akumuluje text a generuje TTS cez Google Cloud.
    Pre každú vetu volá Google TTS paralelne — nižšia latencia.
    gpt_lock zabraňuje súbežným GPT+TTS volaniam.
    """
    try:
        async with gpt_lock:
            azure_client = AzureOpenAI(
                api_key=AZURE_OPENAI_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version="2025-01-01-preview",
            )

            async def _collect_gpt(gpt_stream) -> tuple[str, dict]:
                """Akumuluje GPT odpoveď, vracia (text, tool_calls_acc)."""
                assistant_text_chunks: list[str] = []
                tool_calls_acc: dict = {}

                for chunk in gpt_stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue
                    if delta.content:
                        assistant_text_chunks.append(delta.content)
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {"id": "", "name": "", "args": ""}
                            if tc.id:
                                tool_calls_acc[idx]["id"] = tc.id
                            if tc.function and tc.function.name:
                                tool_calls_acc[idx]["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_calls_acc[idx]["args"] += tc.function.arguments

                return "".join(assistant_text_chunks).strip(), tool_calls_acc

            # Iteratívna slučka pre tool cally
            while True:
                stream = azure_client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=messages,
                    tools=PIZZA_TOOLS_OPENAI,
                    tool_choice="auto",
                    stream=True,
                )

                assistant_content, tool_calls_acc = await _collect_gpt(stream)

                if assistant_content:
                    transcript_lines.append(f"Agent: {assistant_content}")
                    print(f"[v2/transcript] Agent: {assistant_content}")
                    await _tts_to_twilio(assistant_content, twilio_ws, stream_sid)

                # Žiadne tool cally — bežný turn hotový
                if not tool_calls_acc:
                    if assistant_content:
                        messages.append({"role": "assistant", "content": assistant_content})
                    return messages

                # Spracuj tool cally
                tool_call_items = []
                for idx in sorted(tool_calls_acc.keys()):
                    tc = tool_calls_acc[idx]
                    tool_call_items.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["args"]},
                    })

                messages.append({
                    "role": "assistant",
                    "content": assistant_content or None,
                    "tool_calls": tool_call_items,
                })

                should_hangup = False
                for tc_item in tool_call_items:
                    tool_name = tc_item["function"]["name"]
                    try:
                        tool_args = json.loads(tc_item["function"]["arguments"])
                    except Exception:
                        tool_args = {}

                    print(f"[v2/tool_call] name={tool_name} args={tool_args}")
                    result_str = await handle_tool_call(tool_name, tool_args, phone_number, transcript_lines)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_item["id"],
                        "content": result_str,
                    })

                    if tool_name == "ukonci_hovor":
                        should_hangup = True

                if should_hangup:
                    # Rozlúčkový turn — len ak GPT ešte nepovedal rozlúčku v assistant_content
                    if not assistant_content:
                        stream2 = azure_client.chat.completions.create(
                            model=AZURE_OPENAI_DEPLOYMENT,
                            messages=messages,
                            stream=True,
                        )
                        farewell, _ = await _collect_gpt(stream2)
                        if farewell:
                            await _tts_to_twilio(farewell, twilio_ws, stream_sid)
                    await asyncio.sleep(4)
                    await twilio_ws.close()
                    return messages

                # Pokračuj v slučke (GPT dostane výsledky toolov)

    except Exception as e:
        import traceback
        print(f"[v2/gpt_tts] CHYBA: {e}")
        traceback.print_exc()
        return messages


@app.websocket("/ws/voice-v2")
async def ws_voice_v2(websocket: WebSocket):
    """
    Custom pipeline: Twilio mulaw 8kHz → Deepgram STT → GPT-4.1 Azure → Google Cloud TTS → Twilio.
    Google TTS: REST API, mulaw 8kHz, paralelné generovanie viet.
    """
    await websocket.accept()
    print("[ws/voice-v2] Twilio pripojeny")

    stream_sid: Optional[str] = None
    phone_number: str = ""
    transcript_lines: list = []
    gpt_lock = asyncio.Lock()  # len jeden GPT+TTS turn naraz

    # Konverzačný stav pre GPT
    menu_text = format_menu_from_db(TENANT_ID)
    system_instructions = PIZZA_SYSTEM_PROMPT_BASE + "\n\n" + (menu_text if menu_text else PIZZA_MENU_FALLBACK)
    messages: list = [{"role": "system", "content": system_instructions}]

    deepgram_ws = None

    try:
        # --- Otvoriť Deepgram WebSocket ---
        dg_url = _build_deepgram_url()
        print(f"[ws/voice-v2] Pripájam Deepgram: {dg_url}")
        try:
            deepgram_ws = await websockets.connect(
                dg_url,
                additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                max_size=10 * 1024 * 1024,
            )
        except websockets.exceptions.InvalidStatus as e:
            print(f"[ws/voice-v2] Deepgram HTTP {e.response.status_code}: {e.response.headers}")
            try:
                body = await e.response.read()
                print(f"[ws/voice-v2] Deepgram response body: {body.decode(errors='replace')}")
            except Exception:
                pass
            raise
        print("[ws/voice-v2] Deepgram pripojený")

        async def twilio_to_deepgram():
            """Číta audio z Twilia a posiela priamo do Deepgram (mulaw 8kHz, bez konverzie)."""
            nonlocal stream_sid, phone_number
            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    print("[ws/voice-v2] Twilio odpojil")
                    try:
                        await deepgram_ws.send(json.dumps({"type": "CloseStream"}))
                    except Exception:
                        pass
                    return
                msg = json.loads(raw)
                event = msg.get("event")

                if event == "start":
                    start_data = msg["start"]
                    stream_sid = start_data.get("streamSid", "")
                    phone_number = (
                        start_data.get("customParameters", {}).get("phone_number", "")
                        or start_data.get("from", "")
                    )
                    print(f"[ws/voice-v2] stream started sid={stream_sid} phone={phone_number}")

                    # Okamžitý pozdrav priamo cez Google TTS — bez GPT cold start
                    GREETING = "Dobrý deň, Pizzeria Sicilia, akú pizzu si želáte?"
                    try:
                        await _tts_to_twilio(GREETING, websocket, stream_sid)
                        messages.append({"role": "assistant", "content": GREETING})
                        transcript_lines.append(f"Agent: {GREETING}")
                        print("[ws/voice-v2] pozdrav odoslaný")
                    except Exception as e:
                        print(f"[ws/voice-v2] chyba pozdrav TTS: {e}")
                        # Fallback: nechaj GPT vygenerovať pozdrav
                        asyncio.ensure_future(
                            _gpt_stream_and_tts(messages, websocket, stream_sid, phone_number, transcript_lines, gpt_lock)
                        )

                elif event == "media":
                    mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                    try:
                        await deepgram_ws.send(mulaw_bytes)
                    except Exception as e:
                        print(f"[ws/voice-v2] Deepgram send error: {e}")
                        return

                elif event == "stop":
                    print("[ws/voice-v2] Twilio stream stop")
                    try:
                        await deepgram_ws.send(json.dumps({"type": "CloseStream"}))
                    except Exception:
                        pass
                    return

        async def deepgram_to_gpt():
            """Číta transkript z Deepgram a spúšťa GPT+TTS pipeline.

            interim_results=true → dostávame aj interim správy, reagujeme len na:
              - is_final=true  (Deepgram uzavrel utterance cez endpointing)
              - UtteranceEnd   (Deepgram uzavrel utterance cez utterance_end_ms)
            Akumulujeme is_final časti do buffra, GPT spustíme až pri UtteranceEnd.
            """
            nonlocal messages
            utterance_buffer: list[str] = []

            while True:
                try:
                    raw = await deepgram_ws.recv()
                except Exception as e:
                    print(f"[ws/voice-v2] Deepgram WS ukončil: {e}")
                    return

                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "Results":
                    if not msg.get("is_final", False):
                        continue  # ignoruj interim výsledky
                    try:
                        transcript = msg["channel"]["alternatives"][0]["transcript"].strip()
                    except (KeyError, IndexError):
                        continue
                    if transcript:
                        print(f"[v2/transcript/partial] Zákazník: {transcript}")
                        utterance_buffer.append(transcript)

                elif msg_type == "UtteranceEnd":
                    if not utterance_buffer:
                        continue
                    full_transcript = " ".join(utterance_buffer).strip()
                    utterance_buffer.clear()

                    if not full_transcript:
                        continue

                    print(f"[v2/transcript] Zákazník: {full_transcript}")
                    transcript_lines.append(f"Zákazník: {full_transcript}")
                    messages.append({"role": "user", "content": full_transcript})

                    async def _run_gpt():
                        try:
                            await _gpt_stream_and_tts(messages, websocket, stream_sid, phone_number, transcript_lines, gpt_lock)
                        except Exception as e:
                            import traceback
                            print(f"[v2/gpt_tts] ensure_future chyba: {e}")
                            traceback.print_exc()
                    asyncio.ensure_future(_run_gpt())

        await asyncio.gather(
            twilio_to_deepgram(),
            deepgram_to_gpt(),
        )

    except Exception as e:
        print(f"[ws/voice-v2] chyba: {e}")
    finally:
        if deepgram_ws:
            try:
                await deepgram_ws.close()
            except Exception:
                pass
        print("[ws/voice-v2] spojenie ukončené")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
