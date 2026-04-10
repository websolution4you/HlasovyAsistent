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
from google import genai
from google.genai import types

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

# --- GEMINI LIVE API KONFIG ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp").strip()

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
print(f"GOOGLE_API_KEY nastavene: {'ano' if bool(GOOGLE_API_KEY) else 'nie'}")
print(f"GEMINI_MODEL: {GEMINI_MODEL}")
print(f"audioop dostupny: {'ano' if audioop else 'nie'}")
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
            "google_api_key_present": bool(GOOGLE_API_KEY),
            "gemini_model": GEMINI_MODEL,
        },
    }


@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice_webhook(request: Request):
    """
    Twilio voice webhook — smeruje na Gemini Live API endpoint.
    """
    host = request.headers.get("host", "")

    if GOOGLE_API_KEY:
        ws_endpoint = f"wss://{host}/ws/voice-gemini"
        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_endpoint}">
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
# AUDIO KONVERZIA: mulaw 8kHz (Twilio) ↔ PCM 16kHz (Gemini Live API)
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


async def handle_tool_call(tool_name: str, tool_args: dict, phone_number: str, transcript_lines: list = []) -> str:
    """Vykoná tool call od Gemini agenta."""
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
# /ws/voice-gemini: Gemini Live API pipeline
# ---------------------------------------------------------------------------

def _build_gemini_tools() -> list:
    """Konvertuje PIZZA_TOOLS na Gemini Live API formát (types.FunctionDeclaration)."""
    gemini_tools = []
    for tool in PIZZA_TOOLS:
        # Konvertuj properties na Gemini formát
        properties = {}
        for prop_name, prop_def in tool["parameters"]["properties"].items():
            properties[prop_name] = types.Schema(
                type=types.Type.STRING if prop_def["type"] == "string" else types.Type.NUMBER,
                description=prop_def.get("description", "")
            )
        
        gemini_tools.append(
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=tool["parameters"].get("required", [])
                )
            )
        )
    return gemini_tools


@app.websocket("/ws/voice-gemini")
async def ws_voice_gemini(websocket: WebSocket):
    """
    Gemini Live API pipeline: Twilio mulaw 8kHz ↔ Gemini Live API (WebSocket).
    Architektúra: jeden WebSocket, žiadny pipeline (STT+LLM+TTS v jednom).
    """
    await websocket.accept()
    print("[ws/voice-gemini] Twilio pripojený")

    stream_sid: Optional[str] = None
    phone_number: str = ""
    transcript_lines: list = []
    gemini_session = None

    try:
        # Načítaj menu a system prompt
        menu_text = format_menu_from_db(TENANT_ID)
        system_instructions = PIZZA_SYSTEM_PROMPT_BASE + "\n\n" + (menu_text if menu_text else PIZZA_MENU_FALLBACK)

        # Inicializuj Gemini klienta
        client = genai.Client(api_key=GOOGLE_API_KEY)

        # Konfigurácia Gemini Live session
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=system_instructions)]
            ),
            tools=_build_gemini_tools(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"  # Gemini Live voice
                    )
                )
            ),
        )

        print(f"[ws/voice-gemini] Pripájam sa na Gemini Live API model={GEMINI_MODEL}")
        gemini_session = await client.aio.live.connect(model=GEMINI_MODEL, config=config).__aenter__()
        print("[ws/voice-gemini] Gemini Live API pripojený")

        async def twilio_to_gemini():
            """Číta audio z Twilia (mulaw 8kHz), konvertuje na PCM 16kHz a posiela do Gemini."""
            nonlocal stream_sid, phone_number
            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    print("[ws/voice-gemini] Twilio odpojil")
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
                    print(f"[ws/voice-gemini] stream started sid={stream_sid} phone={phone_number}")

                elif event == "media":
                    # Twilio → mulaw 8kHz → PCM 16kHz → Gemini
                    mulaw_b64 = msg["media"]["payload"]
                    mulaw_bytes = base64.b64decode(mulaw_b64)
                    pcm16k = mulaw8k_to_pcm16k(mulaw_bytes)
                    
                    # Pošli do Gemini Live API
                    try:
                        await gemini_session.send(
                            types.LiveClientRealtimeInput(
                                media_chunks=[
                                    types.Blob(
                                        data=pcm16k,
                                        mime_type="audio/pcm;rate=16000"
                                    )
                                ]
                            )
                        )
                    except Exception as e:
                        print(f"[ws/voice-gemini] Gemini send error: {e}")
                        return

                elif event == "stop":
                    print("[ws/voice-gemini] Twilio stream stop")
                    return

        async def gemini_to_twilio():
            """Číta odpovede z Gemini Live API a posiela audio späť do Twilia."""
            nonlocal transcript_lines
            
            try:
                async for response in gemini_session.receive():
                    # Audio data od Gemini (PCM 16kHz)
                    if hasattr(response, 'data') and response.data:
                        # Gemini vracia PCM 16kHz → konvertuj na mulaw 8kHz pre Twilio
                        pcm16k = response.data
                        mulaw8k = pcm16k_to_mulaw8k(pcm16k)
                        mulaw_b64 = base64.b64encode(mulaw8k).decode()
                        
                        if stream_sid:
                            await websocket.send_text(json.dumps({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": mulaw_b64},
                            }))

                    # Server content (text, tool calls, atď.)
                    if hasattr(response, 'server_content') and response.server_content:
                        server_content = response.server_content
                        
                        # Model turn (text response)
                        if hasattr(server_content, 'model_turn') and server_content.model_turn:
                            for part in server_content.model_turn.parts:
                                # Text transcript
                                if hasattr(part, 'text') and part.text:
                                    text = part.text.strip()
                                    if text:
                                        transcript_lines.append(f"Agent: {text}")
                                        print(f"[gemini/transcript] Agent: {text}")
                                
                                # Function call
                                if hasattr(part, 'function_call') and part.function_call:
                                    fc = part.function_call
                                    tool_name = fc.name
                                    tool_args = dict(fc.args) if hasattr(fc, 'args') else {}
                                    
                                    print(f"[gemini/tool_call] name={tool_name} args={tool_args}")
                                    result_str = await handle_tool_call(
                                        tool_name, tool_args, phone_number, transcript_lines
                                    )
                                    
                                    # Pošli výsledok tool callu späť do Gemini
                                    result_data = json.loads(result_str)
                                    await gemini_session.send(
                                        types.LiveClientToolResponse(
                                            function_responses=[
                                                types.FunctionResponse(
                                                    name=tool_name,
                                                    response=result_data
                                                )
                                            ]
                                        )
                                    )
                                    
                                    # Ak je to ukonci_hovor, zavri spojenie
                                    if tool_name == "ukonci_hovor":
                                        await asyncio.sleep(3)
                                        print("[ws/voice-gemini] ukonci_hovor — zatvaram spojenie")
                                        await websocket.close()
                                        return

                        # Turn complete
                        if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
                            print("[gemini] Turn complete")

                    # User transcript (input audio transcription)
                    if hasattr(response, 'tool_call') and response.tool_call:
                        # Spracované vyššie v server_content
                        pass

            except Exception as e:
                print(f"[ws/voice-gemini] Gemini receive error: {e}")
                import traceback
                traceback.print_exc()
                return

        # Spusti oba smery súbežne
        await asyncio.gather(
            twilio_to_gemini(),
            gemini_to_twilio(),
        )

    except Exception as e:
        print(f"[ws/voice-gemini] chyba: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if gemini_session:
            try:
                await gemini_session.__aexit__(None, None, None)
            except Exception:
                pass
        print("[ws/voice-gemini] spojenie ukončené")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
