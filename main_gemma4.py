"""
FastAPI backend pre Twilio Media Streams s Gemma 4 (E4B/E2B) modelom.
Natívne audio-to-audio spracovanie bez Whisper/TTS medzikrokov.
"""

import os
import asyncio
import json
import base64
import time
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from dotenv import load_dotenv

# Lokálne moduly
from engine import Gemma4Engine, AudioChunk, ModelResponse, detect_speech
from utils import mulaw_to_pcm16, pcm16_to_mulaw

# Supabase pre databázu (zachované z pôvodného kódu)
from supabase import create_client, Client
from rapidfuzz import process
import unicodedata

load_dotenv()

app = FastAPI(title="Pizza Sicilia Voice Assistant - Gemma 4")

# CORS konfigurácia
def _parse_cors_origins() -> list[str]:
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

# Konfigurácia
SUPABASE_URL = os.getenv("CORE_SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("CORE_SUPABASE_SERVICE_ROLE_KEY", "").strip()
TENANT_ID = os.getenv("TENANT_ID", "").strip()

# Gemma 4 konfigurácia
GEMMA4_MODEL_PATH = os.getenv("GEMMA4_MODEL_PATH", "gemma-4-e4b").strip()
USE_VLLM = os.getenv("USE_VLLM", "true").lower() == "true"
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))

# Načítaj system prompt
_prompt_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
PIZZA_SYSTEM_PROMPT_BASE = open(_prompt_path, encoding="utf-8").read().strip()

# Fallback menu
PIZZA_MENU_FALLBACK = """
MENU (fallback):
| Pizza | Cena |
|-------|------|
| Margherita | 6.50€ |
| Salami | 7.50€ |
| Hawaii | 7.90€ |
| Pepperoni | 9.90€ |
| Quattro Formaggi | 9.50€ |"""

# Tools pre function calling
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

# Inicializácia Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except Exception as e:
    print(f"Chyba pri inicializacii Supabase: {e}")
    supabase = None

# Startup info
print("--- GEMMA 4 STARTUP KONFIG ---")
print(f"GEMMA4_MODEL_PATH: {GEMMA4_MODEL_PATH}")
print(f"USE_VLLM: {USE_VLLM}")
print(f"GPU_MEMORY_UTILIZATION: {GPU_MEMORY_UTILIZATION}")
print(f"SUPABASE dostupný: {'ano' if supabase else 'nie'}")
print(f"TENANT_ID: {TENANT_ID if TENANT_ID else 'nenastavené'}")
print("------------------------------")

# Globálny Gemma4 engine (singleton)
gemma_engine: Optional[Gemma4Engine] = None


# Pomocné funkcie z pôvodného kódu

ALLERGEN_MAP = {
    "1": "lepok", "2": "kôrovce", "3": "vajcia", "4": "ryby",
    "5": "arašidy", "6": "sója", "7": "mlieko", "8": "orechy",
    "9": "zeler", "10": "horčica", "11": "sezam", "12": "oxid siričitý",
    "13": "vlčí bôb", "14": "mäkkýše",
}

def _normalize(s: str) -> str:
    """Lowercase + odstránenie diakritiky."""
    nfkd = unicodedata.normalize("NFD", s.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


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
    """Fuzzy match adresy voči tabuľke ulíc."""
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
            candidates.append((parts[0], parts[1]))
        candidates.append((address, ""))
        
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
        
        top = process.extractOne(address.lower(), street_names_lower.keys())
        print(f"Address no match: '{address}' | best candidate: {top}")
        return raw_address, 0
    except Exception as e:
        print(f"Chyba pri matchovani adresy: {e}")
        return raw_address, 0


async def handle_tool_call(tool_name: str, tool_args: dict, phone_number: str, transcript_lines: list = []) -> dict:
    """Vykoná tool call od Gemma agenta."""
    if tool_name == "over_adresu":
        raw_address = tool_args.get("query", tool_args.get("address", ""))
        matched_address, confidence = match_street(raw_address, TENANT_ID)
        if confidence > 0:
            print(f"[tool] over_adresu OK: '{raw_address}' -> '{matched_address}'")
            return {"found": True, "best_match": matched_address}
        else:
            print(f"[tool] over_adresu NOT FOUND: '{raw_address}'")
            return {"found": False, "message": "Adresa nie je v zóne doručenia."}
    
    if tool_name == "uloz_objednavku":
        if not tool_args.get("phone_number"):
            tool_args["phone_number"] = phone_number
        if not supabase:
            return {"status": "error", "message": "Supabase nie je dostupný"}
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
            return {"status": "success", "message": "Objednávka uložená."}
        except Exception as e:
            print(f"[tool] uloz_objednavku chyba: {e}")
            return {"status": "error", "message": str(e)}
    
    if tool_name == "ukonci_hovor":
        return {"status": "ok"}
    
    return {"status": "error", "message": f"Neznámy tool: {tool_name}"}


# FastAPI endpoints

@app.get("/")
async def health_check():
    return {"status": "online", "message": "Gemma 4 server bezi.", "model": GEMMA4_MODEL_PATH}


@app.get("/health")
def health():
    return {"status": "ok", "engine": "gemma4"}


@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice_webhook(request: Request):
    """Twilio voice webhook — smeruje na Gemma 4 endpoint."""
    host = request.headers.get("host", "")
    ws_endpoint = f"wss://{host}/media-stream"
    
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_endpoint}">
            <Parameter name="phone_number" value="{{{{From}}}}" />
        </Stream>
    </Connect>
</Response>'''
    return Response(content=twiml, media_type="application/xml")


@app.api_route("/twilio/fallback", methods=["GET", "POST"])
async def twilio_fallback_webhook():
    """Záložný Twilio webhook."""
    twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Vlasta" language="sk-SK">Ospravedlnujeme sa, linka je docasne nedostupna. Skuste prosim zavolat neskor.</Say>
    <Hangup/>
</Response>'''
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream_handler(websocket: WebSocket):
    """
    Hlavný WebSocket endpoint pre Twilio Media Streams.
    Spracováva audio cez Gemma 4 model (natívne audio-to-audio).
    """
    await websocket.accept()
    print("[media-stream] Twilio pripojený")
    
    global gemma_engine
    
    stream_sid: Optional[str] = None
    phone_number: str = ""
    transcript_lines: list = []
    
    # Audio buffering pre VAD (Voice Activity Detection)
    audio_buffer = bytearray()
    silence_duration = 0.0
    last_speech_time = time.time()
    
    # Barge-in detekcia
    user_is_speaking = False
    
    try:
        # Inicializuj Gemma 4 engine ak ešte nie je
        if gemma_engine is None:
            menu_text = format_menu_from_db(TENANT_ID)
            system_instructions = PIZZA_SYSTEM_PROMPT_BASE + "\n\n" + (menu_text if menu_text else PIZZA_MENU_FALLBACK)
            
            gemma_engine = Gemma4Engine(
                model_path=GEMMA4_MODEL_PATH,
                system_prompt=system_instructions,
                tools=PIZZA_TOOLS,
                use_vllm=USE_VLLM,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            )
            await gemma_engine.initialize()
        
        async def audio_input_generator():
            """Generátor pre vstupné audio chunky od používateľa."""
            nonlocal stream_sid, phone_number, audio_buffer, user_is_speaking, last_speech_time
            
            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    print("[media-stream] Twilio odpojil")
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
                    print(f"[media-stream] stream started sid={stream_sid} phone={phone_number}")
                
                elif event == "media":
                    # Twilio → mu-law 8kHz → PCM 16kHz
                    mulaw_b64 = msg["media"]["payload"]
                    mulaw_bytes = base64.b64decode(mulaw_b64)
                    pcm16k = mulaw_to_pcm16(mulaw_bytes, target_rate=16000)
                    
                    # Detekcia reči (VAD)
                    has_speech = detect_speech(pcm16k, threshold=0.02)
                    
                    if has_speech:
                        last_speech_time = time.time()
                        
                        # Barge-in: ak model práve generuje, prerušíme ho
                        if gemma_engine.is_generating and not user_is_speaking:
                            print("[media-stream] Barge-in detekovaný!")
                            gemma_engine.interrupt()
                            user_is_speaking = True
                        
                        audio_buffer.extend(pcm16k)
                    else:
                        # Ticho - ak máme nazbierané audio, pošleme ho
                        silence_duration = time.time() - last_speech_time
                        
                        if silence_duration > 0.8 and len(audio_buffer) > 0:
                            # Pošli nazbierané audio do modelu
                            yield AudioChunk(
                                data=bytes(audio_buffer),
                                sample_rate=16000,
                            )
                            audio_buffer.clear()
                            user_is_speaking = False
                
                elif event == "stop":
                    print("[media-stream] Twilio stream stop")
                    # Pošli zvyšné audio ak existuje
                    if len(audio_buffer) > 0:
                        yield AudioChunk(
                            data=bytes(audio_buffer),
                            sample_rate=16000,
                        )
                    return
        
        # Spracuj audio stream cez Gemma 4
        async def process_and_respond():
            """Spracuje audio od používateľa a streamuje odpovede späť."""
            nonlocal transcript_lines, stream_sid
            
            # Tool handler
            async def tool_handler(tool_name: str, tool_args: dict) -> dict:
                result = await handle_tool_call(tool_name, tool_args, phone_number, transcript_lines)
                
                # Ak je to ukonci_hovor, zavri spojenie
                if tool_name == "ukonci_hovor":
                    await asyncio.sleep(2)
                    print("[media-stream] ukonci_hovor — zatvaram spojenie")
                    await websocket.close()
                
                return result
            
            # Spracuj audio stream
            async for response in gemma_engine.process_audio_stream(
                audio_input_generator(),
                tool_handler=tool_handler,
            ):
                # Audio odpoveď od modelu
                if response.audio_data:
                    # PCM 16kHz → mu-law 8kHz pre Twilio
                    mulaw8k = pcm16_to_mulaw(response.audio_data, source_rate=16000)
                    mulaw_b64 = base64.b64encode(mulaw8k).decode()
                    
                    if stream_sid:
                        await websocket.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": mulaw_b64},
                        }))
                
                # Text transcript (pre logging)
                if response.text:
                    transcript_lines.append(f"Agent: {response.text}")
                    print(f"[gemma/transcript] Agent: {response.text}")
                
                # Tool call
                if response.tool_call:
                    print(f"[gemma/tool_call] {response.tool_call['name']}: {response.tool_call.get('result')}")
                
                # Koniec odpovede
                if response.is_complete:
                    print("[gemma] Response complete")
        
        # Spusti spracovanie
        await process_and_respond()
    
    except Exception as e:
        print(f"[media-stream] chyba: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[media-stream] spojenie ukončené")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main_gemma4:app", host="0.0.0.0", port=port, reload=False)
