import os
import time
import unicodedata
from html import escape as xml_escape
from rapidfuzz import fuzz, process
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# Nacitanie environment premennych (uzitocne pre lokalny vyvoj)
load_dotenv()

# --- ELEVENLABS KONFIGURÁCIA ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID", "").strip()

app = FastAPI(title="ElevenLabs Pizza Webhook")


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
print(f"ELEVENLABS_API_KEY nastavene: {'ano' if bool(ELEVENLABS_API_KEY) else 'nie'}")
print(f"ELEVENLABS_AGENT_ID nastavene: {'ano' if bool(ELEVENLABS_AGENT_ID) else 'nie'}")
print(f"CORS_ALLOW_ORIGINS: {CORS_ALLOW_ORIGINS}")
print("----------------------")

try:
    # Inicializacia Supabase klienta
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Chyba pri inicializacii Supabase: {e}")
    supabase = None


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


def _street_score(query: str, street: str) -> tuple[int, int]:
    """Vráti (primary_score, ratio) — primary uprednostňuje presnejšiu/kratšiu zhodu.
    partial_ratio je penalizovaný 0.85x, aby kratší presný match vyhral nad substrinom.
    """
    q = _normalize(query)
    s = _normalize(street)
    r = fuzz.ratio(q, s)
    pr = fuzz.partial_ratio(q, s)
    primary = round(max(r, pr * 0.85))
    return primary, r


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


async def _check_systems() -> tuple[bool, str]:
    """
    Skontroluje ci su vsetky systemy dostupne pred spustenim hovoru.
    Vracia (ok: bool, reason: str).
    """
    print("[check_systems] Kontrola systemov...")
    print(f"[check_systems] supabase ready: {supabase is not None}")
    print(f"[check_systems] ELEVENLABS_API_KEY nastaveny: {bool(ELEVENLABS_API_KEY)}")
    print(f"[check_systems] ELEVENLABS_AGENT_ID nastaveny: {bool(ELEVENLABS_AGENT_ID)}")

    if not supabase:
        print("[check_systems] FAIL: Supabase klient nie je inicializovany")
        return False, "Supabase klient nie je inicializovany"
    try:
        supabase.table("menu_items").select("name").limit(1).execute()
        print("[check_systems] DB: OK")
    except Exception as e:
        print(f"[check_systems] FAIL: Databaza nedostupna: {e}")
        return False, f"Databaza nedostupna: {e}"
    if not ELEVENLABS_API_KEY:
        print("[check_systems] FAIL: ELEVENLABS_API_KEY chyba")
        return False, "Chyba ELEVENLABS_API_KEY"
    if not ELEVENLABS_AGENT_ID:
        print("[check_systems] FAIL: ELEVENLABS_AGENT_ID chyba")
        return False, "Chyba ELEVENLABS_AGENT_ID"
    print("[check_systems] Vsetko OK")
    return True, "OK"


async def _get_elevenlabs_signed_url(menu: str) -> Optional[str]:
    """
    Ziska podpisanu WebSocket URL od ElevenLabs.
    Tato URL sa pouzije v TwiML <Stream> — prepoji Twilio priamo s ElevenLabs agentom.
    Menu sa doruci cez /api/prompt-config ktore ElevenLabs vola samostatne.
    """
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.elevenlabs.io/v1/convai/conversation/get_signed_url",
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                params={"agent_id": ELEVENLABS_AGENT_ID},
                timeout=8.0,
            )
        if resp.status_code == 200:
            signed_url = resp.json().get("signed_url")
            print("[elevenlabs] Signed URL ziskana OK")
            return signed_url
        else:
            print(f"[elevenlabs] Signed URL chyba: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        print(f"[elevenlabs] Signed URL exception: {e}")
        return None


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

        # Kandidáti na street_part + house_numbto er:
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


@app.api_route("/twilio/incoming", methods=["GET", "POST"])
@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice_webhook(request: Request):
    """
    Hlavny vstupny bod kazdeho hovoru — Render riadi hovor cez ElevenLabs register_call.
    1. Twilio zavola Render /twilio/voice
    2. Render skontroluje DB + ElevenLabs konfiguraciu
    3. Render nacita menu z DB a posle ho ako dynamic variable do ElevenLabs
    4. ElevenLabs vrati hotove TwiML pre Twilio Media Stream
    5. Render vrati toto TwiML priamo Twiliu
    """
    def unavailable_twiml() -> str:
        audio_url = os.getenv("AUDIO_LINKA_NEDOSTUPNA", "").strip()
        if audio_url:
            return f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{xml_escape(audio_url, quote=False)}</Play>
    <Hangup/>
</Response>'''
        return '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Dobry den, lutujeme, nasa objednavkova linka je momentalne nedostupna. Skuste prosim zavolat o chvilu neskor. Prajeme pekny den.</Say>
    <Pause length="1"/>
    <Hangup/>
</Response>'''

    # 1. KONTROLA SYSTÉMOV
    ok, reason = await _check_systems()
    if not ok:
        print(f"[twilio/voice] Systemy nedostupne: {reason}")
        return Response(content=unavailable_twiml(), media_type="application/xml")

    # 2. TWILIO FORM DATA
    try:
        form_data = await request.form()
        from_number = str(form_data.get("From") or "")
        to_number = str(form_data.get("To") or "")
        call_sid = str(form_data.get("CallSid") or "")
        print(f"[twilio/voice] Inbound call: from={from_number}, to={to_number}, call_sid={call_sid}")
    except Exception as e:
        print(f"[twilio/voice] Chyba pri citani Twilio form data: {e}")
        from_number = ""
        to_number = ""
        call_sid = ""

    # 3. MENU Z DB -> DYNAMIC VARIABLE
    menu = format_menu_from_db(TENANT_ID)
    if not menu:
        menu = "Menu momentalne nie je dostupne."
    print(f"[twilio/voice] Menu nacitane, dlzka={len(menu)} znakov")

    # 4. ELEVENLABS REGISTER CALL -> HOTOVE TWIML PRE TWILIO
    try:
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        twiml = client.conversational_ai.twilio.register_call(
            agent_id=ELEVENLABS_AGENT_ID,
            from_number=from_number,
            to_number=to_number,
            direction="inbound",
            conversation_initiation_client_data={
                "dynamic_variables": {
                    "menu": menu,
                    "caller_number": from_number,
                    "call_sid": call_sid,
                }
            },
        )
        print("[twilio/voice] ElevenLabs register_call OK, vraciam TwiML Twiliu")
        return Response(content=twiml, media_type="application/xml")
    except Exception as e:
        print(f"[twilio/voice] ElevenLabs register_call zlyhal: {e}")
        return Response(content=unavailable_twiml(), media_type="application/xml")


@app.api_route("/twilio/fallback", methods=["GET", "POST"])
async def twilio_fallback_webhook():
    """
    Zalozny Twilio webhook, ak primarny handler zlyha.
    """
    twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Ospravedlnujeme sa, linka je docasne nedostupna. Skuste prosim zavolat neskor.</Say>
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
async def search_street(body: SearchStreetRequest):
    """
    Fuzzy vyhladavanie ulice podla casti nazvu (STT vystup z ElevenLabs).
    Cachuje ulice z DB na 5 minut. Vracia max 2 zhody so skore >= 55.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase klient nie je inicializovany.")

    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Parameter query nesmie byt prazdny.")

    try:
        streets = _get_streets_cached(TENANT_ID)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chyba pri nacitani ulic: {e}")

    if not streets:
        return {"found": False, "message": "Na tuto adresu momentalne nevieme dorucit.", "suggestions": []}

    print(f"[search-street] query='{query}' streets_count={len(streets)}")

    results = []
    for s in streets:
        primary, ratio = _street_score(query, s)
        results.append({"street": s, "score": primary, "ratio": ratio})
    results.sort(key=lambda x: (x["score"], x["ratio"]), reverse=True)
    top = [{"street": item["street"], "score": item["score"]} for item in results[:2] if item["score"] >= 55]

    print(f"[search-street] top_results={top}")

    if not top:
        return {"found": False, "message": "Na tuto adresu momentalne nevieme dorucit.", "suggestions": []}

    return {
        "found": True,
        "best_match": top[0]["street"],
        "confidence": top[0]["score"],
        "suggestions": top,
    }


async def send_whatsapp_message(to: str, body: str) -> bool:
    """Odosle WhatsApp spravu cez Twilio REST API"""
    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()

    if not twilio_account_sid or not twilio_auth_token:
        print("Twilio credentials missing. Cannot send WhatsApp messages.")
        return False

    TWILIO_WHATSAPP_NUMBER = '+420910922442'
    
    from_number = f"whatsapp:{TWILIO_WHATSAPP_NUMBER}"
    to_number = to if to.startswith("whatsapp:") else f"whatsapp:{to}"

    try:
        import httpx
        twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_account_sid}/Messages.json"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                twilio_url,
                data={
                    "Body": body,
                    "From": from_number,
                    "To": to_number
                },
                auth=(twilio_account_sid, twilio_auth_token),
                timeout=10.0,
            )
        if resp.status_code in (200, 201):
            print(f"[whatsapp] Message sent successfully to {to_number}")
            return True
        else:
            print(f"[whatsapp] Failed to send message to {to_number}: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"[whatsapp] Exception while sending message to {to_number}: {e}")
        return False

async def send_order_notifications_task(order_data: dict):
    """Bezi na pozadi a posle notifikacie do pizzerie aj zakaznikovi."""
    import asyncio
    
    restaurant_phone = os.getenv("RESTAURANT_PHONE", "+421910922442") # Zmenit na realne cislo pizzerie
    customer_phone = order_data.get("customer_phone", "")
    delivery_address = order_data.get("delivery_address", "")
    pizza_type = order_data.get("pizza_type", "")
    total_price = order_data.get("total_price", 0.0)
    
    # Sprava pre restauraciu
    restaurant_body = (
        "🍕 *NOVÁ OBJEDNÁVKA PIZZE*\n\n"
        f"📞 *Zákazník:* {customer_phone}\n"
        f"📍 *Adresa doručenia:* {delivery_address}\n\n"
        f"🛒 *Objednávka:*\n"
        f"{pizza_type}\n\n"
        f"💰 *Spolu:* {total_price:.2f} €"
    )
    
    # Sprava pre zakaznika
    customer_body = (
        "🍕 *Ďakujeme za Vašu objednávku z PapiZoo!*\n\n"
        f"Vaša objednávka sa pripravuje.\n"
        f"🛒 *Objednávka:*\n{pizza_type}\n"
        f"📍 *Bude doručená na adresu:* {delivery_address}\n"
        f"💰 *Suma k úhrade:* {total_price:.2f} €.\n\n"
        "Dobrú chuť! 😊"
    )
    
    tasks = []
    if restaurant_phone:
        tasks.append(send_whatsapp_message(restaurant_phone, restaurant_body))
    if customer_phone:
        tasks.append(send_whatsapp_message(customer_phone, customer_body))
        
    if tasks:
        await asyncio.gather(*tasks)

@app.post("/api/vytvor-objednavku")
async def vytvor_objednavku(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint pre ElevenLabs agenta (manage_order tool) na ulozenie objednavky do DB.
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

        # Spustenie notifikacii na pozadi (neblokuje agenta)
        background_tasks.add_task(send_order_notifications_task, order_data)

        return {
            "status": "success",
            "message": "Objednavka uspesne zapisana.",
        }

    except Exception as e:
        error_msg = str(e)
        print(f"DEBUG CHYBA: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Chyba Supabase: {error_msg}")


@app.post("/api/end_call")
async def end_call(request: Request):
    """
    ElevenLabs end_call tool endpoint.
    1. Vracia end_call: true — ElevenLabs signal na fyzicke ukoncenie hovoru.
    2. Ak je k dispozicii CallSid, zaves hovor aj priamo cez Twilio API.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    call_sid = body.get("call_sid") or body.get("CallSid") or body.get("conversation_id")
    print(f"[end_call] hovor ukonceny, call_sid={call_sid}, payload={body}")

    # Pokus o zavesenie priamo cez Twilio API (ak mame CallSid a credentials)
    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()

    if call_sid and twilio_account_sid and twilio_auth_token:
        try:
            import httpx
            twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_account_sid}/Calls/{call_sid}.json"
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    twilio_url,
                    data={"Status": "completed"},
                    auth=(twilio_account_sid, twilio_auth_token),
                    timeout=5.0,
                )
            print(f"[end_call] Twilio hangup: {resp.status_code}")
        except Exception as e:
            print(f"[end_call] Twilio hangup chyba (nekriticka): {e}")

    # ElevenLabs signal — toto je hlavny mechanizmus ukoncenia
    return {"end_call": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
