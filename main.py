import os
import time
import unicodedata
from html import escape as xml_escape
from rapidfuzz import fuzz
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
_STREET_MIN_SCORE = 60
_STREET_AUTO_ACCEPT_SCORE = 90
_STREET_AUTO_ACCEPT_MARGIN = 5
_CACHE_TTL = 300  # 5 minút

# Docasny fallback pre posledneho volajuceho. Nepouzivat ako primarny zdroj telefonu.
_LAST_CALLER_PHONE: str = ""

# Kontext hovorov: Twilio CallSid / ElevenLabs conversation_id -> Twilio From cislo.
CALL_CONTEXT: dict[str, str] = {}
CONVERSATION_CONTEXT: dict[str, str] = {}
TWILIO_NUMBER_CONTEXT: set[str] = set()

# Zname vlastne Twilio cisla nikdy neukladame ako customer_phone.
# Dalsie cisla sa daju doplnit cez env TWILIO_OWNED_NUMBERS oddelene ciarkou.
_DEFAULT_TWILIO_OWNED_NUMBERS = {"+420910922442", "+420910925466"}


def _normalize_phone(phone: str) -> str:
    return str(phone or "").strip().replace(" ", "")


def _twilio_owned_numbers() -> set[str]:
    raw = os.getenv("TWILIO_OWNED_NUMBERS", "").strip()
    configured = {_normalize_phone(item) for item in raw.split(",") if item.strip()}
    return {num for num in (_DEFAULT_TWILIO_OWNED_NUMBERS | configured | TWILIO_NUMBER_CONTEXT) if num}


def _is_twilio_owned_number(phone: str) -> bool:
    normalized = _normalize_phone(phone)
    return bool(normalized and normalized in _twilio_owned_numbers())


def _first_customer_phone_candidate(*phones: str) -> str:
    for phone in phones:
        normalized = _normalize_phone(phone)
        if normalized and not _is_twilio_owned_number(normalized):
            return normalized
    return ""



def _normalize(s: str) -> str:
    """Lowercase + odstránenie diakritiky."""
    nfkd = unicodedata.normalize("NFD", s.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def similarity_score(a: str, b: str) -> float:
    # vráti číslo 0.0 až 1.0
    return SequenceMatcher(None, a, b).ratio()


def classify_address_match(input_text: str, candidate_address: str, existing_score: float = None) -> dict:
    # vráti confidence, match_type, requires_confirmation, reason
    input_norm = _normalize(input_text)
    candidate_norm = _normalize(candidate_address)
    
    if existing_score is not None:
        # Pouzijeme skóre od 0 do 1
        score = existing_score / 100.0 if existing_score > 1.0 else existing_score
    else:
        # Fallback na SequenceMatcher, ak skóre chýba
        score = similarity_score(input_norm, candidate_norm)
    
    if input_norm == candidate_norm or score >= 0.98:
        return {
            "confidence": round(score, 2),
            "match_type": "exact",
            "requires_confirmation": False,
            "reason": "Presná zhoda."
        }
    elif score >= 0.90:
        return {
            "confidence": round(score, 2),
            "match_type": "normalized",
            "requires_confirmation": True,
            "reason": "Vysoká podobnosť (líši sa formátovanie alebo preklep)."
        }
    elif score >= 0.60:
        return {
            "confidence": round(score, 2),
            "match_type": "fuzzy",
            "requires_confirmation": True,
            "reason": "Čiastočná zhoda (zrejme preklep alebo časť názvu)."
        }
    else:
        return {
            "confidence": round(score, 2),
            "match_type": "low_confidence",
            "requires_confirmation": True,
            "reason": "Veľmi slabá zhoda."
        }


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


def _street_query_candidates(raw_address: str) -> list[str]:
    """Build street-name candidates from an address without trusting the house number."""
    address = raw_address.strip()
    if not address:
        return []

    candidates = []
    parts = address.rsplit(maxsplit=1)
    if len(parts) == 2:
        candidates.append(parts[0])
    candidates.append(address)

    cleaned = []
    seen = set()
    for candidate in candidates:
        value = candidate.strip(" ,.")
        key = _normalize(value)
        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)
    return cleaned


def _rank_streets(raw_address: str, streets: list[str]) -> list[dict]:
    candidates = _street_query_candidates(raw_address)
    results = []
    for street in streets:
        best_score = 0
        best_ratio = 0
        for candidate in candidates:
            primary, ratio = _street_score(candidate, street)
            if (primary, ratio) > (best_score, best_ratio):
                best_score = primary
                best_ratio = ratio
        results.append({"street": street, "score": best_score, "ratio": best_ratio})
    results.sort(key=lambda x: (x["score"], x["ratio"]), reverse=True)
    return results


def _street_resolution(raw_address: str, streets: list[str]) -> dict:
    ranked = _rank_streets(raw_address, streets)
    top = [item for item in ranked[:5] if item["score"] >= _STREET_MIN_SCORE]
    best = top[0] if top else None
    second = top[1] if len(top) > 1 else None
    margin = best["score"] - second["score"] if best and second else 100
    auto_accept = bool(
        best
        and best["score"] >= _STREET_AUTO_ACCEPT_SCORE
        and margin >= _STREET_AUTO_ACCEPT_MARGIN
    )
    return {
        "best": best,
        "suggestions": top,
        "margin": margin,
        "auto_accept": auto_accept,
        "all_ranked_top5": ranked[:5] # pridany debug pre zistenie pred odfiltrovanim
    }


def _get_streets_cached(tenant_id: str) -> list[str]:
    """Načíta ulice z DB, výsledok cachuje na 5 minút."""
    now = time.monotonic()
    if (
        _STREETS_CACHE["tenant_id"] == tenant_id
        and _STREETS_CACHE["data"]
        and now - _STREETS_CACHE["timestamp"] < _CACHE_TTL
    ):
        return _STREETS_CACHE["data"]

    if not supabase:
        raise Exception("Supabase klient nie je inicializovany")

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
    """Fuzzy match adresy voci tabulke ulic. Vracia (matched_address, confidence 0-100)."""
    if not supabase or not tenant_id:
        return raw_address, 0

    try:
        streets = _get_streets_cached(tenant_id)
        if not streets:
            return raw_address, 0

        address = raw_address.strip()
        resolution = _street_resolution(address, streets)
        best = resolution["best"]
        if not best:
            print(f"Address no match: '{address}'")
            return raw_address, 0

        confidence = int(best["score"])
        if not resolution["auto_accept"]:
            print(
                f"Address uncertain: '{address}' -> '{best['street']}' "
                f"score={confidence} margin={resolution['margin']}"
            )
            return raw_address, confidence

        parts = address.rsplit(maxsplit=1)
        house_number = parts[1] if len(parts) == 2 else ""
        matched_address = f"{best['street']} {house_number}".strip() if house_number else best["street"]
        print(f"Address match: '{address}' -> '{matched_address}' score={confidence}")
        return matched_address, confidence
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

    #TWILIO FORM DATA
    try:
        form_data = await request.form()
        twilio_payload = dict(form_data)
        from_number = _normalize_phone(form_data.get("From") or "")
        caller_number = _normalize_phone(form_data.get("Caller") or "")
        to_number = _normalize_phone(form_data.get("To") or "")
        called_number = _normalize_phone(form_data.get("Called") or "")
        call_sid = str(form_data.get("CallSid") or "")
        if to_number:
            TWILIO_NUMBER_CONTEXT.add(to_number)
        if called_number:
            TWILIO_NUMBER_CONTEXT.add(called_number)
        customer_number = _first_customer_phone_candidate(from_number, caller_number)
        print(f"[twilio/voice] raw Twilio payload: {twilio_payload}")
        print(f"[twilio/voice] Inbound call: from={from_number}, caller={caller_number}, to={to_number}, called={called_number}, call_sid={call_sid}, resolved_customer={customer_number}")
        if customer_number:
            global _LAST_CALLER_PHONE
            _LAST_CALLER_PHONE = customer_number
            if call_sid:
                CALL_CONTEXT[call_sid] = customer_number
            print(f"[twilio/voice] _LAST_CALLER_PHONE={customer_number}, CALL_CONTEXT[{call_sid}]={customer_number if call_sid else ''}")
        elif from_number or caller_number:
            print(f"[twilio/voice] WARNING: no non-Twilio caller resolved, from={from_number}, caller={caller_number}, owned={sorted(_twilio_owned_numbers())}")
    except Exception as e:
        print(f"[twilio/voice] Chyba pri citani Twilio form data: {e}")
        from_number = ""
        caller_number = ""
        customer_number = ""
        to_number = ""
        called_number = ""
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
                    "caller_number": customer_number or from_number,
                    "from_number": from_number,
                    "to_number": to_number,
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
    Vrati found=True iba pri vysokej a jednoznacnej zhode.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase klient nie je inicializovany.")

    query = body.query.strip()
    if not query:
        # Prazdny vstup
        return {
            "ok": False,
            "input": query,
            "query": query,
            "candidates": [],
            "selected_candidate": None,
            "match_type": "not_found",
            "requires_confirmation": False,
            "reason": "Prázdny dopyt.",
            # Zachovanie starych odpovedi pre kompatibilitu
            "found": False,
            "needs_confirmation": False,
            "best_match": None,
            "message": "Nezadal si žiadnu ulicu.",
            "suggestions": []
        }

    try:
        streets = _get_streets_cached(TENANT_ID)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chyba pri nacitani ulic: {e}")

    if not streets:
        return {
            "ok": False,
            "input": query,
            "query": query,
            "candidates": [],
            "selected_candidate": None,
            "match_type": "not_found",
            "requires_confirmation": False,
            "reason": "Nenašiel sa žiadny vhodný kandidát. (Zoznam ulíc je prázdny.)",
            "found": False,
            "needs_confirmation": False,
            "best_match": None,
            "message": "Na tuto adresu momentalne nevieme dorucit.", 
            "suggestions": []
        }

        print(f"[search-street] query='{query}' streets_count={len(streets)}")

    resolution = _street_resolution(query, streets)
    
        # DEBUG OBJEKT
    debug_info = {
        "street_min_score": _STREET_MIN_SCORE,
        "street_auto_accept_score": _STREET_AUTO_ACCEPT_SCORE,
        "street_auto_accept_margin": _STREET_AUTO_ACCEPT_MARGIN,
        "raw_query": query,
        "normalized_query": _normalize(query),
        "street_source": "supabase",
        "street_count": len(streets),
        "tenant_id": TENANT_ID,
        "fallback_used": False,
        "top_raw_candidates": [
            {
                "street": item["street"],
                "score": item["score"],
                "normalized_street": _normalize(item["street"])
            }
            for item in resolution.get("all_ranked_top5", [])
        ]
    }
    
    candidates = []
    
    # Vybuduj zoznam candidates v pozadovanom formate
    for item in resolution["suggestions"]:
        street = item["street"]
        classification = classify_address_match(query, street, item["score"])
        candidate = {
            "address": street,
            "confidence": classification["confidence"],
            "match_type": classification["match_type"],
            "requires_confirmation": classification["requires_confirmation"],
            "reason": classification["reason"]
        }
        candidates.append(candidate)
        
    top_old_style = [{"street": item["street"], "score": item["score"]} for item in resolution["suggestions"]]

    print(f"[search-street] top_results={top_old_style} margin={resolution['margin']} auto_accept={resolution['auto_accept']}")

        try:
        if not candidates:
            return {
                "ok": False,
                "input": query,
                "query": query,
                "candidates": [],
                "selected_candidate": None,
                "match_type": "not_found",
                "requires_confirmation": False,
                "reason": "Nenašiel sa žiadny vhodný kandidát.",
                "found": False,
                "needs_confirmation": True,
                "best_match": None,
                "message": "Nerozumel som presne nazvu ulice. Poproste zakaznika, aby ulicu zopakoval po pismenach alebo povedal blizsi orientacny bod.",
                "suggestions": [],
                "debug": debug_info
            }
            
        best_candidate = candidates[0]
        
        # Detekcia ambiguous (viacero relevantnych kandidatov s podobnym skore)
        if len(candidates) > 1:
            top1_score = best_candidate["confidence"] * 100
            top2_score = candidates[1]["confidence"] * 100
            margin_score = top1_score - top2_score
            
            top1_norm = _normalize(str(best_candidate.get("address") or ""))
            top2_norm = _normalize(str(candidates[1].get("address") or ""))
            
            if not top1_norm or not top2_norm:
                is_significantly_shorter = False
            else:
                is_significantly_shorter = len(top2_norm) < (len(top1_norm) * 0.70)
            
            if (
                top1_score >= _STREET_MIN_SCORE
                and top2_score >= 75
                and margin_score <= 4
                and not is_significantly_shorter
            ):
                best_candidate["match_type"] = "ambiguous"
                best_candidate["requires_confirmation"] = True
                best_candidate["reason"] = "Nájdených viacero podobných možností, nutné upresniť."

        needs_confirmation = not resolution["auto_accept"] or best_candidate["requires_confirmation"]

        # Pre stary parameter message:
        if needs_confirmation:
            message = "Adresa je neista. Nepotvrdzujte objednavku; najprv zakaznikovi precitajte najpravdepodobnejsiu ulicu a vypytajte si jasne ano/nie potvrdenie."
            if best_candidate["match_type"] == "ambiguous":
                message = "Nájdených viacero možností. Poproste zákazníka, aby upresnil ulicu."
        else:
            message = "Ulica najdena a potvrdzena."

        return {
            "ok": True,
            "input": query,
            "query": query,
            "candidates": candidates,
            "selected_candidate": {
                "address": best_candidate["address"],
                "confidence": best_candidate["confidence"],
                "match_type": best_candidate["match_type"],
                "requires_confirmation": needs_confirmation
            },
            "found": not needs_confirmation, # Starý agent možno očakáva found=True len pri auto_accept
            "best_match": top_old_style[0]["street"],
            "confidence": top_old_style[0]["score"],
            "needs_confirmation": needs_confirmation,
            "margin": resolution["margin"],
            "message": message,
            "suggestions": top_old_style,
            "debug": debug_info
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "input": query,
            "query": query,
            "candidates": [],
            "selected_candidate": None,
            "match_type": "error",
            "requires_confirmation": False,
            "reason": f"Interna chyba pri spracovani adresy: {str(e)}",
            "found": False,
            "needs_confirmation": True,
            "best_match": None,
            "message": "Nastala chyba pri vyhladavani adresy.",
            "suggestions": [],
            "debug": debug_info
        }

# --- WHATSAPP LOGIKA (DOPLNOK) ---

async def send_whatsapp_message(to: str, message: str, template_sid: str = None, variables: dict = None) -> bool:
    """Odosle WhatsApp spravu cez Twilio REST API (podporuje free-form aj šablóny)"""
    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    
    if not twilio_account_sid or not twilio_auth_token:
        print("[whatsapp] CHYBA: Chybaju Twilio credentials")
        return False
        
    TWILIO_WHATSAPP_NUMBER = '+420910922442'
    from_number = f"whatsapp:{TWILIO_WHATSAPP_NUMBER}"
    to_number = to if to.startswith("whatsapp:") else f"whatsapp:{to}"
    
    try:
        import httpx
        import json
        twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_account_sid}/Messages.json"
        
        data = {
            "From": from_number,
            "To": to_number,
        }
        
        # Ak pouzivame sablonu (produkcia)
        if template_sid:
            data["ContentSid"] = template_sid
            if variables:
                data["ContentVariables"] = json.dumps(variables)
            print(f"[whatsapp] Odosielam SABLONU {template_sid} na {to_number}")
        else:
            # Free-form sprava (len v ramci 24h okna)
            data["Body"] = message
            print(f"[whatsapp] Odosielam BODY na {to_number}")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                twilio_url,
                data=data,
                auth=(twilio_account_sid, twilio_auth_token),
                timeout=10.0
            )
        print(f"[whatsapp] Twilio response: {resp.status_code} - {resp.text}")
        return resp.status_code in [200, 201]
    except Exception as e:
        print(f"[whatsapp] CHYBA: {e}")
        return False


async def send_order_notifications_task(order_data: dict):
    """Spracuje a odosle notifikacie pre zakaznika aj restauraciu (len WhatsApp)."""
    # KONFIGURACIA SABLON
    TPL_CUSTOMER = os.getenv("TWILIO_TPL_CUSTOMER") 
    TPL_RESTAURANT = os.getenv("TWILIO_TPL_RESTAURANT")
    RESTAURANT_PHONE = os.getenv("RESTAURANT_PHONE", "+421910922442")
    
    pizza = order_data.get("pizza_type", "Pizza")
    address = order_data.get("delivery_address", "u nas")
    price = str(order_data.get("total_price", "0"))
    phone = order_data.get("customer_phone", "")
    notes = order_data.get("notes", "-")

    # 1. NOTIFIKACIA PRE RESTAURACIU
    msg_rest = f"✅ *NOVÁ OBJEDNÁVKA* \n\nZákazník: {phone}\nAdresa: {address}\nPizza: {pizza}\nSuma: {price} €"
    vars_rest = {"1": phone, "2": address, "3": pizza, "4": price, "5": notes}
    await send_whatsapp_message(RESTAURANT_PHONE, msg_rest, TPL_RESTAURANT, vars_rest)

    # 2. NOTIFIKACIA PRE ZAKAZNIKA
    if phone and phone.startswith("+"):
        msg_cust = f"Dobrý deň! Vaša objednávka z Papizoo ({pizza}) sa pripravuje. Suma: {price} €."
        vars_cust = {"1": pizza, "2": address, "3": price}
        await send_whatsapp_message(phone, msg_cust, TPL_CUSTOMER, vars_cust)
    
    print(f"[whatsapp] Notifikacie spracovane.")


@app.post("/api/vytvor-objednavku")
async def vytvor_objednavku(request: Request, background_tasks: BackgroundTasks):
    """
    ElevenLabs tool endpoint pre vytvorenie objednavky.
    Zapisuje do Supabase a planuje WhatsApp notifikacie.
    """
    try:
        body = await request.json()
        print(f"[vytvor-objednavku] raw body: {body}")

        order = ManageOrder(**body)
    except Exception as e:
        print(f"[vytvor-objednavku] validacna chyba: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    try:
        matched_address, confidence = match_street(order.delivery_address, TENANT_ID)
        if confidence < _STREET_AUTO_ACCEPT_SCORE:
            print(
                f"[vytvor-objednavku] address needs confirmation: "
                f"raw='{order.delivery_address}' confidence={confidence}"
            )
            return {
                "status": "needs_confirmation",
                "message": "Adresu sa nepodarilo spolahlivo overit. Objednavku este neuzatvarajte; vypytajte si potvrdenie ulice a cisla domu.",
                "delivery_address": order.delivery_address,
                "address_confidence": confidence,
                        }

                # caller_number poslali sme sami do ElevenLabs z Twilio From — je to spravne cislo volajuceho
        caller_number = _normalize_phone(body.get("caller_number") or "")
        payload_phone = _normalize_phone(body.get("customer_phone") or order.customer_phone or "")

        if caller_number and not _is_twilio_owned_number(caller_number):
            real_phone = caller_number
        elif _LAST_CALLER_PHONE and not _is_twilio_owned_number(_LAST_CALLER_PHONE):
            real_phone = _LAST_CALLER_PHONE
            print("[vytvor-objednavku] WARNING: caller_number chybalo, pouzivam _LAST_CALLER_PHONE")
        elif payload_phone and not _is_twilio_owned_number(payload_phone):
            real_phone = payload_phone
            print("[vytvor-objednavku] WARNING: pouzivam payload customer_phone")
        else:
            real_phone = ""
            print(f"[vytvor-objednavku] WARNING: ziadne platne cislo; caller_number={caller_number}, payload_phone={payload_phone}")

        print(f"[vytvor-objednavku] phone_resolution caller_number={caller_number}, _LAST_CALLER_PHONE={_LAST_CALLER_PHONE}, final={real_phone}")

        order_data = {
            "tenant_id": TENANT_ID,
            "customer_phone": real_phone,
            "delivery_address": matched_address,
            "pizza_type": order.pizza_type,
            "total_price": float(order.total_price),
            "upsell_accepted": order.upsell_accepted,
            "notes": order.transcript,
            "status": "NEW",
        }

        supabase.table("pizza_orders").insert(order_data).execute()
        print(f"[vytvor-objednavku] INSERT pizza_orders OK")

        # OKAMZITA NOTIFIKACIA (uz nie na pozadi, aby to bolo hned)
        await send_order_notifications_task(order_data)

        return {"status": "success", "message": "Objednavka uspesne zapisana."}
    except Exception as e:
        print(f"[vytvor-objednavku] CHYBA: {e}")
        raise HTTPException(status_code=500, detail=str(e))





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

    call_sid = body.get("call_sid") or body.get("CallSid") or ""
    conversation_id = body.get("conversation_id") or body.get("conversationId") or ""
    if conversation_id and call_sid and call_sid in CALL_CONTEXT:
        CONVERSATION_CONTEXT[conversation_id] = CALL_CONTEXT[call_sid]
    print(f"[end_call] hovor ukonceny, call_sid={call_sid}, conversation_id={conversation_id}, payload={body}")

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
