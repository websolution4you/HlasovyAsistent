import os
import time
import datetime
import unicodedata
from difflib import SequenceMatcher
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
ELEVENLABS_AGENT_ID_PIZZA = os.getenv("ELEVENLABS_AGENT_ID_PIZZA", "").strip()

app = FastAPI(title="ElevenLabs Pizza Webhook")


def _parse_cors_origins() -> list[str]:
    """
    Nacita povolene originy pre web frontend.
    Defaultne povoli localhosty pre lokalny vyvoj.
    V Renderi nastav CORS_ALLOW_ORIGINS ako ciarkou oddeleny zoznam.
    """
    raw = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173,https://telio.sk,https://www.telio.sk",
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
NTC_TENANT_ID = os.getenv("NTC_TENANT_ID", "595cbb6c-1019-41ae-b1c2-a60c13c8dcdf").strip()

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
print(f"ELEVENLABS_AGENT_ID_PIZZA nastavene: {'ano' if bool(ELEVENLABS_AGENT_ID_PIZZA) else 'nie'}")
print(f"TWILIO_ACCOUNT_SID nastavene: {'ano' if bool(os.getenv('TWILIO_ACCOUNT_SID')) else 'nie'}")
twilio_app_sid_env = (
    os.getenv("TWILIO_TWIML_APP_SID") or
    os.getenv("TWILIO_APP_SID") or
    os.getenv("TWIML_APP_SID") or
    os.getenv("TWILIO_TWIML_APP_ID") or
    os.getenv("TWILIO_APP_ID")
)
print(f"TWILIO_TWIML_APP_SID nastavene: {'ano' if bool(twilio_app_sid_env) else 'nie'}")
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


class HumanFallbackRequest(BaseModel):
    caller_number: Optional[str] = None
    reason: Optional[str] = None


class CheckAvailabilityRequest(BaseModel):
    sport: str  # e.g., "badminton", "squash", "tennis", "tennis-clay"
    start_time_iso: str  # e.g., "2026-06-21T10:00:00"
    duration_minutes: Optional[int] = 60


class CreateBookingRequest(BaseModel):
    sport: str
    court_id: str  # e.g., "badminton-1"
    customer_name: str
    customer_phone: Optional[str] = None
    start_time_iso: str
    duration_minutes: Optional[int] = 60
    notes: Optional[str] = None
    caller_number: Optional[str] = None


ALLERGEN_MAP = {
    "1": "lepok", "2": "kôrovce", "3": "vajcia", "4": "ryby",
    "5": "arašidy", "6": "sója", "7": "mlieko", "8": "orechy",
    "9": "zeler", "10": "horčica", "11": "sezam", "12": "oxid siričitý",
    "13": "vlčí bôb", "14": "mäkkýše",
}


_STREETS_CACHE: dict = {"data": [], "tenant_id": "", "timestamp": 0.0}
_STREET_MIN_SCORE = 60
_STREET_AUTO_ACCEPT_SCORE = 75
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
    """Lowercase + odstránenie diakritiky + slovensko-fonetická normalizácia pre hlasových asistentov."""
    if not s:
        return ""
    import re
    # 1. Odstránenie diakritiky
    nfkd = unicodedata.normalize("NFD", s.lower().strip())
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    
    # 2. Slovenská fonetická normalizácia (prepis cudzích/historických mien na fonetické slovenské ekvivalenty)
    text = text.replace("cz", "c")     # Czauczika -> caucika (rovnako ako čaučika -> caucika)
    text = text.replace("sch", "s")    # Greschika -> gresika (rovnako ako grešika -> gresika)
    text = text.replace("y", "i")      # Zjednotenie i/y (Ruskynovský -> ruskinovski)
    
    # 3. Zjednotenie dvojitých hlások na jednu (Bottova -> botova, Hermann -> herman)
    text = re.sub(r'([a-z])\1', r'\1', text)
    
    return text


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


def clean_street_text(text: str) -> str:
    """Odstráni všeobecné slová/predložky a ponechá len jedinečnú časť názvu ulice pre lepšie párovanie."""
    normalized = _normalize(text)
    words = normalized.split()
    generics = {
        "ulica", "cesta", "priechod", "riadok", "chodnik", 
        "namestie", "sidlisko", "alej", "farma", "tunel", 
        "odpocivadlo", "za", "pod", "nad", "pri", "na", 
        "v", "s", "z", "do"
    }
    filtered = [w for w in words if w not in generics]
    if not filtered:
        # Ak by nezostalo nič (napr. dopyt bol len "Ulica"), vrátime pôvodné slová
        filtered = words
    return " ".join(filtered)


def _street_score(query: str, street: str) -> tuple[int, int]:
    """Vypočíta robustné skóre podobnosti založené na pokrytí tokenov (slov).
    Zabraňuje chybnému priradeniu krátkych názvov (napr. Nová) pre dlhé dopyty s preklepmi.
    """
    q = _normalize(query)
    s = _normalize(street)
    
    if q == s:
        return 100, 100
        
    qc = clean_street_text(query)
    sc = clean_street_text(street)
    
    if qc == sc:
        return 100, round(fuzz.ratio(q, s))
        
    r = fuzz.ratio(q, s)
    tsr = fuzz.token_sort_ratio(q, s)
    tset = fuzz.token_set_ratio(q, s)
    
    r_clean = fuzz.ratio(qc, sc)
    tsr_clean = fuzz.token_sort_ratio(qc, sc)
    tset_clean = fuzz.token_set_ratio(qc, sc)
    
    q_tokens = qc.split()
    s_tokens = sc.split()
    
    if not q_tokens or not s_tokens:
        return round(max(r, tsr)), r
        
    # Výpočet zhody slov: ako dobre každé slovo z dopytu pasuje na nejaké slovo z ulice
    q_matches = []
    for qt in q_tokens:
        best_val = 0
        for st in s_tokens:
            val = fuzz.ratio(qt, st)
            if val > best_val:
                best_val = val
        q_matches.append(best_val)
        
    s_matches = []
    for st in s_tokens:
        best_val = 0
        for qt in q_tokens:
            val = fuzz.ratio(st, qt)
            if val > best_val:
                best_val = val
        s_matches.append(best_val)
        
    q_cov = sum(q_matches) / len(q_matches)
    
    # Penalizácia za rozdiel v dĺžke (počte slov)
    len_diff_penalty = 0.04 * max(0, len(s_tokens) - len(q_tokens))
    partial_score = max(0.0, q_cov - len_diff_penalty * 100)
    
    primary = round(max(r, tsr, r_clean, tsr_clean, partial_score))
    
    # Ak je to veľmi kvalitný čiastočný match (napr. "Greschika" -> "Viktora Greschika")
    if tset_clean >= 90 and q_cov >= 85:
        primary = max(primary, round(q_cov))
        
    # Prvý a posledný znak ako tie-breaker/bonus pre vyčistené vlastné názvy
    if primary >= 55 and qc and sc:
        if qc[0] == sc[0]:
            primary += 5
        if qc[-1] == sc[-1]:
            primary += 2
            
    primary = min(100, primary)
    return primary, r


def _street_query_candidates(raw_address: str) -> list[str]:
    """Vybuduje kandidátov názvu ulice. Odstráni orientačné číslo, len ak skutočne vyzerá ako číslo domu."""
    import re
    address = raw_address.strip()
    if not address:
        return []

    candidates = []
    # Regex pre detekciu čísla domu na konci dopytu (napr. "12", "12a", "1456/12", "12/B")
    match = re.search(r'^(.*?)\s+(\d+[\w\-/]*)$', address)
    if match:
        candidates.append(match.group(1).strip())
    
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
    
    if best and second:
        if best["score"] == second["score"]:
            # Ak je skóre očistených názvov zhodné, pozrieme sa na pomer neupravených názvov na rozuzlenie remízy.
            # Ak zákazník explicitne povedal typ ulice (napr. "cesta" vs "ulica"), pomer neočistených slov
            # bude výrazne vyšší pre správnu ulicu.
            ratio_diff = best["ratio"] - second["ratio"]
            if ratio_diff >= 10 and best["ratio"] >= 95:
                margin = ratio_diff
            else:
                margin = 0
        else:
            margin = best["score"] - second["score"]
    else:
        margin = 100

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

    streets = []
    start = 0
    page_size = 1000
    while True:
        result = supabase.table("streets").select("name").range(start, start + page_size - 1).execute()
        if not result.data:
            break
        streets.extend([row["name"] for row in result.data])
        if len(result.data) < page_size:
            break
        start += page_size

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
    print(f"[check_systems] ELEVENLABS_AGENT_ID_PIZZA nastaveny: {bool(ELEVENLABS_AGENT_ID_PIZZA)}")

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
    if not ELEVENLABS_AGENT_ID and not ELEVENLABS_AGENT_ID_PIZZA:
        print("[check_systems] FAIL: Ani ELEVENLABS_AGENT_ID ani ELEVENLABS_AGENT_ID_PIZZA nie je nastaveny")
        return False, "Chyba ELEVENLABS_AGENT_ID aj ELEVENLABS_AGENT_ID_PIZZA"
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


@app.get("/twilio/token")
async def get_twilio_token():
    """
    Generates a Twilio capability token for browser-based WebRTC calls.
    Uses TWILIO_ACCOUNT_SID, TWILIO_API_KEY, and TWILIO_API_SECRET.
    """
    import time
    from twilio.jwt.access_token import AccessToken
    from twilio.jwt.access_token.grants import VoiceGrant

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    api_key = os.getenv("TWILIO_API_KEY", "").strip()
    api_secret = os.getenv("TWILIO_API_SECRET", "").strip()
    twiml_app_sid = (
        os.getenv("TWILIO_TWIML_APP_SID") or
        os.getenv("TWILIO_APP_SID") or
        os.getenv("TWIML_APP_SID") or
        os.getenv("TWILIO_TWIML_APP_ID") or
        os.getenv("TWILIO_APP_ID") or
        ""
    ).strip()

    if not account_sid or not api_key or not api_secret or not twiml_app_sid:
        print(f"[twilio/token] FAIL: Missing credentials. account_sid={bool(account_sid)}, api_key={bool(api_key)}, api_secret={bool(api_secret)}, twiml_app_sid={bool(twiml_app_sid)}")
        raise HTTPException(status_code=500, detail="Missing Twilio credentials (including TwiML App SID) on server")

    identity = f"web-user-{int(time.time())}"
    
    # Create Access Token
    token = AccessToken(account_sid, api_key, api_secret, identity=identity)
    
    # Create Voice Grant
    voice_grant = VoiceGrant(
        outgoing_application_sid=twiml_app_sid,
        incoming_allow=True
    )
    token.add_grant(voice_grant)

    print(f"[twilio/token] Token generated successfully for identity={identity}")
    return {"token": token.to_jwt(), "identity": identity}


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
    def unavailable_twiml(error_msg: str = None) -> str:
        audio_url = os.getenv("AUDIO_LINKA_NEDOSTUPNA", "").strip()
        if audio_url and not error_msg:
            return f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{xml_escape(audio_url, quote=False)}</Play>
    <Hangup/>
</Response>'''
        
        msg = "Sorry, our order line is currently unavailable. Please try again later."
        lang = "en-US"
        if error_msg:
            msg = error_msg
            
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="{lang}">{xml_escape(msg, quote=False)}</Say>
    <Pause length="1"/>
    <Hangup/>
</Response>'''

    # 1. KONTROLA SYSTÉMOV
    ok, reason = await _check_systems()
    if not ok:
        print(f"[twilio/voice] Systemy nedostupne: {reason}")
        return Response(content=unavailable_twiml(f"System error: systems unavailable: {reason}"), media_type="application/xml")

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

    # Smerovanie podla cisla alebo parametra business_type
    business_type = form_data.get("business_type") or ""
    dialed_number = called_number or to_number
    ntc_phone = os.getenv("NTC_PHONE_NUMBER", "+420910925466").strip()
    elevenlabs_ntc_agent_id = os.getenv("ELEVENLABS_NTC_AGENT_ID", "9901kv6j21rhfccr7f0nbdhew5ew").strip()

    is_ntc = (
        business_type == "taxi" or 
        (dialed_number == ntc_phone or (dialed_number and dialed_number.endswith(ntc_phone.replace("+", "")))) or
        (dialed_number and dialed_number.endswith("922442"))
    )

    if is_ntc:
        agent_id = elevenlabs_ntc_agent_id or ELEVENLABS_AGENT_ID_PIZZA or ELEVENLABS_AGENT_ID
        active_tenant_id = NTC_TENANT_ID
        menu = ""
        print(f"[twilio/voice] Routing call to NTC Voice Assistant. AgentID={agent_id}, Tenant={active_tenant_id}")
    else:
        agent_id = ELEVENLABS_AGENT_ID_PIZZA or ELEVENLABS_AGENT_ID
        active_tenant_id = TENANT_ID
        # 3. MENU Z DB -> DYNAMIC VARIABLE
        menu = format_menu_from_db(active_tenant_id)
        if not menu:
            menu = "Menu momentalne nie je dostupne."
        print(f"[twilio/voice] Routing call to Pizzeria. AgentID={agent_id}, Tenant={active_tenant_id}")

    # 4. ELEVENLABS REGISTER CALL -> HOTOVE TWIML PRE TWILIO
    try:
        from elevenlabs import ElevenLabs

        # Support separate API key for NTC if configured in environment variables
        el_api_key = (os.getenv("ELEVENLABS_NTC_API_KEY") or "").strip() if is_ntc else ""
        if not el_api_key:
            el_api_key = ELEVENLABS_API_KEY
            
        print(f"[twilio/voice] Creating ElevenLabs client using {'NTC' if (is_ntc and os.getenv('ELEVENLABS_NTC_API_KEY')) else 'default'} API key.")
        client = ElevenLabs(api_key=el_api_key)
        twiml = client.conversational_ai.twilio.register_call(
            agent_id=agent_id,
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
                    "tenant_id": active_tenant_id,
                }
            },
        )
        print("[twilio/voice] ElevenLabs register_call OK, vraciam TwiML Twiliu")
        return Response(content=twiml, media_type="application/xml")
    except Exception as e:
        import traceback
        traceback.print_exc()
        status_code = getattr(e, "status_code", None)
        body = getattr(e, "body", None)
        headers = getattr(e, "headers", None)
        
        clean_detail = ""
        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, dict):
                clean_detail = detail.get("message") or detail.get("status") or str(detail)
            else:
                clean_detail = str(detail) if detail else ""
        elif isinstance(body, str):
            clean_detail = body
            
        if not clean_detail:
            clean_detail = str(body) if body else str(e)
            
        print(f"[twilio/voice] ElevenLabs register_call failed: Status: {status_code} | Detail: {clean_detail} | Headers: {headers}")
        spoken_msg = f"Eleven Labs call registration failed. Status: {status_code or 'unknown'}. Detail: {clean_detail}"
        return Response(content=unavailable_twiml(spoken_msg), media_type="application/xml")


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
async def prompt_config(request: Request):
    """
    ElevenLabs Server URL endpoint — volá sa pred každým hovorom.
    Vracia dynamic_variables s aktuálnym menu z DB alebo neutrálne dáta pre NTC.
    """
    tenant = request.query_params.get("tenant", "pizzeria")
    
    if tenant == "ntc":
        return {
            "dynamic_variables": {
                "menu": "U nás si môžete rezervovať kurty na tenis a bedminton.",
            }
        }

    menu_text = format_menu_from_db(TENANT_ID)
    return {
        "dynamic_variables": {
            "menu": menu_text if menu_text else "Menu nie je momentálne dostupné.",
        }
    }


def build_address_message(found: bool, needs_confirmation: bool, best_match: str | None, match_type: str) -> str:
    if match_type == "ambiguous":
        return "Adresa je nejednoznacna. Poproste zakaznika, aby adresu zopakoval alebo spresnil."
    if found is True and needs_confirmation is False:
        return "Ulica najdena a potvrdzena."
    if needs_confirmation is True and best_match:
        return "Adresa je neista. Nepotvrdzujte objednavku; najprv zakaznikovi precitajte najpravdepodobnejsiu adresu a vypytajte si jasne ano/nie potvrdenie."
    return "Nerozumel som presne nazvu ulice. Poproste zakaznika, aby ulicu zopakoval po pismenach alebo povedal blizsi orientacny bod."


@app.post("/api/search-street")
async def search_street(body: SearchStreetRequest):
    """
    Fuzzy vyhladavanie ulice podla casti nazvu (STT vystup z ElevenLabs).
    Vrati found=True iba pri vysokej a jednoznacnej zhode.
    """
    # Bezpecna inicializacia pre pripad predcasneho zlyhania
    debug_info = None
    original_query = body.query.strip() if hasattr(body, 'query') else ""
    query = original_query
    street_query = original_query
    house_number = None
    had_house_number = False

    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase klient nie je inicializovany.")
    
    import re
    # Extrakcia cisla domu (ak konci na cislo s volitelnym pismenom, napr. "12A", "1456/12", "12/B")
    match = re.search(r'^(.*?)\s+(\d+[\w\-/]*)$', query)
    if match:
        street_query = match.group(1).strip()
        house_number = match.group(2)
        had_house_number = True
    else:
        street_query = query
        house_number = None
        had_house_number = False

    if not street_query:
        # Prazdny vstup
        return {
            "ok": False,
            "input": original_query,
            "query": original_query,
            "street_query": street_query,
            "house_number": house_number,
            "had_house_number": had_house_number,
            "full_address": None,
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
            "input": original_query,
            "query": original_query,
            "street_query": street_query,
            "house_number": house_number,
            "had_house_number": had_house_number,
            "full_address": None,
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

    print(f"[search-street] query='{street_query}' streets_count={len(streets)}")

    resolution = _street_resolution(street_query, streets)
    
        # DEBUG OBJEKT
    debug_info = {
        "street_min_score": _STREET_MIN_SCORE,
        "street_auto_accept_score": _STREET_AUTO_ACCEPT_SCORE,
        "street_auto_accept_margin": _STREET_AUTO_ACCEPT_MARGIN,
        "raw_query": street_query,
        "original_query": original_query,
        "street_query": street_query,
        "extracted_house_number": house_number,
        "had_house_number": had_house_number,
        "normalized_query": _normalize(street_query),
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
    top_old_style = [{"street": item["street"], "score": item["score"]} for item in resolution["suggestions"]]
    
            # Vybuduj zoznam candidates v pozadovanom formate
    for item in resolution["suggestions"]:
        street = item["street"]
        classification = classify_address_match(street_query, street, item["score"])
        candidate = {
            "address": street,
            "confidence": classification["confidence"],
            "match_type": classification["match_type"],
            "requires_confirmation": classification["requires_confirmation"],
            "reason": classification["reason"]
        }
        candidates.append(candidate)

    print(f"[search-street] top_results={top_old_style} margin={resolution['margin']} auto_accept={resolution['auto_accept']}")

    try:
        found = False
        needs_confirmation = True
        best_match = None
        confidence = 0
        match_type = "not_found"
        requires_confirmation = False
        message = "Nerozumel som presne nazvu ulice. Poproste zakaznika, aby ulicu zopakoval po pismenach alebo povedal blizsi orientacny bod."

        if not candidates:
            return {
                "ok": False,
                "input": original_query,
                "query": original_query,
                "street_query": street_query,
                "house_number": house_number,
                "had_house_number": had_house_number,
                "full_address": None,
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
                and not is_significantly_shorter):
                    best_candidate["match_type"] = "ambiguous"
                    best_candidate["requires_confirmation"] = True
                    best_candidate["reason"] = "Nájdených viacero podobných možností, nutné upresniť."

        needs_confirmation = not resolution["auto_accept"]
        if not needs_confirmation and best_candidate["match_type"] != "ambiguous":
            best_candidate["requires_confirmation"] = False

        found = not needs_confirmation
        best_match = top_old_style[0]["street"] if top_old_style else None
        confidence = top_old_style[0]["score"] if top_old_style else 0
        match_type = best_candidate["match_type"]
        
        message = build_address_message(found, needs_confirmation, best_match, match_type)
            
        full_address = None
        if top_old_style:
            best_match_str = top_old_style[0]["street"]
            if house_number:
                full_address = f"{best_match_str} {house_number}"
            else:
                full_address = best_match_str

        return {
            "ok": True,
            "input": original_query,
            "query": original_query,
            "street_query": street_query,
            "house_number": house_number,
            "had_house_number": had_house_number,
            "full_address": full_address,
            "candidates": candidates,
            "selected_candidate": {
                "address": best_candidate["address"],
                "confidence": best_candidate["confidence"],
                "match_type": best_candidate["match_type"],
                "requires_confirmation": needs_confirmation
            },
            "found": found,
            "best_match": best_match,
            "confidence": confidence,
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
            "input": original_query,
            "query": original_query,
            "street_query": street_query if 'street_query' in locals() else "",
            "house_number": house_number if 'house_number' in locals() else None,
            "had_house_number": had_house_number if 'had_house_number' in locals() else False,
            "full_address": None,
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
            "debug": debug_info if ('debug_info' in locals() and debug_info is not None) else {}
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
        if resp.status_code not in [200, 201]:
            print(f"[whatsapp] CHYBA pri odosielani: {resp.status_code}")
        return resp.status_code in [200, 201]
    except Exception as e:
        print(f"[whatsapp] CHYBA: {e}")
        return False


async def send_order_notifications_task(order_data: dict):
    """Spracuje a odosle notifikacie pre zakaznika aj restauraciu (len WhatsApp)."""
    # PREPINAC NOTIFIKACII (Elegantne vypnutie/zapnutie cez Render)
    ENABLE_WHATSAPP = os.getenv("ENABLE_WHATSAPP", "false").lower() == "true"
    
    if not ENABLE_WHATSAPP:
        print(f"[notifikacie] WhatsApp je vypnutý (ENABLE_WHATSAPP=false).")
        return
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
    
    print(f"[notifikacie] Objednávka {pizza} spracovaná.")


def normalize_sport(sport: str) -> str:
    s = str(sport or "").lower().strip()
    if "badminton" in s or "bedminton" in s:
        return "badminton"
    if "squash" in s or "skvoš" in s:
        return "squash"
    if "clay" in s or "antuka" in s:
        return "tennis-clay"
    if "tennis" in s or "tenis" in s:
        return "tennis"
    return "badminton"  # default fallback


def format_court_name(court_id: str) -> str:
    parts = court_id.split("-")
    if len(parts) < 2:
        return court_id
    sport = parts[0]
    num = parts[1].zfill(2)
    if sport == "tennis-clay":
        return f"Dvorec {num}"
    return f"Kurt {num}"


@app.post("/api/ntc-check-availability")
async def ntc_check_availability(req: CheckAvailabilityRequest):
    """
    Checks Google Calendar events to find free courts for NTC bookings.
    """
    sport_key = normalize_sport(req.sport)
    
    # Parse dates
    try:
        start_str = req.start_time_iso.replace("Z", "+00:00")
        if "+" not in start_str and "-" not in start_str.split("T")[-1]:
            # No timezone offset, assume Europe/Bratislava local time (+02:00 in summer)
            start_str += "+02:00"
        start_dt = datetime.datetime.fromisoformat(start_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chybný formát start_time_iso: {e}")

    duration = req.duration_minutes or 60
    end_dt = start_dt + datetime.timedelta(minutes=duration)

    print(f"[ntc-check] Checking availability for {sport_key} from {start_dt.isoformat()} to {end_dt.isoformat()}")

    # Buffer of 1 minute to avoid boundaries overlap
    time_min = (start_dt + datetime.timedelta(minutes=1)).isoformat()
    time_max = (end_dt - datetime.timedelta(minutes=1)).isoformat()
    
    from google_calendar import list_calendar_events
    events = await list_calendar_events(NTC_TENANT_ID, time_min, time_max)

    busy_courts = set()
    for event in events:
        summary = event.get("summary", "")
        description = event.get("description", "")
        
        # Check description and summary for court ID
        court_id = None
        
        # Parse description
        for line in description.split("\n"):
            parts = line.split(":")
            if len(parts) >= 2:
                key = parts[0].strip().lower()
                val = ":".join(parts[1:]).strip()
                if key in ["kurt id", "court id", "courtid", "court"]:
                    court_id = val.strip().lower()
                    break
        
        if not court_id:
            import re
            court_pattern = r"(badminton|squash|tennis|tennis-clay)-\d+"
            desc_match = re.search(court_pattern, description, re.IGNORECASE)
            if desc_match:
                court_id = desc_match.group(0).lower()
            else:
                summary_match = re.search(court_pattern, summary, re.IGNORECASE)
                if summary_match:
                    court_id = summary_match.group(0).lower()

        if court_id:
            busy_courts.add(court_id)

    # Determine court capacity
    # badminton: 14 courts, tennis: 8, squash: 4, clay: 4
    limit = 14 if sport_key == "badminton" else (8 if sport_key == "tennis" else 4)
    all_sport_courts = [f"{sport_key}-{i}" for i in range(1, limit + 1)]
    free_courts = [c for c in all_sport_courts if c not in busy_courts]

    if not free_courts:
        return {
            "status": "busy",
            "message": f"Pre {req.sport} v tomto čase nie sú žiadne voľné kurty.",
            "free_courts": []
        }

    free_court_names = [format_court_name(c) for c in free_courts]
    
    return {
        "status": "available",
        "message": f"Pre {req.sport} sú voľné nasledovné kurty: {', '.join(free_court_names)}.",
        "free_courts": free_courts,
        "free_court_names": free_court_names
    }


async def send_ntc_booking_notification(phone: str, sport: str, court_id: str, start_iso: str, duration: int):
    """Odosle WhatsApp notifikaciu o rezervacii kurtu v NTC cez Twilio REST API"""
    ENABLE_WHATSAPP = os.getenv("ENABLE_WHATSAPP", "false").lower() == "true"
    if not ENABLE_WHATSAPP:
        print("[ntc-notifikacie] WhatsApp je vypnuty (ENABLE_WHATSAPP=false).")
        return False

    TPL_NTC_CUSTOMER = os.getenv("TWILIO_TPL_NTC_CUSTOMER", "HXf06a4115a2733f1bba940a857301c106").strip()
    if not phone:
        print("[ntc-notifikacie] Chybajuce telefonne cislo zakaznika, neodosielam.")
        return False

    # 1. Format sport
    sport_map = {
        "tennis": "Tenis",
        "tennis-clay": "Tenis antuka",
        "badminton": "Bedminton",
        "squash": "Squash"
    }
    sport_formatted = sport_map.get(sport.lower().strip(), sport)

    # 2. Format court
    court_formatted = format_court_name(court_id)

    # 3. Format date & time (in Europe/Bratislava timezone)
    try:
        # Standard input: e.g. 2026-06-21T10:00:00+02:00 or 2026-06-21T10:00:00
        clean_start = start_iso.replace("Z", "+00:00")
        if "+" not in clean_start and "-" not in clean_start.split("T")[-1]:
            # No offset, assume Europe/Bratislava local time
            clean_start += "+02:00"
        
        start_dt = datetime.datetime.fromisoformat(clean_start)
        
        # Date format: e.g., "15. 10. 2026"
        date_formatted = start_dt.strftime("%d. %m. %Y").replace(" 0", " ")
        if date_formatted.startswith("0"):
            date_formatted = date_formatted[1:]
            
        end_dt = start_dt + datetime.timedelta(minutes=duration)
        time_formatted = f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}"
    except Exception as e:
        print(f"[ntc-notifikacie] Chyba pri formatovani datumu/casu ({start_iso}): {e}")
        date_formatted = start_iso.split("T")[0] if "T" in start_iso else start_iso
        time_formatted = f"{duration} min"

    msg_body = f"Potvrdzujeme rezerváciu kurtu. Šport: {sport_formatted}, Kurt: {court_formatted}, Dátum: {date_formatted}, Čas: {time_formatted}"
    
    # Premenné pre novú schválenú šablónu (zakaznik_potvrdenie_ntc):
    # {{1}} -> Šport, {{2}} -> Kurt, {{3}} -> Dátum, {{4}} -> Čas
    vars_cust = {
        "1": sport_formatted,
        "2": court_formatted,
        "3": date_formatted,
        "4": time_formatted
    }

    print(f"[ntc-notifikacie] Posielam NTC WA notifikaciu na {phone}: {msg_body}")
    return await send_whatsapp_message(phone, msg_body, TPL_NTC_CUSTOMER, vars_cust)


@app.post("/api/ntc-create-booking")
async def ntc_create_booking(req: CreateBookingRequest, background_tasks: BackgroundTasks):
    """
    Saves a booking in the Supabase bookings table and Google Calendar.
    """
    sport_key = normalize_sport(req.sport)
    
    try:
        start_str = req.start_time_iso.replace("Z", "+00:00")
        if "+" not in start_str and "-" not in start_str.split("T")[-1]:
            # No timezone offset, assume Europe/Bratislava local time (+02:00 in summer)
            start_str += "+02:00"
        start_dt = datetime.datetime.fromisoformat(start_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chybný formát start_time_iso: {e}")

    duration = req.duration_minutes or 60
    end_dt = start_dt + datetime.timedelta(minutes=duration)

    # 1. Save to Supabase bookings table
    import json
    notes_obj = {
        "courtId": req.court_id,
        "source": "voice-assistant",
        "notes": req.notes or "Rezervácia cez hlasového asistenta"
    }

    booking_data = {
        "tenant_id": NTC_TENANT_ID,
        "customer_name": req.customer_name,
        "customer_phone": req.customer_phone or req.caller_number or "",
        "start_at": start_dt.isoformat(),
        "end_at": end_dt.isoformat(),
        "status": "confirmed",
        "notes": json.dumps(notes_obj)
    }

    try:
        db_res = supabase.table("bookings").insert(booking_data).execute()
        if not db_res.data:
            raise Exception("Chyba: DB nevrátila žiadne dáta.")
        db_booking = db_res.data[0]
    except Exception as db_err:
        print(f"[ntc-booking] Database insertion failed: {db_err}")
        raise HTTPException(status_code=500, detail=f"Zápis do Supabase zlyhal: {db_err}")

    # 2. Sync to Google Calendar
    court_label = req.court_id.replace("-", " ").upper()
    summary = f"Rezervácia: {court_label} ({req.customer_name})"
    
    description = "\n".join([
        f"Kurt ID: {req.court_id}",
        f"Zákazník: {req.customer_name}",
        f"Telefón: {req.customer_phone or req.caller_number or 'Neznáme'}",
        "Kanál: Hlas Telio",
        f"Poznámka: {req.notes or ''}"
    ])

    from google_calendar import create_calendar_event
    calendar_event_id = await create_calendar_event(
        tenant_id=NTC_TENANT_ID,
        summary=summary,
        description=description,
        start_iso=start_dt.isoformat(),
        end_iso=end_dt.isoformat(),
        color_id="7"  # Peacock (light blue) for voice reservations
    )

    # 3. Update database record with calendar_event_id
    if calendar_event_id:
        try:
            supabase.table("bookings") \
                .update({"calendar_event_id": calendar_event_id}) \
                .eq("id", db_booking["id"]) \
                .execute()
        except Exception as update_err:
            print(f"[ntc-booking] Failed to update calendar_event_id in DB: {update_err}")

    # 4. Send WhatsApp Notification to Customer on Background
    customer_phone = req.customer_phone or req.caller_number
    if customer_phone:
        print(f"[ntc-booking] Planujem odoslanie WhatsApp notifikacie na {customer_phone}")
        background_tasks.add_task(
            send_ntc_booking_notification,
            phone=customer_phone,
            sport=req.sport,
            court_id=req.court_id,
            start_iso=req.start_time_iso,
            duration=duration
        )

    court_name_spoken = format_court_name(req.court_id)
    return {
        "status": "success",
        "message": f"Rezervácia pre {req.customer_name} na {court_name_spoken} bola úspešne vytvorená.",
        "booking_id": calendar_event_id or db_booking["id"]
    }


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


@app.post("/api/request-human-fallback")
@app.post("/api/request-human")
async def request_human_fallback(request: Request):
    """
    ElevenLabs tool endpoint pre vyziadanie human fallback (spojenie s obsluhou).
    Odosle WhatsApp notifikaciu restauracii.
    """
    try:
        body = await request.json()
        print(f"[fallback] raw body: {body}")
        req_data = HumanFallbackRequest(**body)
    except Exception as e:
        print(f"[fallback] validacna chyba: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    try:
        # Ziskanie a normalizacia cisla zakaznika
        raw_caller = req_data.caller_number or body.get("caller_number") or ""
        
        # Ak nam ElevenLabs poslal "unknown" alebo neplatne cislo (nema ziadne cislice),
        # alebo ak je cislo prazdne ci patriace Twiliu, skusime ho ziskat inak:
        normalized_caller = _normalize_phone(raw_caller)
        is_invalid = (
            not normalized_caller 
            or "unknown" in normalized_caller.lower() 
            or not any(c.isdigit() for c in normalized_caller)
            or _is_twilio_owned_number(normalized_caller)
        )
        
        if is_invalid:
            # 1. Skusime vytiahnut z dynamic_variables od ElevenLabs (ak su pritomne)
            dyn_vars = body.get("dynamic_variables", {})
            el_caller = _normalize_phone(dyn_vars.get("caller_number") or dyn_vars.get("from_number") or "")
            
            if (
                el_caller 
                and not _is_twilio_owned_number(el_caller) 
                and "unknown" not in el_caller.lower() 
                and any(c.isdigit() for c in el_caller)
            ):
                caller_number = el_caller
            # 2. Ak nemame cislo z dynamic_variables, pouzijeme globalny _LAST_CALLER_PHONE
            elif _LAST_CALLER_PHONE and not _is_twilio_owned_number(_LAST_CALLER_PHONE):
                caller_number = _LAST_CALLER_PHONE
            else:
                caller_number = ""
        else:
            caller_number = normalized_caller

        reason = req_data.reason or "-"
        print(f"[fallback] Vyziadana obsluha pre zakaznika '{caller_number}', dovod: '{reason}'")

        # Zápis do Supabase tabuľky fallback_requests
        try:
            fallback_data = {
                "tenant_id": TENANT_ID,
                "customer_phone": caller_number,
                "reason": reason,
                "status": "NEW"
            }
            supabase.table("fallback_requests").insert(fallback_data).execute()
            print(f"[fallback] Zápis do fallback_requests bol úspešný.")
        except Exception as db_err:
            print(f"[fallback] Varovanie: Zápis do DB zlyhal: {db_err}")

        ENABLE_WHATSAPP = os.getenv("ENABLE_WHATSAPP", "false").lower() == "true"
        if ENABLE_WHATSAPP:
            RESTAURANT_PHONE = os.getenv("RESTAURANT_PHONE", "+421910922442")
            TPL_RESTAURANT_FALLBACK = os.getenv("TWILIO_TPL_RESTAURANT_FALLBACK")
            
            msg_rest = f"⚠️ *ŽIADOSŤ O KONTAKT* \n\nZákazník na čísle {caller_number} žiada o rozhovor s obsluhou.\nDôvod: {reason}"
            vars_rest = {"1": caller_number, "2": reason}
            
            await send_whatsapp_message(RESTAURANT_PHONE, msg_rest, TPL_RESTAURANT_FALLBACK, vars_rest)
            print(f"[fallback] WhatsApp notifikacia odoslana restauracii.")
        else:
            print(f"[fallback] WhatsApp je vypnuty, neodosielam notifikaciu.")

        return {"status": "success", "message": "Obsluha bola notifikovana."}
    except Exception as e:
        print(f"[fallback] CHYBA: {e}")
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
