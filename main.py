import os
import time
import difflib
import unicodedata
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# Nacitanie environment premennych (uzitocne pre lokalny vyvoj)
load_dotenv()

app = FastAPI(title="ElevenLabs Pizza Webhook")

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
    customer_phone: str = ""
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


def _partial_ratio(query: str, target: str) -> float:
    """Najlepší SequenceMatcher.ratio() pre query ako okno v target (partial match)."""
    if not query or not target:
        return 0.0
    if len(query) > len(target):
        return difflib.SequenceMatcher(None, query, target).ratio()
    best = 0.0
    for i in range(len(target) - len(query) + 1):
        r = difflib.SequenceMatcher(None, query, target[i:i + len(query)]).ratio()
        if r > best:
            best = r
            if best == 1.0:
                break
    return best


def _street_score(query: str, street: str) -> int:
    """Vráti skóre 0–100 (max z full ratio a partial ratio)."""
    q = _normalize(query)
    s = _normalize(street)
    full = difflib.SequenceMatcher(None, q, s).ratio()
    partial = _partial_ratio(q, s)
    return round(max(full, partial) * 100)


def _get_streets_cached(tenant_id: str) -> list[str]:
    """Načíta ulice z DB, výsledok cachuje na 5 minút."""
    now = time.monotonic()
    if (
        _STREETS_CACHE["tenant_id"] == tenant_id
        and _STREETS_CACHE["data"]
        and now - _STREETS_CACHE["timestamp"] < _CACHE_TTL
    ):
        return _STREETS_CACHE["data"]

    result = supabase.table("streets").select("name").eq("tenant_id", tenant_id).execute()
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
        result = supabase.table("streets").select("name").eq("tenant_id", tenant_id).execute()
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
            matches = difflib.get_close_matches(
                street_part.lower(), street_names_lower.keys(), n=1, cutoff=0.6
            )
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


@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice_webhook():
    """
    Zakladny Twilio voice webhook pre prichadzajuce hovory.
    Zatial vracia jednoduche TwiML, aby cislo smerovalo na tento projekt.
    """
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
async def search_street(body: SearchStreetRequest):
    """
    Fuzzy vyhľadávanie ulice podľa časti názvu (STT výstup z ElevenLabs).
    Cachuje ulice z DB na 5 minút. Vracia max 2 zhody so skóre ≥ 55.
    """
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
async def vytvor_objednavku(order: ManageOrder):
    """
    Endpoint pre ElevenLabs agenta (manage_order tool) na ulozenie objednavky do DB.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase klient nie je inicializovany.")

    try:
        matched_address, confidence = match_street(order.delivery_address, TENANT_ID)

        order_data = {
            "tenant_id": TENANT_ID,
            "customer_phone": order.customer_phone,
            "phone_raw": order.customer_phone,
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

        return {
            "status": "success",
            "message": "Objednavka uspesne zapisana.",
        }

    except Exception as e:
        error_msg = str(e)
        print(f"DEBUG CHYBA: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Chyba Supabase: {error_msg}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
