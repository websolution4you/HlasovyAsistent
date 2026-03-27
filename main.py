import os
import difflib
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


class ManageOrder(BaseModel):
    pizza_type: str
    total_price: float
    delivery_address: str
    customer_phone: str = ""
    upsell_item: Optional[str] = None
    upsell_accepted: bool = False


ALLERGEN_MAP = {
    "1": "lepok", "2": "kôrovce", "3": "vajcia", "4": "ryby",
    "5": "arašidy", "6": "sója", "7": "mlieko", "8": "orechy",
    "9": "zeler", "10": "horčica", "11": "sezam", "12": "oxid siričitý",
    "13": "vlčí bôb", "14": "mäkkýše",
}


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
    """Fuzzy match adresy voči tabuľke ulíc. Vracia (matched_address, confidence 0-1)."""
    if not supabase or not tenant_id:
        return raw_address, 0

    try:
        result = supabase.table("streets").select("name").eq("tenant_id", tenant_id).execute()
        if not result.data:
            return raw_address, 0

        street_names = [s["name"] for s in result.data]

        # Skus oddeliť číslo domu z konca adresy
        parts = raw_address.strip().rsplit(maxsplit=1)
        if len(parts) == 2 and any(c.isdigit() for c in parts[1]):
            street_part = parts[0]
            house_number = parts[1]
        else:
            street_part = raw_address.strip()
            house_number = ""

        # Fuzzy match voči názvom ulíc
        street_names_lower = {name.lower(): name for name in street_names}
        matches = difflib.get_close_matches(
            street_part.lower(), street_names_lower.keys(), n=1, cutoff=0.6
        )

        if matches:
            matched_name = street_names_lower[matches[0]]
            matched_address = f"{matched_name} {house_number}".strip() if house_number else matched_name
            return matched_address, 1

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
            "pizza_type": order.pizza_type,
            "total_price": order.total_price,
            "delivery_address": matched_address,
            "address_raw": order.delivery_address,
            "address_confidence": confidence,
            "upsell_offered": order.upsell_item is not None,
            "upsell_item": order.upsell_item,
            "upsell_accepted": order.upsell_accepted,
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


# Instrukcia pre Render Start Command:
# uvicorn main:app --host 0.0.0.0 --port $PORT
