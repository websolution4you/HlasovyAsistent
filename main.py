import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
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

if not SUPABASE_URL or not SUPABASE_KEY:
    print("--- CHYBA KONFIGURACIE ---")
    if not SUPABASE_URL:
        print("Chyba premenna: CORE_SUPABASE_URL")
    if not SUPABASE_KEY:
        print("Chyba premenna: CORE_SUPABASE_SERVICE_ROLE_KEY")
    print("--------------------------")

try:
    # Inicializacia Supabase klienta
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Chyba pri inicializacii Supabase: {e}")
    supabase = None


class PizzaOrder(BaseModel):
    pizza_type: str
    upsell: str = "ziadny"
    address: str
    phone_number: str  # Pridané pole pre telefón


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
    <Say voice="alice" language="en-US">{message}</Say>
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
    <Say voice="alice" language="en-US">Ospravedlnujeme sa, linka je docasne nedostupna. Skuste prosim zavolat neskor.</Say>
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


@app.post("/api/vytvor-objednavku")
async def vytvor_objednavku(order: PizzaOrder):
    """
    Endpoint pre ElevenLabs agenta na ulozenie objednavky do DB.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase klient nie je inicializovany.")

    try:
        order_data = {
            "pizza": order.pizza_type,
            "upsell": order.upsell,
            "adresa": order.address,
            "telefon": order.phone_number  # Zápis čísla do DB
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
