import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

# Načítanie environment premenných (užitočné pre lokálny vývoj)
load_dotenv()

app = FastAPI(title="ElevenLabs Pizza Webhook")

# --- KONFIGURÁCIA SUPABASE ---
SUPABASE_URL = os.getenv("CORE_SUPABASE_URL")
SUPABASE_KEY = os.getenv("CORE_SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    # V produkcii na Renderi je dôležité mať tieto premenné nastavené
    print("VAROVANIE: Chýbajú environment premenné SUPABASE_URL alebo SUPABASE_KEY!")

try:
    # Inicializácia Supabase klienta
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Chyba pri inicializácii Supabase: {e}")
    supabase = None

# --- PYDANTIC MODELY ---
class PizzaOrder(BaseModel):
    pizza_type: str
    upsell: str = "žiadny"
    address: str

# --- ENDPOINTY ---

@app.get("/")
async def health_check():
    """
    Jednoduchý health check endpoint pre Render.
    """
    return {"status": "online", "message": "Server beží."}

@app.post("/api/vytvor-objednavku")
async def vytvor_objednavku(order: PizzaOrder):
    """
    Endpoint pre ElevenLabs agenta na uloženie objednávky do DB.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase klient nie je inicializovaný.")

    try:
        # Príprava dát pre Supabase (mapovanie modelov na názvy stĺpcov v DB)
        order_data = {
            "pizza": order.pizza_type,
            "upsell": order.upsell,
            "adresa": order.address
        }

        # Zápis do tabuľky 'pizza_orders' (zmena názvu pre integráciu)
        response = supabase.table("pizza_orders").insert(order_data).execute()

        # ElevenLabs očakáva jednoduchú a jasnú odpoveď
        return {
            "status": "success",
            "message": "Objednávka úspešne zapísaná."
        }

    except Exception as e:
        print(f"Chyba pri zápise do databázy: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Nepodarilo sa zapísať objednávku: {str(e)}"
        )

# Inštrukcia pre Render Start Command:
# uvicorn main:app --host 0.0.0.0 --port $PORT
