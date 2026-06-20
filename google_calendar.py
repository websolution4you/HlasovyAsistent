import os
import datetime
import httpx
from supabase import create_client, Client

SUPABASE_URL = os.getenv("CORE_SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("CORE_SUPABASE_SERVICE_ROLE_KEY", "").strip()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

# Initialize localized supabase client if keys are present
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def _get_connection(tenant_id: str):
    if not supabase:
        print("[gcal] Supabase client is not configured!")
        return None

    try:
        response = supabase.table("calendar_connections") \
            .select("*") \
            .eq("tenant_id", tenant_id) \
            .eq("provider", "google") \
            .execute()
        
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        print(f"[gcal] Failed to fetch calendar connection from DB: {e}")
        return None

def get_access_token(tenant_id: str) -> str:
    connection = _get_connection(tenant_id)
    if not connection:
        print(f"[gcal] No Google Calendar connection details found for tenant {tenant_id}")
        return ""

    access_token = connection.get("access_token")
    refresh_token = connection.get("refresh_token")
    token_expiry = connection.get("token_expiry")

    # Check if token is expired or will expire in the next 60 seconds
    is_expired = True
    if token_expiry:
        try:
            # Parse ISO timestamp (handling both Z and +00 offset formats)
            expiry_str = token_expiry.replace("Z", "+00:00")
            expiry_dt = datetime.datetime.fromisoformat(expiry_str)
            now_dt = datetime.datetime.now(datetime.timezone.utc)
            if expiry_dt > now_dt + datetime.timedelta(seconds=60):
                is_expired = False
        except Exception as parse_err:
            print(f"[gcal] Failed to parse expiry date '{token_expiry}': {parse_err}")

    if not is_expired and access_token:
        return access_token

    # Token is expired, refresh it
    if not refresh_token:
        print(f"[gcal] Cannot refresh token, refresh_token is missing for tenant {tenant_id}")
        return ""

    print(f"[gcal] Access token is expired. Refreshing token for tenant {tenant_id}...")
    try:
        url = "https://oauth2.googleapis.com/token"
        payload = {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        
        resp = httpx.post(url, data=payload, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            new_access_token = data.get("access_token")
            expires_in = data.get("expires_in", 3600)
            
            # Calculate new expiry time
            new_expiry_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=expires_in)
            new_expiry = new_expiry_dt.isoformat()

            # Save back to Supabase
            supabase.table("calendar_connections") \
                .update({
                    "access_token": new_access_token,
                    "token_expiry": new_expiry
                }) \
                .eq("id", connection["id"]) \
                .execute()

            print(f"[gcal] Token refreshed and saved successfully for tenant {tenant_id}")
            return new_access_token
        else:
            print(f"[gcal] Google refresh token request failed: {resp.status_code} {resp.text}")
            return ""
    except Exception as e:
        print(f"[gcal] Exception occurred during token refresh: {e}")
        return ""

async def list_calendar_events(tenant_id: str, time_min_iso: str, time_max_iso: str) -> list:
    access_token = get_access_token(tenant_id)
    if not access_token:
        print("[gcal] Unable to fetch events: no access token available")
        return []

    connection = _get_connection(tenant_id)
    calendar_id = connection.get("calendar_id") if connection else "primary"
    if not calendar_id:
        calendar_id = "primary"

    url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "timeMin": time_min_iso,
        "timeMax": time_max_iso,
        "singleEvents": "true",
        "orderBy": "startTime"
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, params=params, timeout=10.0)
            
        if resp.status_code == 200:
            return resp.json().get("items", [])
        else:
            print(f"[gcal] Failed to list calendar events: {resp.status_code} {resp.text}")
            return []
    except Exception as e:
        print(f"[gcal] Exception occurred listing calendar events: {e}")
        return []

async def create_calendar_event(tenant_id: str, summary: str, description: str, start_iso: str, end_iso: str, color_id: str = "1") -> str:
    access_token = get_access_token(tenant_id)
    if not access_token:
        print("[gcal] Unable to create event: no access token available")
        return ""

    connection = _get_connection(tenant_id)
    calendar_id = connection.get("calendar_id") if connection else "primary"
    if not calendar_id:
        calendar_id = "primary"

    url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
        "colorId": color_id
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=body, timeout=10.0)
            
        if resp.status_code in [200, 201]:
            event_id = resp.json().get("id")
            print(f"[gcal] Event successfully created in Google Calendar: {event_id}")
            return event_id
        else:
            print(f"[gcal] Failed to create Google Calendar event: {resp.status_code} {resp.text}")
            return ""
    except Exception as e:
        print(f"[gcal] Exception occurred creating calendar event: {e}")
        return ""
