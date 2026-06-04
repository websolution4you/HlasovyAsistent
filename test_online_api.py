import urllib.request
import urllib.error
import json
import sys

# Zaistenie UTF-8 kódovania pre Windows terminál
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "https://hlasovyasistent-652999054235.europe-west3.run.app"

# Testovacie prípady (vstupy od zákazníka)
TEST_CASES = [
    "hlavna",
    "hlvana 12",
    "mlynska 5",
    "stefanikova 24A",
    "domanovce 2",
    "neexistujuca ulica v databaze"
]

def check_health():
    print("=" * 90)
    print(f"KONTROLA DOSTUPNOSTI ONLINE SERVERA: {BASE_URL}")
    print("=" * 90)
    
    # 1. Test GET /health
    try:
        req = urllib.request.Request(f"{BASE_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            body = json.loads(response.read().decode('utf-8'))
            print(f"GET /health: STATUS {status} | Odpoved: {body}")
    except Exception as e:
        print(f"GET /health zlyhal: {e}")
        
    # 2. Test GET /health/config
    try:
        req = urllib.request.Request(f"{BASE_URL}/health/config", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            body = json.loads(response.read().decode('utf-8'))
            print(f"GET /health/config: STATUS {status} | Konfiguracia: {body}")
    except Exception as e:
        print(f"GET /health/config zlyhal (mozno je endpoint chraneny alebo nedostupny): {e}")
    print("=" * 90 + "\n")

def run_online_tests():
    print("=" * 90)
    print("SPUSTAM TESTOVANIE OVEROVANIA ADRESY PROTI ONLINE CLOUD RUN API")
    print("=" * 90)
    
    # Hlavička tabuľky
    print(f"{'Vstup od zakaznika':<30} | {'Vysledok overenia (Matched)':<30} | {'Found':<5} | {'Needs Conf':<10}")
    print("-" * 90)
    
    url = f"{BASE_URL}/api/search-street"
    
    for query in TEST_CASES:
        payload = {"query": query}
        data = json.dumps(payload).encode('utf-8')
        
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                resp_body = json.loads(response.read().decode('utf-8'))
                
                # Extrakcia vysledkov
                found = "ANO" if resp_body.get("found") else "NIE"
                needs_conf = "ANO" if resp_body.get("needs_confirmation") else "NIE"
                
                selected = resp_body.get("selected_candidate")
                matched_address = "Ziadna"
                if selected and isinstance(selected, dict):
                    matched_address = selected.get("address", "Ziadna")
                elif resp_body.get("best_match"):
                    matched_address = resp_body.get("best_match")
                
                # Zobrazenie cisla domu ak je v full_address
                if resp_body.get("full_address"):
                    matched_address = resp_body.get("full_address")
                    
                print(f"{query:<30} | {matched_address:<30} | {found:<5} | {needs_conf:<10}")
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            print(f"{query:<30} | CHYBA SERVERA {e.code}: {error_body[:40]:<30} | -     | -")
        except Exception as e:
            print(f"{query:<30} | NEOCAKAVANA CHYBA: {str(e)[:40]:<30} | -     | -")
            
    print("=" * 90)

if __name__ == "__main__":
    check_health()
    run_online_tests()
