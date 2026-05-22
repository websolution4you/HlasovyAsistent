import sys
import os

# Zaistenie UTF-8 kódovania pre Windows terminál
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Pridame aktualny adresar do sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    import main
    from main import _street_resolution, _normalize, classify_address_match
except ImportError as e:
    print(f"Chyba pri importe main.py: {e}")
    sys.exit(1)

# Definovanie testovacích ulíc (mock databáza pre prípad, že nepoužijeme Supabase)
MOCK_STREETS = [
    "Hlavná",
    "Mlynská",
    "Levočská",
    "Spišská",
    "Štefánikova",
    "Francisciho",
    "Záhradná",
    "Domanovce",
    "SNP",
    "Partizánska",
    "Okružná",
    "Duklianska"
]

# Testovacie prípady (vstupy od zákazníka) a očakávané správanie
TEST_CASES = [
    # Vstup, Očakávaná ulica, Poznámka
    ("hlavna", "Hlavná", "Bez diakritiky, male pismena"),
    ("hlvana 12", "Hlavná", "Preklep a cislo domu"),
    ("mlynska 5", "Mlynská", "Bez diakritiky s cislom domu"),
    ("stefanikova 24A", "Štefánikova", "Bez diakritiky, cislo s pismenom"),
    ("francisciho", "Francisciho", "Presna zhoda bez diakritiky"),
    ("zahradna", "Záhradná", "Bez diakritiky"),
    ("domanovce 2", "Domanovce", "Nazov obce s cislom"),
    ("es en pe 4", "SNP", "Foneticky prepis SNP (ak by ho niekto takto povedal)"),
    ("okruzna", "Okružná", "Bez diakritiky"),
    ("duklianska", "Duklianska", "Presna zhoda"),
    ("neexistujuca ulica", None, "Ulica, ktora nie je v databaze")
]

def run_tests():
    print("=" * 90)
    print("SPUSTAM TESTOVANIE FUZZY MATCHINGU ADRESY (OFFLINE MOCK REZIM)")
    print("=" * 90)
    
    # Hlavička tabuľky (bez diakritiky pre konzolu)
    print(f"{'Vstup od zakaznika':<22} | {'Najlepsia zhoda':<15} | {'Skore':<5} | {'Margin':<6} | {'Auto-Accept':<14} | {'Typ zhody':<12}")
    print("-" * 90)
    
    for query, expected, note in TEST_CASES:
        # Spustíme rovnakú logiku ako v main.py
        resolution = _street_resolution(query, MOCK_STREETS)
        best = resolution["best"]
        margin = resolution["margin"]
        auto_accept = resolution["auto_accept"]
        
        best_street = best["street"] if best else "Ziadna"
        score = best["score"] if best else 0
        
        # Určenie typu zhody
        match_type = "not_found"
        if best:
            classification = classify_address_match(query, best["street"], best["score"])
            match_type = classification["match_type"]
            
        auto_accept_str = "ANO" if auto_accept else "NIE (Potvrdit)"
        
        print(f"{query:<22} | {best_street:<15} | {score:<5} | {margin:<6} | {auto_accept_str:<14} | {match_type:<12}")
        
    print("=" * 90)
    print("Poznamka: 'Auto-Accept = NIE' znamena, ze asistent sa zakaznika opyta na potvrdenie adresy.")
    print("=" * 90)

if __name__ == "__main__":
    run_tests()

