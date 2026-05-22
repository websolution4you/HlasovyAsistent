import sys
import os
import csv

# Zaistenie UTF-8 kódovania pre Windows terminál
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Pridame aktualny adresar do sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import main
from test_accuracy import load_streets, make_typo, get_partial_names

def run_sweep():
    streets = load_streets()
    
    # Vygenerujeme vopred testovacie prípady, aby bol sweep rýchly a konzistentný
    test_cases = []
    
    for street in streets:
        # 1. Presné názvy
        test_cases.append((street, street, "exact"))
        
        # 2. Preklepy
        for n in [1, 2, 3, 4, 5]:
            typo = make_typo(street, n)
            if typo != street:
                test_cases.append((typo, street, f"typo{n}"))
                
        # 3. Čiastočné názvy
        partials = get_partial_names(street)
        for part in partials:
            test_cases.append((part, street, "partial"))
            
    print(f"Vygenerovaných {len(test_cases)} testovacích prípadov.")
    print("Spúšťam sweep parametrov pre zistenie optimálneho nastavenia...\n")
    
    # Skúsime rôzne hodnoty pre _STREET_AUTO_ACCEPT_SCORE a _STREET_AUTO_ACCEPT_MARGIN
    print(f"{'Auto-Score Threshold':<20} | {'Auto-Margin':<12} | {'Auto-Akceptované':<16} | {'Nebezpečné Auto-Akcepty':<24} | {'Komentár'}")
    print("-" * 90)
    
    best_config = None
    max_auto_accepted = -1
    
    for score_thresh in [80, 82, 85, 87, 88, 90, 92, 95]:
        for margin_thresh in [0, 2, 3, 5, 8, 10]:
            # Dočasne prepíšeme globálne nastavenia v main.py
            main._STREET_AUTO_ACCEPT_SCORE = score_thresh
            main._STREET_AUTO_ACCEPT_MARGIN = margin_thresh
            
            auto_accepted_count = 0
            dangerous_auto_accepts = []
            
            for query, original_street, category in test_cases:
                resolution = main._street_resolution(query, streets)
                best = resolution["best"]
                auto_accept = resolution["auto_accept"]
                
                if best:
                    matched_street = best["street"]
                    is_correct = main._normalize(matched_street) == main._normalize(original_street)
                    
                    if is_correct:
                        if auto_accept:
                            auto_accepted_count += 1
                    else:
                        if auto_accept:
                            # Nebezpečný prípad: priradili sme zlú ulicu a automaticky ju akceptovali
                            # Vylúčime známe defaulty (Levočská/Vinica) pre spravodlivé porovnanie
                            if query not in ["Levočská", "Vinica"]:
                                dangerous_auto_accepts.append((query, original_street, matched_street, score_thresh, margin_thresh))
                                
            # Vyhodnotenie
            num_dangerous = len(dangerous_auto_accepts)
            status_comment = ""
            if num_dangerous == 0:
                status_comment = "🟢 Bezpečné! (0 chýb)"
                if auto_accepted_count > max_auto_accepted:
                    max_auto_accepted = auto_accepted_count
                    best_config = (score_thresh, margin_thresh)
            else:
                status_comment = f"🔴 NEBEZPEČNÉ ({num_dangerous} chýb)"
                
            print(f"{score_thresh:<20} | {margin_thresh:<12} | {auto_accepted_count:<16} | {num_dangerous:<24} | {status_comment}")
            
    print("\n" + "=" * 90)
    print("VÝSLEDOK HĽADANIA OPTIMÁLNEHO NASTAVENIA:")
    print("=" * 90)
    if best_config:
        print(f"Najlepšie bezpečné nastavenie (s maximálnym počtom automatických akceptácií):")
        print(f"  👉 _STREET_AUTO_ACCEPT_SCORE = {best_config[0]}")
        print(f"  👉 _STREET_AUTO_ACCEPT_MARGIN = {best_config[1]}")
        print(f"  Počet automaticky akceptovaných dopytov bez nutnosti otravovať zákazníka: {max_auto_accepted} z celkovo {len(test_cases)}")
    else:
        print("Nepodarilo sa nájsť žiadne stopercentne bezpečné nastavenie.")
    print("=" * 90)

if __name__ == "__main__":
    run_sweep()
