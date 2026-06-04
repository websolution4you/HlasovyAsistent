import sys
import os
import csv

# Ensure UTF-8 output for Windows Terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import main
from main import _street_resolution, _normalize
from test_accuracy import make_typo

def load_streets():
    streets = []
    with open("streets_database.csv", mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            streets.append(row['name'])
    return streets

def test_typos():
    streets = load_streets()
    
    print("=" * 120)
    print("ANALÝZA 2, 3, 4 A 5-ZNAKOVÝCH PREKLEPOV PRE VŠETKY ULICE")
    print("=" * 120)
    
    failures = []
    success_samples = []
    
    for street in streets:
        for num_typos in [2, 3, 4, 5]:
            t = make_typo(street, num_typos)
            if t != street:
                res = _street_resolution(t, streets)
                best = res["best"]
                matched = best["street"] if best else "Nenájdené"
                score = best["score"] if best else 0
                auto = res["auto_accept"]
                
                if _normalize(matched) == _normalize(street):
                    # Zoberieme pár vzoriek pre každú úroveň preklepov pre ukážku
                    if len([x for x in success_samples if x[2] == num_typos]) < 3:
                        success_samples.append((street, t, num_typos, matched, score, auto))
                else:
                    failures.append((street, t, num_typos, matched, score, auto))

    print(f"\n✅ UKÁŽKA ÚSPEŠNÝCH PÁROVANÍ (Systém správne priradil ulicu napriek degradácii):")
    print("-" * 120)
    print(f"{'Originálna ulica':<30} | {'Dopyt s preklepmi':<25} | {'Preklepy':<8} | {'Priradené v DB':<25} | {'Skóre':<5} | {'Auto'}")
    print("-" * 120)
    # Zoradíme úspešné ukážky podľa počtu preklepov
    success_samples.sort(key=lambda x: x[2])
    for orig, query, num, matched, score, auto in success_samples:
        auto_str = "🟢 ÁNO" if auto else "🟡 Potvrdiť"
        print(f"{orig:<30} | {query:<25} | {num:<8} | {matched:<25} | {score:<5} | {auto_str}")
        
    print("\n" + "=" * 120)
    print(f"❌ VŠETKY CHYBNÉ/NENÁJDENÉ PÁROVANIA V CELEJ DATABÁZE (Pre 2, 3, 4, 5 preklepov):")
    print("=" * 120)
    print(f"{'Originálna ulica':<30} | {'Dopyt s preklepmi':<25} | {'Preklepy':<8} | {'Priradené (Chybne)':<25} | {'Skóre':<5} | {'Auto'}")
    print("-" * 120)
    # Zoradíme podľa počtu preklepov, potom podľa rizika (auto=True najprv)
    failures.sort(key=lambda x: (x[2], -1 if x[5] else 1))
    for orig, query, num, matched, score, auto in failures:
        auto_str = "🔴 ÁNO (Vážne!)" if auto else "🟡 Potvrdiť (Bezpečné)"
        print(f"{orig:<30} | {query:<25} | {num:<8} | {matched:<25} | {score:<5} | {auto_str}")
    print("-" * 120)

if __name__ == "__main__":
    test_typos()
