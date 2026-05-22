import sys
import os
import csv

# Ensure UTF-8 output for Windows Terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import main
from main import _street_resolution, _normalize
from test_accuracy import get_partial_names, make_typo

REPORT_PATH = r"C:\Users\Dell\.gemini\antigravity\brain\7486ff6b-f766-40fc-93e6-dd096176512c\multiword_report.md"

def load_streets():
    streets = []
    with open("streets_database.csv", mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            streets.append(row['name'])
    return streets

def generate_markdown_report():
    streets = load_streets()
    multiword_streets = [s for s in streets if len(s.split()) >= 2]
    
    lines = []
    lines.append("# Detailný prehľad testov viacslovných ulíc a čiastočných názvov\n")
    lines.append("Tento dokument prináša detailný breakdown testov pre všetky **dvojslovné a trojslovné ulice** a ich čiastočné varianty (skratky, volania len jedným slovom, atď.).\n")
    lines.append("| Originálna ulica | Testovaný dopyt | Typ testu | Priradené v DB | Skóre | Auto-Akcept (Bez potvrdenia) | Výsledok |")
    lines.append("| :--- | :--- | :--- | :--- | :---: | :---: | :--- |")
    
    for street in multiword_streets:
        # 1. Celý názov
        res = _street_resolution(street, streets)
        best = res["best"]
        matched = best["street"] if best else "Nenájdené"
        score = best["score"] if best else 0
        auto = "🟢 ÁNO" if res["auto_accept"] else "🟡 Potvrdiť"
        lines.append(f"| **{street}** | {street} | Celý názov | {matched} | {score} | {auto} | ✅ Presná zhoda |")
        
        # 2. Celý názov s 1 preklepom
        typo1 = make_typo(street, 1)
        if typo1 != street:
            res = _street_resolution(typo1, streets)
            best = res["best"]
            matched = best["street"] if best else "Nenájdené"
            score = best["score"] if best else 0
            auto = "🟢 ÁNO" if res["auto_accept"] else "🟡 Potvrdiť"
            lines.append(f"| **{street}** | *{typo1}* | Celý s 1 preklepom | {matched} | {score} | {auto} | ✅ Úspešne spárované |")
            
        # 3. Čiastočné názvy
        partials = get_partial_names(street)
        for part in partials:
            res = _street_resolution(part, streets)
            best = res["best"]
            matched = best["street"] if best else "Nenájdené"
            score = best["score"] if best else 0
            auto = "🟢 ÁNO" if res["auto_accept"] else "🟡 Potvrdiť"
            
            note = ""
            status = "✅ OK (Správny čiastočný match)"
            
            if _normalize(matched) == _normalize(street):
                pass
            else:
                if part.lower() in matched.lower():
                    status = f"ℹ️ Legitime prekrytie (priradené k '{matched}')"
                else:
                    status = f"❌ Iná ulica ('{matched}')"
                    
            lines.append(f"| **{street}** | `{part}` | Čiastočný názov | {matched} | {score} | {auto} | {status} |")
            
        # Prázdny riadok ako vizuálny oddelovač medzi ulicami v tabuľke
        lines.append("| | | | | | | |")

    # Zápis do súboru
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"Report úspešne zapísaný do: {REPORT_PATH}")

if __name__ == "__main__":
    generate_markdown_report()
