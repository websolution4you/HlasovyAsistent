import sys
import os
import csv
from difflib import SequenceMatcher

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

CSV_PATH = "streets_database.csv"

# Pomocný deterministický generátor preklepov (typos)
def make_typo(text: str, num_changes: int) -> str:
    chars = list(text)
    if len(chars) < num_changes + 1:
        return text
    
    vowel_map = {
        'a': 'e', 'e': 'i', 'i': 'y', 'y': 'o', 'o': 'u', 'u': 'a',
        'á': 'é', 'é': 'í', 'í': 'ý', 'ý': 'ó', 'ó': 'ú', 'ú': 'á',
        'ä': 'e', 'ô': 'o', 'í': 'i', 'ý': 'y', 'š': 's', 'č': 'c',
        'ž': 'z', 'ť': 't', 'ď': 'd', 'ň': 'n', 'ľ': 'l'
    }
    
    changed = 0
    # 1. Zmena: prehodenie dvoch susediacich písmen v strede slova (transpozícia)
    if num_changes >= 1 and len(chars) >= 4:
        mid = len(chars) // 2
        if chars[mid].isalpha() and chars[mid-1].isalpha():
            chars[mid], chars[mid-1] = chars[mid-1], chars[mid]
            changed += 1
            
    # 2. Zmena: zámena samohlások s diakritikou alebo bez
    idx = 0
    while changed < num_changes and idx < len(chars):
        c = chars[idx].lower()
        if c in vowel_map:
            is_upper = chars[idx].isupper()
            new_c = vowel_map[c]
            chars[idx] = new_c.upper() if is_upper else new_c
            changed += 1
        idx += 1
        
    # 3. Zmena: nútené nahradenie spoluhlások na konci slova
    idx = len(chars) - 1
    while changed < num_changes and idx >= 0:
        if chars[idx].isalpha() and chars[idx].lower() not in vowel_map:
            is_upper = chars[idx].isupper()
            chars[idx] = 'x'.upper() if is_upper else 'x'
            changed += 1
        idx -= 1
        
    return "".join(chars)

# Generátor čiastočných názvov pre dvojslovné a trojslovné ulice
def get_partial_names(name: str) -> list[str]:
    # Odstránime generické slová, ktoré by spôsobili príliš veľkú neurčitosť
    ignore_words = {"ulica", "cesta", "priechod", "riadok", "nad", "pod", "za", "pri"}
    words = [w.strip(" .,") for w in name.split() if w.strip(" .,")]
    filtered_words = [w for w in words if w.lower() not in ignore_words]
    
    partials = []
    if len(filtered_words) >= 2:
        # Pre dvojslovné (napr. "Viktora Greschika" -> "Greschika", "Viktora")
        # Posledné slovo (zvyčajne priezvisko) je najdôležitejšie
        partials.append(filtered_words[-1])
        # Skúsime aj prvé slovo
        partials.append(filtered_words[0])
        
    if len(filtered_words) >= 3:
        # Pre trojslovné (napr. "Námestie Štefana Kluberta" -> "Štefana Kluberta", "Kluberta")
        # Posledné dve slová
        partials.append(" ".join(filtered_words[-2:]))
        # Samotné priezvisko
        partials.append(filtered_words[-1])
        
    # Odstránime duplicity a prázdne položky
    seen = set()
    cleaned = []
    for p in partials:
        if p and p.lower() not in ignore_words and len(p) > 2 and p not in seen:
            cleaned.append(p)
            seen.add(p)
    return cleaned

def load_streets() -> list[str]:
    streets = []
    if not os.path.exists(CSV_PATH):
        print(f"Chyba: Súbor {CSV_PATH} neexistuje!")
        sys.exit(1)
        
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            streets.append(row['name'])
    return streets

def run_accuracy_test():
    streets = load_streets()
    print("=" * 100)
    print(f"NAČÍTANÝCH {len(streets)} ULÍC Z DATABÁZY. SPUŠŤAM HĹBKOVÝ TEST PRESNOSTI...")
    print("=" * 100)
    
    stats = {
        "exact": {"total": 0, "correct_auto": 0, "correct_conf": 0, "wrong": 0, "not_found": 0},
        "typo1": {"total": 0, "correct_auto": 0, "correct_conf": 0, "wrong": 0, "not_found": 0},
        "typo2": {"total": 0, "correct_auto": 0, "correct_conf": 0, "wrong": 0, "not_found": 0},
        "typo3": {"total": 0, "correct_auto": 0, "correct_conf": 0, "wrong": 0, "not_found": 0},
        "typo4": {"total": 0, "correct_auto": 0, "correct_conf": 0, "wrong": 0, "not_found": 0},
        "typo5": {"total": 0, "correct_auto": 0, "correct_conf": 0, "wrong": 0, "not_found": 0},
        "partial": {"total": 0, "correct_auto": 0, "correct_conf": 0, "wrong": 0, "not_found": 0}
    }
    
    wrong_matches = [] # Pre ukladanie false-positives
    
    def evaluate_test(query: str, original_street: str, category: str):
        resolution = _street_resolution(query, streets)
        best = resolution["best"]
        auto_accept = resolution["auto_accept"]
        
        stats[category]["total"] += 1
        
        if not best:
            stats[category]["not_found"] += 1
            return "NOT_FOUND", "Žiadna", 0, False
            
        matched_street = best["street"]
        score = best["score"]
        
        # Keďže názov môže obsahovať drobné odchýlky, porovnáme normalizované hodnoty
        if _normalize(matched_street) == _normalize(original_street):
            if auto_accept:
                stats[category]["correct_auto"] += 1
                return "CORRECT_AUTO", matched_street, score, auto_accept
            else:
                stats[category]["correct_conf"] += 1
                return "CORRECT_CONF", matched_street, score, auto_accept
        else:
            stats[category]["wrong"] += 1
            # Zaznamenáme nebezpečné nesprávne priradenie
            wrong_matches.append({
                "query": query,
                "expected": original_street,
                "matched": matched_street,
                "score": score,
                "auto_accept": auto_accept,
                "category": category
            })
            return "WRONG", matched_street, score, auto_accept

    print(f"{'Originálna ulica':<30} | {'Testovací dopyt':<25} | {'Kategória':<8} | {'Výsledok':<12} | {'Priradené':<30} | {'Skóre':<5} | {'Auto'}")
    print("-" * 120)
    
    for street in streets:
        # 1. PRESNÝ NÁZOV
        evaluate_test(street, street, "exact")
        
        # 2. JEDEN PREKLEP (1 Typo)
        typo1 = make_typo(street, 1)
        if typo1 != street:
            evaluate_test(typo1, street, "typo1")
            
        # 3. DVA PREKLEPY (2 Typos)
        typo2 = make_typo(street, 2)
        if typo2 != street:
            evaluate_test(typo2, street, "typo2")
            
        # 4. TRI PREKLEPY (3 Typos)
        typo3 = make_typo(street, 3)
        if typo3 != street:
            evaluate_test(typo3, street, "typo3")
            
        # 4b. ŠTYRI PREKLEPY (4 Typos)
        typo4 = make_typo(street, 4)
        if typo4 != street:
            evaluate_test(typo4, street, "typo4")
            
        # 4c. PÄŤ PREKLEPOV (5 Typos)
        typo5 = make_typo(street, 5)
        if typo5 != street:
            evaluate_test(typo5, street, "typo5")
            
        # 5. ČIASTOČNÝ NÁZOV (Skratka pre dvojslovné/trojslovné)
        partials = get_partial_names(street)
        for part in partials:
            evaluate_test(part, street, "partial")
            
    # Zobrazenie celkových štatistík
    print("=" * 120)
    print("SÚHRNNÝ PREHĽAD VÝSLEDKOV TESTU:")
    print("=" * 120)
    print(f"{'Kategória testu':<15} | {'Celkovo':<8} | {'Presne (Auto)':<13} | {'S potvrdením':<12} | {'Nesprávne (!!!)':<15} | {'Nenájdené':<10} | {'Úspešnosť %':<10}")
    print("-" * 120)
    
    for cat, data in stats.items():
        total = data["total"]
        if total == 0:
            continue
        correct = data["correct_auto"] + data["correct_conf"]
        success_rate = (correct / total) * 100
        
        print(f"{cat:<15} | {total:<8} | {data['correct_auto']:<13} | {data['correct_conf']:<12} | {data['wrong']:<15} | {data['not_found']:<10} | {success_rate:.1f}%")
        
    print("=" * 120)
    print(f"Celkovo chýb (nebezpečné false-positives): {len(wrong_matches)}")
    print("=" * 120)
    
    if wrong_matches:
        print("\nZOZNAM NEBEZPEČNÝCH ZLYHANÍ (Keď asistent priradil ÚPLNE INÚ ulicu):")
        print("-" * 120)
        print(f"{'Dopyt od zákazníka':<25} | {'Očakávané (Správne)':<30} | {'Priradené (Chybne)':<30} | {'Skóre':<5} | {'Auto-Akcept'}")
        print("-" * 120)
        # Zoradíme chybné priradenia podľa skóre (najvyššie skóre sú najnebezpečnejšie)
        wrong_matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Zobrazíme top 25 najvážnejších chýb pre prehľadnosť
        for item in wrong_matches[:25]:
            auto_str = "ÁNO (Vážna chyba!)" if item["auto_accept"] else "NIE (Bezpečné - potvrdí sa)"
            print(f"{item['query']:<25} | {item['expected']:<30} | {item['matched']:<30} | {item['score']:<5} | {auto_str}")
            
        if len(wrong_matches) > 25:
            print(f"... a ďalších {len(wrong_matches) - 25} chýb.")
        print("-" * 120)
        
if __name__ == "__main__":
    run_accuracy_test()
