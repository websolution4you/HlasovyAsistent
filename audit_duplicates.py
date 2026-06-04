import csv
import sys
from rapidfuzz import fuzz

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import main
from main import _normalize, clean_street_text

def load_streets():
    streets = []
    with open("streets_database.csv", mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            streets.append(row['name'])
    return streets

def audit():
    streets = load_streets()
    print("Spúšťam audit databázy ulíc na duplicity a blízke zhody...")
    
    cleaned_map = {}
    for street in streets:
        norm = _normalize(street)
        clean = clean_street_text(street)
        
        # Uložíme si informácie pre porovnanie
        cleaned_map[street] = {
            "norm": norm,
            "clean": clean,
            "words": set(clean.split())
        }
    
    # 1. IDENTICKÉ PO OČISTENÍ (Generické slová ako "Ulica", "cesta" atď. urobili rozdiel)
    exact_duplicates = []
    seen = {}
    for street, data in cleaned_map.items():
        clean = data["clean"]
        if clean in seen:
            exact_duplicates.append((seen[clean], street, clean))
        else:
            seen[clean] = street
            
    print("\n" + "="*80)
    print("1. DUPLICITY: Úplne identické ulice po odstránení generických slov (ulica/cesta/...)")
    print("="*80)
    if exact_duplicates:
        for u1, u2, clean in exact_duplicates:
            print(f"🔴 '{u1}'  VS  '{u2}'  (obe sa očistia na: '{clean}')")
    else:
        print("Žiadne identické duplicity po očistení.")
        
    # 2. VEĽMI PODOBNÉ S NÁZVOM (napr. Kežmarská cesta vs Kežmarská ulica)
    # Tieto majú zhodnú hlavnú časť, ale líšia sa v type cesty (cesta/ulica)
    same_base_different_type = []
    for i, s1 in enumerate(streets):
        for s2 in streets[i+1:]:
            c1 = cleaned_map[s1]["clean"]
            c2 = cleaned_map[s2]["clean"]
            
            # Ak sa očistené názvy zhodujú (alebo sú extrémne blízko, napr. Kežmarská ulica/cesta)
            if c1 == c2 and s1 != s2:
                # Už zachytené v identických duplicitách, preskočíme
                continue
                
            # Kontrola či ide o kombináciu cesta/ulica s rovnakým základom
            # napr. "Kežmarská ulica" a "Kežmarská cesta"
            w1 = set(s1.lower().split())
            w2 = set(s2.lower().split())
            
            # Spoločné slová bez typov ciest
            base1 = w1 - {"ulica", "cesta", "priechod", "riadok", "namestie", "sidlisko"}
            base2 = w2 - {"ulica", "cesta", "priechod", "riadok", "namestie", "sidlisko"}
            
            if base1 == base2 and base1:
                same_base_different_type.append((s1, s2, " ".join(base1)))
                
    print("\n" + "="*80)
    print("2. BLÍZKE ZHODY: Rovnaký základ, ale iný typ cesty (napr. ulica vs cesta)")
    print("="*80)
    if same_base_different_type:
        for u1, u2, base in same_base_different_type:
            print(f"🟡 '{u1}'  VS  '{u2}'  (spoločný základ: '{base}')")
    else:
        print("Žiadne blízke zhody rovnakého základu s iným typom cesty.")

    # 3. PREKRYTIE JEDNOSLOVNÝCH A VIACSLOVNÝCH NÁZVOV (napr. Levočská vs Levočská Dolina)
    overlaps = []
    for s1 in streets:
        for s2 in streets:
            if s1 == s2:
                continue
            c1 = cleaned_map[s1]["clean"]
            c2 = cleaned_map[s2]["clean"]
            
            # Ak je jedno slovo obsiahnuté v druhom viacslovnom a je to celé slovo
            words1 = cleaned_map[s1]["words"]
            words2 = cleaned_map[s2]["words"]
            
            if len(words1) == 1 and words1.issubset(words2):
                overlaps.append((s1, s2))
                
    print("\n" + "="*80)
    print("3. PREKRYTIA: Jednoslovný názov je súčasťou viacslovného (spôsobuje nejednoznačnosť pri skratkách)")
    print("="*80)
    if overlaps:
        for short, long in overlaps:
            print(f"🔵 '{short}' je obsiahnuté v '{long}'")
    else:
        print("Žiadne prekrytia.")

if __name__ == "__main__":
    audit()
