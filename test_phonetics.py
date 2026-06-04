import sys
import os
import csv

# Ensure UTF-8 output for Windows Terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import main
from main import _street_resolution

def load_streets():
    streets = []
    with open("streets_database.csv", mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            streets.append(row['name'])
    return streets

def run_tests():
    streets = load_streets()
    test_queries = [
        "čaučika",
        "Jozefa čaučika",
        "grešika",
        "herman",
        "hermana",
        "botova"
    ]
    
    print("=" * 80)
    print("RUNNING PHONETIC VERIFICATION TESTS")
    print("=" * 80)
    print(f"{'Query':<20} | {'Best Match':<30} | {'Score':<5} | {'Auto-Accept'}")
    print("-" * 80)
    
    for q in test_queries:
        res = _street_resolution(q, streets)
        best = res["best"]
        if best:
            print(f"{q:<20} | {best['street']:<30} | {best['score']:<5} | {res['auto_accept']}")
        else:
            print(f"{q:<20} | {'None':<30} | {'N/A':<5} | {res['auto_accept']}")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
