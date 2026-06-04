import unicodedata
from rapidfuzz import fuzz

def _normalize(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def clean_street_text(text: str) -> str:
    normalized = _normalize(text)
    words = normalized.split()
    generics = {
        "ulica", "cesta", "priechod", "riadok", "chodnik", 
        "namestie", "sidlisko", "alej", "farma", "tunel", 
        "odpocivadlo", "za", "pod", "nad", "pri", "na", 
        "v", "s", "z", "do"
    }
    filtered = [w for w in words if w not in generics]
    if not filtered:
        filtered = words
    return " ".join(filtered)

def new_street_score(query: str, street: str) -> tuple[int, int]:
    q = _normalize(query)
    s = _normalize(street)
    
    if q == s:
        return 100, 100
        
    qc = clean_street_text(query)
    sc = clean_street_text(street)
    
    if qc == sc:
        return 100, 100
        
    r = fuzz.ratio(q, s)
    tsr = fuzz.token_sort_ratio(q, s)
    tset = fuzz.token_set_ratio(q, s)
    
    r_clean = fuzz.ratio(qc, sc)
    tsr_clean = fuzz.token_sort_ratio(qc, sc)
    tset_clean = fuzz.token_set_ratio(qc, sc)
    
    q_tokens = qc.split()
    s_tokens = sc.split()
    
    if not q_tokens or not s_tokens:
        return round(max(r, tsr)), r
        
    q_matches = []
    for qt in q_tokens:
        best_val = 0
        for st in s_tokens:
            val = fuzz.ratio(qt, st)
            if val > best_val:
                best_val = val
        q_matches.append(best_val)
        
    s_matches = []
    for st in s_tokens:
        best_val = 0
        for qt in q_tokens:
            val = fuzz.ratio(st, qt)
            if val > best_val:
                best_val = val
        s_matches.append(best_val)
        
    q_cov = sum(q_matches) / len(q_matches)
    s_cov = sum(s_matches) / len(s_matches)
    
    # Coverage score: how well the query matches the street
    # We penalize slightly for length difference to favor closer lengths
    len_diff_penalty = 0.04 * max(0, len(s_tokens) - len(q_tokens))
    partial_score = max(0.0, q_cov - len_diff_penalty * 100)
    
    primary = round(max(r, tsr, r_clean, tsr_clean, partial_score))
    
    # If it's a very high quality token set match, boost it
    if tset_clean >= 90 and q_cov >= 85:
        primary = max(primary, round(q_cov))
        
    return primary, r

# Test cases
cases = [
    # (query, street_name, expected_to_be_high_or_low)
    ("kakuicnova", "nova", "low"),
    ("kakuicnova", "kukucinova", "high"),
    ("kakaicnova", "nova", "low"),
    ("kakaicnova", "kukucinova", "high"),
    ("ze sderiou", "za sedriou", "high"),
    ("ze sderiou", "jozefa czauczika", "low"),
    ("levocske", "levocske luky", "high"),
    ("levocske", "levocska", "low"),
    ("levocske", "levocske kupele", "high"),
    ("levoca", "odpocivadlo levoca", "high"),
    ("levoca", "levocska", "low"),
    ("czauczika", "ulica czauczika", "high"),
    ("czauczika", "jozefa czauczika", "high"),
    ("namestie", "namestie stefana kluberta", "high/medium"),
    ("namestie", "namestie majstra pavla", "high/medium"),
]

for q, s, exp in cases:
    score, r = new_street_score(q, s)
    print(f"Query: '{q}' | Street: '{s}' | Expected: {exp:<11} | Score: {score} | Ratio: {r}")
