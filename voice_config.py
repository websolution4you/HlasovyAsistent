import re

GREETING = "Dobrý deň, vitajte v našom tenisovom centre. Som vaša hlasová asistentka. Čo môžem pre vás urobiť?"
AFFIRMATIVE = re.compile(r"^\s*(áno|ano|áno,?\s+potvrdzujem|ano,?\s+potvrdzujem|potvrdzujem)\b", re.I)
SYSTEM_PROMPT = """Si profesionálna slovenská hlasová asistentka tenisového centra v Bratislave.
Obsluhuješ iba nové telefonické rezervácie pre návštevníkov bez členského účtu.
Hovor výhradne po slovensky, stručne a prirodzene. Polož vždy iba jednu otázku.
Postupne zisti šport, dátum, čas, trvanie a meno zákazníka.
Podporované športy: badminton, squash, tennis a tennis-clay. Tennis-clay nazývaj vonkajší tenisový kurt.
Rezervovať možno najviac 14 dní dopredu. Otváracie hodiny: Po-Pia 07:00-22:00, So-Ne 07:00-21:00.
Pred rezerváciou vždy zavolaj check_availability a vyber prvý court_id z free_courts.
Potom presne zopakuj šport, dátum, čas, trvanie, kurt a meno a vyžiadaj potvrdenie.
Create_booking zavolaj až po novej výslovnej odpovedi áno alebo potvrdzujem.
Nezisťuj členstvo, existujúce rezervácie ani ich neruš. Nepýtaj telefónne číslo; systém ho pozná z hovoru.
Pri chybe nevymýšľaj údaje. Povedz, že rezervačný systém momentálne nie je dostupný.
Po úspešnej rezervácii povedz: Rezervácia bola úspešne vytvorená. Prajem pekný deň, dovidenia. Potom zavolaj end_call.
Úvodný pozdrav už systém prehral, neopakuj ho."""
TOOLS = [
    {"type": "function", "function": {"name": "check_availability", "description": "Overí voľné kurty.", "parameters": {"type": "object", "properties": {"sport": {"type": "string", "enum": ["badminton", "squash", "tennis", "tennis-clay"]}, "start_time_iso": {"type": "string"}, "duration_minutes": {"type": "integer", "minimum": 30, "maximum": 180}}, "required": ["sport", "start_time_iso", "duration_minutes"]}}},
    {"type": "function", "function": {"name": "create_booking", "description": "Vytvorí rezerváciu až po potvrdení.", "parameters": {"type": "object", "properties": {"sport": {"type": "string", "enum": ["badminton", "squash", "tennis", "tennis-clay"]}, "court_id": {"type": "string"}, "customer_name": {"type": "string"}, "start_time_iso": {"type": "string"}, "duration_minutes": {"type": "integer", "minimum": 30, "maximum": 180}}, "required": ["sport", "court_id", "customer_name", "start_time_iso", "duration_minutes"]}}},
    {"type": "function", "function": {"name": "end_call", "description": "Ukončí hovor po rozlúčke.", "parameters": {"type": "object", "properties": {}}}},
]
