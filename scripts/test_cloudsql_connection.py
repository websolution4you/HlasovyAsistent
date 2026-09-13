#!/usr/bin/env python3
"""
Test connection to Google Cloud SQL and verify migrated tables and data.
"""

import asyncio
import getpass
import os
import ssl
import sys
import asyncpg

CLOUDSQL_HOST = os.getenv("DB_HOST", "34.159.129.208")
CLOUDSQL_PORT = int(os.getenv("DB_PORT", "5432"))
CLOUDSQL_DB = os.getenv("DB_NAME", "telio")
CLOUDSQL_USER = os.getenv("DB_USER", "telio_app")


async def test_connection():
    db_password = os.getenv("DB_PASSWORD")
    if not db_password:
        print("=" * 60)
        print("Overenie spojenia na Google Cloud SQL")
        print(f"Host: {CLOUDSQL_HOST}:{CLOUDSQL_PORT}, DB: {CLOUDSQL_DB}, User: {CLOUDSQL_USER}")
        print("=" * 60)
        db_password = getpass.getpass(f"Zadajte heslo pre používateľa '{CLOUDSQL_USER}': ")

    if not db_password:
        print("Chyba: Heslo nebolo zadané.")
        sys.exit(1)

    print(f"\nPripájam sa k Google Cloud SQL ({CLOUDSQL_HOST})...")
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    try:
        conn = await asyncpg.connect(
            host=CLOUDSQL_HOST,
            port=CLOUDSQL_PORT,
            user=CLOUDSQL_USER,
            password=db_password,
            database=CLOUDSQL_DB,
            ssl=ssl_ctx,
            timeout=10,
        )
        print("Spojenie úspešne nadviazané!\n")

        # Kontrola tabuliek a poctu riadkov
        tables = ["tenants", "booking_users", "bookings", "calendar_connections", "wallets", "payments", "wallet_transactions"]
        print("Stav tabuliek v databáze 'telio':")
        for table in tables:
            try:
                count = await conn.fetchval(f"SELECT count(*) FROM public.{table};")
                print(f"  ✓ public.{table}: {count} záznamov")
            except Exception as e:
                print(f"  ✗ public.{table}: CHYBA ({e})")

        # Kontrola wallet funkcií
        print("\nKontrola funkcií peňaženky:")
        try:
            report = await conn.fetch("SELECT * FROM public.wallet_integrity_report();")
            print(f"  ✓ wallet_integrity_report: OK (vrátených {len(report)} riadkov)")
        except Exception as e:
            print(f"  ✗ wallet_integrity_report: CHYBA ({e})")

        await conn.close()
        print("\nTest úspešne dokončený! Databáza je pripravená.")
    except Exception as exc:
        print(f"\nChyba pripojenia: {exc}")
        print("\nTIP: Ak pripojenie vypršalo (timeout), skontrolujte v Google Cloud Console:")
        print("1. Inštancia 'telio-postgres' -> Connections -> Networking")
        print("2. V sekcii 'Authorized networks' pridajte verejnú IP vášho počítača (alebo 0.0.0.0/0 pre dočasný test).")
        print("3. Uistite sa, že používateľ 'telio_app' má zadané správne heslo.")


if __name__ == "__main__":
    asyncio.run(test_connection())
