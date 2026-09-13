#!/usr/bin/env python3
"""
Export core tables and test data from Supabase using CORE_SUPABASE_SERVICE_ROLE_KEY.
Does not require PostgreSQL direct password.
"""

import getpass
import json
import os
import sys
from supabase import create_client, Client

SUPABASE_URL = "https://iejmuamnqwblmokjwzdt.supabase.co"

CORE_TABLES = [
    "tenants",
    "booking_users",
    "bookings",
    "wallets",
    "payments",
    "wallet_transactions",
    "calendar_connections",
]

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "migrations",
    "000_core_from_supabase.sql",
)


def get_client() -> Client:
    key = os.getenv("CORE_SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        print("=" * 60)
        print("Export zo Supabase cez Service Role Key")
        print(f"URL: {SUPABASE_URL}")
        print("Kľúč sa na obrazovke nezobrazuje a nikam sa neukladá.")
        print("=" * 60)
        key = getpass.getpass("Vložte CORE_SUPABASE_SERVICE_ROLE_KEY: ").strip()

    if not key:
        print("Chyba: Kľúč nebol zadaný.")
        sys.exit(1)

    print("\nPripájam sa k Supabase cez API...")
    try:
        client = create_client(SUPABASE_URL, key)
        print("Pripojenie úspešné!\n")
        return client
    except Exception as e:
        print(f"Chyba inicializácie klienta: {e}")
        sys.exit(1)


def fetch_table_data(client: Client, table: str):
    try:
        res = client.table(table).select("*").execute()
        return res.data or []
    except Exception as e:
        print(f"  Tabuľka '{table}' neexistuje alebo k nej nie je prístup: {e}")
        return None


def infer_pg_type(col_name: str, values: list) -> str:
    non_nulls = [v for v in values if v is not None]
    if not non_nulls:
        if col_name.endswith("_id") or col_name == "id":
            return "uuid"
        if col_name.endswith("_at"):
            return "timestamptz"
        return "text"

    val = non_nulls[0]
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, int):
        return "bigint" if any(v > 2147483647 for v in non_nulls if isinstance(v, int)) else "integer"
    if isinstance(val, float):
        return "numeric(12, 2)"
    if isinstance(val, (dict, list)):
        return "jsonb"
    if isinstance(val, str):
        if col_name.endswith("_at") or "date" in col_name:
            return "timestamptz"
        # check uuid length and dashes
        if (col_name.endswith("_id") or col_name == "id") and len(val) == 36 and val.count("-") == 4:
            return "uuid"
        return "text"
    return "text"


def generate_sql(data_by_table: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("-- ==========================================================\n")
        f.write("-- Core Telio schema and test data exported from Supabase\n")
        f.write("-- Target: Google Cloud SQL (PostgreSQL)\n")
        f.write("-- ==========================================================\n\n")
        f.write("BEGIN;\n\n")
        f.write("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";\n")
        f.write("CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";\n\n")

        # 1. Base DDL definitions
        f.write("-- 1. TENANTS\n")
        f.write("""CREATE TABLE IF NOT EXISTS public.tenants (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slug text NOT NULL UNIQUE,
    name text NOT NULL,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

""")

        f.write("-- 2. BOOKING USERS\n")
        f.write("""CREATE TABLE IF NOT EXISTS public.booking_users (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL,
    phone text NOT NULL,
    email text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_booking_users_phone ON public.booking_users(phone);

""")

        f.write("-- 3. BOOKINGS\n")
        f.write("""CREATE TABLE IF NOT EXISTS public.bookings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL REFERENCES public.tenants(id) ON DELETE RESTRICT,
    user_id uuid REFERENCES public.booking_users(id) ON DELETE SET NULL,
    court_id text NOT NULL,
    sport text NOT NULL,
    customer_name text NOT NULL,
    customer_phone text NOT NULL,
    start_at timestamptz NOT NULL,
    end_at timestamptz NOT NULL,
    status text NOT NULL DEFAULT 'confirmed',
    notes text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bookings_tenant_time ON public.bookings(tenant_id, start_at, end_at);
CREATE INDEX IF NOT EXISTS idx_bookings_court_time ON public.bookings(court_id, start_at, end_at);

""")

        f.write("-- 4. CALENDAR CONNECTIONS (if used)\n")
        f.write("""CREATE TABLE IF NOT EXISTS public.calendar_connections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid REFERENCES public.tenants(id) ON DELETE CASCADE,
    provider text NOT NULL,
    credentials jsonb NOT NULL DEFAULT '{}'::jsonb,
    calendar_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

""")

        # Tables for wallets and payments will be created by 001_wallet_ledger.sql or inline here
        # But let's check if we have other tables with data
        for table, rows in data_by_table.items():
            if table in ["tenants", "booking_users", "bookings", "calendar_connections", "wallets", "payments", "wallet_transactions"]:
                continue
            if rows:
                f.write(f"-- Custom table: public.{table}\n")
                f.write(f"CREATE TABLE IF NOT EXISTS public.{table} (\n")
                sample = rows[0]
                col_defs = []
                for col in sample.keys():
                    all_vals = [r.get(col) for r in rows]
                    ptype = infer_pg_type(col, all_vals)
                    pk = " PRIMARY KEY" if col == "id" else ""
                    col_defs.append(f"    {col} {ptype}{pk}")
                f.write(",\n".join(col_defs))
                f.write("\n);\n\n")

        # 2. Insert Data
        f.write("-- ==========================================================\n")
        f.write("-- DATA INSERTS\n")
        f.write("-- ==========================================================\n\n")

        # Insert order: tenants -> booking_users -> bookings -> etc.
        ordered_tables = ["tenants", "booking_users", "bookings", "calendar_connections", "wallets", "payments", "wallet_transactions"]
        for extra in data_by_table.keys():
            if extra not in ordered_tables:
                ordered_tables.append(extra)

        for table in ordered_tables:
            rows = data_by_table.get(table)
            if not rows:
                continue

            f.write(f"-- Data for public.{table} ({len(rows)} rows)\n")
            col_names = list(rows[0].keys())
            f.write(f"INSERT INTO public.{table} ({', '.join(col_names)})\nVALUES\n")
            row_strings = []
            for row in rows:
                val_strings = []
                for col in col_names:
                    val = row.get(col)
                    if val is None:
                        val_strings.append("NULL")
                    elif isinstance(val, bool):
                        val_strings.append("TRUE" if val else "FALSE")
                    elif isinstance(val, (int, float)):
                        val_strings.append(str(val))
                    elif isinstance(val, (dict, list)):
                        val_str = json.dumps(val).replace("'", "''")
                        val_strings.append(f"'{val_str}'::jsonb")
                    else:
                        val_str = str(val).replace("'", "''")
                        val_strings.append(f"'{val_str}'")
                row_strings.append("    (" + ", ".join(val_strings) + ")")
            f.write(",\n".join(row_strings))
            f.write("\nON CONFLICT DO NOTHING;\n\n")

        f.write("COMMIT;\n")

    print(f"\nGenerovanie úspešné! SQL súbor uložený do:\n{output_path}")


def main():
    client = get_client()
    data_by_table = {}

    print("Sťahujem dáta z tabuliek:")
    for table in CORE_TABLES:
        print(f"  -> Čítam 'public.{table}'...")
        rows = fetch_table_data(client, table)
        if rows is not None:
            print(f"     Nájdených {len(rows)} záznamov.")
            if rows:
                data_by_table[table] = rows
        else:
            print(f"     (tabuľka preskočená)")

    print("\nGenerujem výsledný SQL súbor pre Google Cloud SQL...")
    generate_sql(data_by_table, OUTPUT_FILE)


if __name__ == "__main__":
    main()
