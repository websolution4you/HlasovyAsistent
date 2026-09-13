#!/usr/bin/env python3
"""
Export relevant product tables and data from Supabase to Google Cloud SQL.
Extracts only the core tables:
- tenants
- booking_users
- bookings
- wallets
- payments
- wallet_transactions
- calendar_connections (if exists)

Ignores legacy pizza delivery tables (streets, menu_items, pizza_orders, etc.).
"""

import getpass
import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

SUPABASE_HOST = "db.iejmuamnqwblmokjwzdt.supabase.co"
SUPABASE_PORT = 5432
SUPABASE_DB = "postgres"
SUPABASE_USER = "postgres"

CORE_TABLES = [
    "tenants",
    "booking_users",
    "bookings",
    "wallets",
    "payments",
    "wallet_transactions",
    "calendar_connections",
]

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "migrations", "000_core_from_supabase.sql")


def get_connection():
    db_password = os.getenv("SUPABASE_DB_PASSWORD")
    if not db_password:
        print("=" * 60)
        print("Bezpečné pripojenie k Supabase databáze")
        print(f"Host: {SUPABASE_HOST}:{SUPABASE_PORT}, DB: {SUPABASE_DB}, User: {SUPABASE_USER}")
        print("Heslo sa na obrazovke nezobrazuje a nikam sa neukladá.")
        print("=" * 60)
        db_password = getpass.getpass("Zadajte Supabase heslo pre 'postgres': ")

    if not db_password:
        print("Chyba: Heslo nebolo zadané.")
        sys.exit(1)

    print("\nPripájam sa k Supabase...")
    try:
        conn = psycopg2.connect(
            host=SUPABASE_HOST,
            port=SUPABASE_PORT,
            dbname=SUPABASE_DB,
            user=SUPABASE_USER,
            password=db_password,
            sslmode="require",
            connect_timeout=10,
        )
        print("Pripojenie k Supabase úspešné!\n")
        return conn
    except Exception as e:
        print(f"Chyba pripojenia k Supabase: {e}")
        sys.exit(1)


def get_existing_tables(cursor):
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    """)
    all_tables = [row["table_name"] for row in cursor.fetchall()]
    return [t for t in CORE_TABLES if t in all_tables]


def export_table_ddl_and_data(conn, tables, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with conn.cursor(cursor_factory=RealDictCursor) as cur, open(output_path, "w", encoding="utf-8") as f:
        f.write("-- ==========================================================\n")
        f.write("-- Core Telio schema and test data exported from Supabase\n")
        f.write("-- Target: Google Cloud SQL (PostgreSQL)\n")
        f.write("-- Tables: " + ", ".join(tables) + "\n")
        f.write("-- ==========================================================\n\n")
        f.write("BEGIN;\n\n")
        f.write("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";\n")
        f.write("CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";\n\n")

        # 1. Export Columns & Table Definitions
        for table in tables:
            print(f"Spracovávam štruktúru tabuľky: {table}...")
            cur.execute("""
                SELECT 
                    column_name, 
                    data_type, 
                    udt_name,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale,
                    is_nullable, 
                    column_default
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
            """, (table,))
            columns = cur.fetchall()

            f.write(f"-- Štruktúra tabuľky public.{table}\n")
            f.write(f"CREATE TABLE IF NOT EXISTS public.{table} (\n")
            col_defs = []
            for col in columns:
                cname = col["column_name"]
                dtype = col["udt_name"]
                if dtype == "varchar" and col["character_maximum_length"]:
                    dtype = f"varchar({col['character_maximum_length']})"
                elif dtype == "numeric" and col["numeric_precision"]:
                    dtype = f"numeric({col['numeric_precision']}, {col['numeric_scale'] or 0})"
                elif dtype == "int4":
                    dtype = "integer"
                elif dtype == "int8":
                    dtype = "bigint"
                elif dtype == "bool":
                    dtype = "boolean"
                elif dtype == "timestamptz":
                    dtype = "timestamptz"

                nullable = "" if col["is_nullable"] == "YES" else " NOT NULL"
                default = f" DEFAULT {col['column_default']}" if col["column_default"] is not None else ""
                col_defs.append(f"    {cname} {dtype}{nullable}{default}")

            # Primary key
            cur.execute("""
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = 'public'
                  AND tc.table_name = %s;
            """, (table,))
            pk_cols = [r["column_name"] for r in cur.fetchall()]
            if pk_cols:
                col_defs.append(f"    CONSTRAINT {table}_pkey PRIMARY KEY ({', '.join(pk_cols)})")

            f.write(",\n".join(col_defs))
            f.write("\n);\n\n")

        # 2. Export Foreign Keys and Constraints
        print("\nSpracovávam constraints a cudzie kľúče...")
        for table in tables:
            cur.execute("""
                SELECT conname, pg_get_constraintdef(c.oid) as def
                FROM pg_constraint c
                JOIN pg_namespace n ON n.oid = c.connamespace
                JOIN pg_class cl ON cl.oid = c.conrelid
                WHERE n.nspname = 'public' 
                  AND cl.relname = %s 
                  AND c.contype IN ('u', 'f', 'c');
            """, (table,))
            constraints = cur.fetchall()
            for c in constraints:
                cname = c["conname"]
                cdef = c["def"]
                f.write(f"DO $$ BEGIN\n")
                f.write(f"    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = '{cname}') THEN\n")
                f.write(f"        ALTER TABLE public.{table} ADD CONSTRAINT {cname} {cdef};\n")
                f.write(f"    END IF;\n")
                f.write(f"END $$;\n")
        f.write("\n")

        # 3. Export Data
        print("\nExportujem testovacie dáta...")
        for table in tables:
            cur.execute(f"SELECT COUNT(*) as cnt FROM public.{table};")
            count = cur.fetchone()["cnt"]
            print(f"Tabuľka public.{table}: {count} riadkov")
            if count == 0:
                continue

            cur.execute(f"SELECT * FROM public.{table};")
            rows = cur.fetchall()
            if rows:
                col_names = list(rows[0].keys())
                f.write(f"-- Dáta pre public.{table} ({len(rows)} riadkov)\n")
                f.write(f"INSERT INTO public.{table} ({', '.join(col_names)})\nVALUES\n")
                row_strings = []
                for row in rows:
                    val_strings = []
                    for col in col_names:
                        val = row[col]
                        if val is None:
                            val_strings.append("NULL")
                        elif isinstance(val, bool):
                            val_strings.append("TRUE" if val else "FALSE")
                        elif isinstance(val, (int, float)):
                            val_strings.append(str(val))
                        elif isinstance(val, (dict, list)):
                            import json
                            val_str = json.dumps(val).replace("'", "''")
                            val_strings.append(f"'{val_str}'::jsonb")
                        else:
                            val_str = str(val).replace("'", "''")
                            val_strings.append(f"'{val_str}'")
                    row_strings.append("    (" + ", ".join(val_strings) + ")")
                f.write(",\n".join(row_strings))
                f.write("\nON CONFLICT DO NOTHING;\n\n")

        f.write("COMMIT;\n")
    print(f"\nExport úspešne dokončený! Súbor uložený do:\n{output_path}")


def main():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            available_tables = get_existing_tables(cur)
            print("Nájdené relevantné tabuľky v Supabase:")
            for t in available_tables:
                print(f"  - public.{t}")
            print()
            export_table_ddl_and_data(conn, available_tables, OUTPUT_FILE)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
