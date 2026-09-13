"""
Database connection pool and query helpers for PostgreSQL / Google Cloud SQL.
Supports Cloud Run Unix socket mounts and TCP connections with SSL.
"""

import os
import ssl
from typing import Any, List, Optional, Sequence
import asyncpg

_pool: Optional[asyncpg.Pool] = None

INSTANCE_CONNECTION_NAME = os.getenv(
    "INSTANCE_CONNECTION_NAME",
    "project-800f9b01-ef95-4b9e-853:europe-west3:telio-postgres",
).strip()


async def init_db_pool() -> Optional[asyncpg.Pool]:
    global _pool
    if _pool is not None:
        return _pool

    database_url = os.getenv("DATABASE_URL", "").strip()
    db_user = os.getenv("DB_USER", "telio_app").strip()
    db_password = os.getenv("DB_PASSWORD", "").strip()
    db_name = os.getenv("DB_NAME", "telio").strip()
    db_host = os.getenv("DB_HOST", "").strip()
    db_port = int(os.getenv("DB_PORT", "5432"))

    # Cloud Run auto-mounts unix domain socket at /cloudsql/<INSTANCE_CONNECTION_NAME>
    cloud_sql_socket_dir = f"/cloudsql/{INSTANCE_CONNECTION_NAME}"

    try:
        if database_url:
            print(f"[db] Initializing pool from DATABASE_URL...")
            _pool = await asyncpg.create_pool(
                dsn=database_url,
                min_size=1,
                max_size=10,
                command_timeout=30,
            )
        elif os.path.exists(cloud_sql_socket_dir):
            print(f"[db] Connecting via Cloud SQL Unix socket: {cloud_sql_socket_dir}")
            _pool = await asyncpg.create_pool(
                host=cloud_sql_socket_dir,
                user=db_user,
                password=db_password,
                database=db_name,
                min_size=1,
                max_size=10,
                command_timeout=30,
            )
        elif db_host:
            print(f"[db] Connecting via TCP to {db_host}:{db_port} (db={db_name}, user={db_user})...")
            # Google Cloud SQL requires SSL on public IP
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            _pool = await asyncpg.create_pool(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                database=db_name,
                ssl=ssl_ctx,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        else:
            print("[db] Warning: No database connection parameters configured (DATABASE_URL, DB_HOST, or Cloud SQL socket).")
            return None

        print("[db] PostgreSQL connection pool initialized successfully.")
        return _pool
    except Exception as exc:
        print(f"[db] Failed to initialize PostgreSQL connection pool: {exc}")
        _pool = None
        return None


async def close_db_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        print("[db] PostgreSQL connection pool closed.")


def get_pool() -> Optional[asyncpg.Pool]:
    return _pool


async def db_fetch(query: str, *args: Any) -> List[asyncpg.Record]:
    pool = get_pool()
    if not pool:
        raise RuntimeError("Database connection pool is not initialized.")
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def db_fetchrow(query: str, *args: Any) -> Optional[asyncpg.Record]:
    pool = get_pool()
    if not pool:
        raise RuntimeError("Database connection pool is not initialized.")
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def db_fetchval(query: str, *args: Any) -> Any:
    pool = get_pool()
    if not pool:
        raise RuntimeError("Database connection pool is not initialized.")
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)


async def db_execute(query: str, *args: Any) -> str:
    pool = get_pool()
    if not pool:
        raise RuntimeError("Database connection pool is not initialized.")
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)
