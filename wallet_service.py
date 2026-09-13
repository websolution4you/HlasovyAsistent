import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any


class WalletOperationError(RuntimeError):
    """Base error for atomic wallet operations."""


class WalletRequiredError(WalletOperationError):
    """The caller is not an identified user with a wallet."""


class InsufficientBalanceError(WalletOperationError):
    """The wallet cannot cover the booking price."""


class BookingConflictError(WalletOperationError):
    """The requested court became unavailable."""


def wallet_enabled_for_user(mode: str, user_id: str, allowed_user_ids: set[str]) -> bool:
    """Return whether wallet charging applies to this user in the configured mode."""
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode == "on":
        return True
    if normalized_mode == "test":
        normalized_user_id = str(user_id or "").strip().lower()
        return bool(normalized_user_id and normalized_user_id in allowed_user_ids)
    return False


@dataclass(frozen=True)
class WalletBookingResult:
    booking_id: str
    charged_eur: Decimal
    balance_eur: Decimal
    created: bool


def booking_idempotency_key(
    operation_id: str,
    user_id: str,
    court_id: str,
    sport: str,
    start_at: str,
    end_at: str,
) -> str:
    """Return a stable, non-sensitive key for retries of one booking operation."""
    canonical = json.dumps(
        {
            "operation_id": str(operation_id or "").strip(),
            "user_id": str(user_id or "").strip(),
            "court_id": str(court_id or "").strip().lower(),
            "sport": str(sport or "").strip().lower(),
            "start_at": start_at,
            "end_at": end_at,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_ntc_booking_with_wallet(
    supabase: Any,
    *,
    user_id: str,
    court_id: str,
    sport: str,
    customer_name: str,
    customer_phone: str,
    start_at: str,
    end_at: str,
    notes: str,
    idempotency_key: str,
) -> WalletBookingResult:
    if not user_id:
        raise WalletRequiredError("Používateľa sa nepodarilo identifikovať.")
    if not supabase:
        raise WalletOperationError("Databázové pripojenie nie je dostupné.")

    params = {
        "p_user_id": user_id,
        "p_court_id": court_id,
        "p_sport": sport,
        "p_customer_name": customer_name,
        "p_customer_phone": customer_phone,
        "p_start_at": start_at,
        "p_end_at": end_at,
        "p_notes": notes,
        "p_idempotency_key": idempotency_key,
    }

    try:
        response = supabase.rpc("wallet_create_ntc_booking", params).execute()
    except Exception as exc:
        message = str(exc).lower()
        if "insufficient wallet balance" in message:
            raise InsufficientBalanceError("Nedostatočný zostatok v peňaženke.") from exc
        if "wallet does not exist" in message:
            raise WalletRequiredError("Používateľ nemá vytvorenú peňaženku.") from exc
        if "selected court is no longer available" in message:
            raise BookingConflictError("Vybraný kurt už nie je voľný.") from exc
        raise WalletOperationError("Rezerváciu s platbou sa nepodarilo vytvoriť.") from exc

    rows = response.data or []
    if not rows:
        raise WalletOperationError("Databáza nevrátila výsledok rezervácie.")

    row = rows[0]
    return WalletBookingResult(
        booking_id=str(row["booking_id"]),
        charged_eur=Decimal(str(row["charged_eur"])),
        balance_eur=Decimal(str(row["balance_eur"])),
        created=bool(row["created"]),
    )


async def create_ntc_booking_with_wallet_db(
    *,
    user_id: str,
    court_id: str,
    sport: str,
    customer_name: str,
    customer_phone: str,
    start_at: str,
    end_at: str,
    notes: str,
    idempotency_key: str,
) -> WalletBookingResult:
    """Execute wallet booking directly against Google Cloud SQL / PostgreSQL via asyncpg."""
    from database import db_fetchrow, get_pool
    if not user_id:
        raise WalletRequiredError("Používateľa sa nepodarilo identifikovať.")
    if not get_pool():
        raise WalletOperationError("Databázové pripojenie nie je dostupné.")

    query = """
        SELECT booking_id, charged_eur, balance_eur, created
          FROM public.wallet_create_ntc_booking(
              $1::uuid,
              $2::text,
              $3::text,
              $4::text,
              $5::text,
              $6::timestamptz,
              $7::timestamptz,
              $8::text,
              $9::text
          );
    """
    try:
        row = await db_fetchrow(
            query,
            user_id,
            court_id,
            sport,
            customer_name,
            customer_phone,
            start_at,
            end_at,
            notes,
            idempotency_key,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "insufficient wallet balance" in message:
            raise InsufficientBalanceError("Nedostatočný zostatok v peňaženke.") from exc
        if "wallet does not exist" in message:
            raise WalletRequiredError("Používateľ nemá vytvorenú peňaženku.") from exc
        if "selected court is no longer available" in message:
            raise BookingConflictError("Vybraný kurt už nie je voľný.") from exc
        raise WalletOperationError(f"Rezerváciu s platbou sa nepodarilo vytvoriť: {exc}") from exc

    if not row:
        raise WalletOperationError("Databáza nevrátila výsledok rezervácie.")

    return WalletBookingResult(
        booking_id=str(row["booking_id"]),
        charged_eur=Decimal(str(row["charged_eur"])),
        balance_eur=Decimal(str(row["balance_eur"])),
        created=bool(row["created"]),
    )

