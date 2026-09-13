-- Atomic NTC booking charges and refunds in EUR.
-- Temporary pricing rule: every booking costs EUR 10.00.
-- Replace the constant with a server-side pricing table in a later migration.

begin;

create or replace function public.wallet_create_ntc_booking(
    p_user_id uuid,
    p_court_id text,
    p_sport text,
    p_customer_name text,
    p_customer_phone text,
    p_start_at timestamptz,
    p_end_at timestamptz,
    p_notes text,
    p_idempotency_key text
)
returns table (
    booking_id uuid,
    charged_eur numeric(12, 2),
    balance_eur numeric(12, 2),
    created boolean
)
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_tenant_id uuid;
    v_wallet public.wallets%rowtype;
    v_existing_transaction public.wallet_transactions%rowtype;
    v_booking_id uuid;
    v_price_eur constant numeric(12, 2) := 10.00;
begin
    if p_user_id is null then
        raise exception 'An identified booking user is required' using errcode = '22023';
    end if;
    if p_idempotency_key is null or btrim(p_idempotency_key) = '' then
        raise exception 'Idempotency key is required' using errcode = '22023';
    end if;
    if p_court_id is null or btrim(p_court_id) = '' then
        raise exception 'Court ID is required' using errcode = '22023';
    end if;
    if p_sport is null or btrim(p_sport) = '' then
        raise exception 'Sport is required' using errcode = '22023';
    end if;
    if p_start_at is null or p_end_at is null or p_end_at <= p_start_at then
        raise exception 'Invalid booking interval' using errcode = '22023';
    end if;

    select id into v_tenant_id
      from public.tenants
     where slug = 'ntc' and is_active = true;
    if not found then
        raise exception 'Active NTC tenant does not exist' using errcode = 'P0002';
    end if;

    perform 1 from public.booking_users where id = p_user_id;
    if not found then
        raise exception 'Booking user % does not exist', p_user_id using errcode = 'P0002';
    end if;

    -- Serialize retries using the same caller-generated operation key.
    perform pg_advisory_xact_lock(hashtextextended('wallet-booking:' || p_idempotency_key, 0));

    select * into v_existing_transaction
      from public.wallet_transactions
     where idempotency_key = 'booking_charge:' || p_idempotency_key;
    if found then
        if v_existing_transaction.tenant_id <> v_tenant_id
           or v_existing_transaction.user_id <> p_user_id then
            raise exception 'Idempotency key was already used for another wallet operation'
                using errcode = '22023';
        end if;
        if not exists (
            select 1
              from public.bookings b
             where b.id = v_existing_transaction.booking_id
               and b.tenant_id = v_tenant_id
               and b.user_id = p_user_id
               and lower(b.court_id) = lower(btrim(p_court_id))
               and lower(b.sport) = lower(btrim(p_sport))
               and b.start_at = p_start_at
               and b.end_at = p_end_at
        ) then
            raise exception 'Idempotency key was reused with different booking data'
                using errcode = '22023';
        end if;
        select * into strict v_wallet
          from public.wallets
         where id = v_existing_transaction.wallet_id;
        return query
        select v_existing_transaction.booking_id,
               abs(v_existing_transaction.amount_eur),
               v_wallet.balance_eur,
               false;
        return;
    end if;

    -- Serialize bookings for this tenant/court before checking overlap.
    perform pg_advisory_xact_lock(
        hashtextextended('court:' || v_tenant_id::text || ':' || lower(btrim(p_court_id)), 0)
    );

    if exists (
        select 1
          from public.bookings b
         where b.tenant_id = v_tenant_id
           and lower(b.court_id) = lower(btrim(p_court_id))
           and lower(b.status) in ('confirmed', 'pending')
           and b.start_at < p_end_at
           and b.end_at > p_start_at
    ) then
        raise exception 'The selected court is no longer available' using errcode = '23P01';
    end if;

    select * into v_wallet
      from public.wallets
     where tenant_id = v_tenant_id and user_id = p_user_id
     for update;
    if not found then
        raise exception 'Wallet does not exist for this NTC user' using errcode = 'P0002';
    end if;
    if v_wallet.balance_eur < v_price_eur then
        raise exception 'Insufficient wallet balance' using errcode = 'P0001';
    end if;

    insert into public.bookings (
        tenant_id, user_id, court_id, sport, customer_name, customer_phone,
        start_at, end_at, status, notes
    ) values (
        v_tenant_id, p_user_id, lower(btrim(p_court_id)), lower(btrim(p_sport)),
        nullif(btrim(p_customer_name), ''), nullif(btrim(p_customer_phone), ''),
        p_start_at, p_end_at, 'confirmed', p_notes
    ) returning id into v_booking_id;

    insert into public.wallet_transactions (
        wallet_id, tenant_id, user_id, amount_eur, type, booking_id,
        reference_id, idempotency_key, metadata
    ) values (
        v_wallet.id, v_tenant_id, p_user_id, -v_price_eur, 'booking_charge',
        v_booking_id, v_booking_id::text, 'booking_charge:' || p_idempotency_key,
        jsonb_build_object('pricing_rule', 'temporary_flat_rate', 'price_eur', v_price_eur)
    );

    update public.wallets as w
       set balance_eur = w.balance_eur - v_price_eur
     where w.id = v_wallet.id
     returning w.* into v_wallet;

    return query select v_booking_id, v_price_eur, v_wallet.balance_eur, true;
end;
$$;

create or replace function public.wallet_refund_ntc_booking(
    p_booking_id uuid
)
returns table (
    booking_id uuid,
    refunded_eur numeric(12, 2),
    balance_eur numeric(12, 2),
    refunded boolean
)
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_tenant_id uuid;
    v_booking public.bookings%rowtype;
    v_charge public.wallet_transactions%rowtype;
    v_existing_refund public.wallet_transactions%rowtype;
    v_wallet public.wallets%rowtype;
    v_refund_eur numeric(12, 2);
begin
    if p_booking_id is null then
        raise exception 'Booking ID is required' using errcode = '22023';
    end if;

    select id into v_tenant_id from public.tenants where slug = 'ntc';
    if not found then
        raise exception 'NTC tenant does not exist' using errcode = 'P0002';
    end if;

    -- Serializes duplicate cancellation requests and locks the booking row.
    select * into v_booking
      from public.bookings
     where id = p_booking_id and tenant_id = v_tenant_id
     for update;
    if not found then
        raise exception 'NTC booking % does not exist', p_booking_id using errcode = 'P0002';
    end if;

    select wt.* into v_existing_refund
      from public.wallet_transactions as wt
     where wt.booking_id = p_booking_id and wt.type = 'refund';
    if found then
        select * into strict v_wallet
          from public.wallets
         where id = v_existing_refund.wallet_id;
        return query
        select p_booking_id, v_existing_refund.amount_eur, v_wallet.balance_eur, false;
        return;
    end if;

    select wt.* into v_charge
      from public.wallet_transactions as wt
     where wt.booking_id = p_booking_id and wt.type = 'booking_charge';
    if not found then
        raise exception 'Booking % has no wallet charge to refund', p_booking_id using errcode = 'P0002';
    end if;

    select * into strict v_wallet
      from public.wallets
     where id = v_charge.wallet_id
     for update;

    v_refund_eur := abs(v_charge.amount_eur);

    insert into public.wallet_transactions (
        wallet_id, tenant_id, user_id, amount_eur, type, booking_id,
        reference_id, idempotency_key, metadata
    ) values (
        v_wallet.id, v_charge.tenant_id, v_charge.user_id, v_refund_eur,
        'refund', p_booking_id, p_booking_id::text,
        'booking_refund:' || p_booking_id::text,
        jsonb_build_object('charge_transaction_id', v_charge.id)
    );

    update public.wallets as w
       set balance_eur = w.balance_eur + v_refund_eur
     where w.id = v_wallet.id
     returning w.* into v_wallet;

    update public.bookings set status = 'cancelled' where id = p_booking_id;

    return query select p_booking_id, v_refund_eur, v_wallet.balance_eur, true;
end;
$$;

revoke execute on function public.wallet_create_ntc_booking(
    uuid, text, text, text, text, timestamptz, timestamptz, text, text
) from public;
revoke execute on function public.wallet_refund_ntc_booking(uuid) from public;

do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        revoke execute on function public.wallet_create_ntc_booking(
            uuid, text, text, text, text, timestamptz, timestamptz, text, text
        ) from anon;
        revoke execute on function public.wallet_refund_ntc_booking(uuid) from anon;
    end if;

    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        revoke execute on function public.wallet_create_ntc_booking(
            uuid, text, text, text, text, timestamptz, timestamptz, text, text
        ) from authenticated;
        revoke execute on function public.wallet_refund_ntc_booking(uuid) from authenticated;
    end if;

    if exists (select 1 from pg_roles where rolname = 'service_role') then
        grant execute on function public.wallet_create_ntc_booking(
            uuid, text, text, text, text, timestamptz, timestamptz, text, text
        ) to service_role;
        grant execute on function public.wallet_refund_ntc_booking(uuid) to service_role;
    end if;
end;
$$;

commit;
