-- Auditable manual wallet adjustments for administration and controlled tests.
-- Never expose this function to frontend roles.

begin;

create or replace function public.wallet_manual_adjustment(
    p_tenant_id uuid,
    p_user_id uuid,
    p_amount_eur numeric(12, 2),
    p_reason text,
    p_idempotency_key text,
    p_metadata jsonb default '{}'::jsonb
)
returns table (
    transaction_id uuid,
    wallet_id uuid,
    balance_eur numeric(12, 2),
    created boolean
)
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_wallet public.wallets%rowtype;
    v_transaction public.wallet_transactions%rowtype;
    v_ledger_key text;
begin
    if p_amount_eur is null or p_amount_eur = 0 then
        raise exception 'Adjustment amount must be non-zero' using errcode = '22023';
    end if;
    if p_reason is null or btrim(p_reason) = '' then
        raise exception 'Adjustment reason is required' using errcode = '22023';
    end if;
    if p_idempotency_key is null or btrim(p_idempotency_key) = '' then
        raise exception 'Idempotency key is required' using errcode = '22023';
    end if;
    if p_metadata is null or jsonb_typeof(p_metadata) <> 'object' then
        raise exception 'Metadata must be a JSON object' using errcode = '22023';
    end if;

    perform 1 from public.tenants where id = p_tenant_id;
    if not found then
        raise exception 'Tenant % does not exist', p_tenant_id using errcode = 'P0002';
    end if;
    perform 1 from public.booking_users where id = p_user_id;
    if not found then
        raise exception 'Booking user % does not exist', p_user_id using errcode = 'P0002';
    end if;

    v_ledger_key := 'manual_adjustment:' || btrim(p_idempotency_key);
    perform pg_advisory_xact_lock(hashtextextended(v_ledger_key, 0));

    select * into v_transaction
      from public.wallet_transactions
     where idempotency_key = v_ledger_key;
    if found then
        if v_transaction.tenant_id <> p_tenant_id
           or v_transaction.user_id <> p_user_id
           or v_transaction.amount_eur <> p_amount_eur
           or v_transaction.metadata->>'reason' <> btrim(p_reason) then
            raise exception 'Idempotency key was reused with different adjustment data'
                using errcode = '22023';
        end if;
        select * into strict v_wallet
          from public.wallets
         where id = v_transaction.wallet_id;
        return query select v_transaction.id, v_wallet.id, v_wallet.balance_eur, false;
        return;
    end if;

    insert into public.wallets (tenant_id, user_id)
    values (p_tenant_id, p_user_id)
    on conflict (tenant_id, user_id) do nothing;

    select * into strict v_wallet
      from public.wallets
     where tenant_id = p_tenant_id and user_id = p_user_id
     for update;

    if v_wallet.balance_eur + p_amount_eur < 0 then
        raise exception 'Adjustment would make wallet balance negative' using errcode = 'P0001';
    end if;

    insert into public.wallet_transactions (
        wallet_id, tenant_id, user_id, amount_eur, type,
        reference_id, idempotency_key, metadata
    ) values (
        v_wallet.id, p_tenant_id, p_user_id, p_amount_eur, 'manual_adjustment',
        btrim(p_reason), v_ledger_key,
        p_metadata || jsonb_build_object('reason', btrim(p_reason))
    ) returning * into v_transaction;

    update public.wallets as w
       set balance_eur = w.balance_eur + p_amount_eur
     where w.id = v_wallet.id
     returning w.* into v_wallet;

    return query select v_transaction.id, v_wallet.id, v_wallet.balance_eur, true;
end;
$$;

revoke execute on function public.wallet_manual_adjustment(
    uuid, uuid, numeric, text, text, jsonb
) from public;

do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        revoke execute on function public.wallet_manual_adjustment(
            uuid, uuid, numeric, text, text, jsonb
        ) from anon;
    end if;
    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        revoke execute on function public.wallet_manual_adjustment(
            uuid, uuid, numeric, text, text, jsonb
        ) from authenticated;
    end if;
    if exists (select 1 from pg_roles where rolname = 'service_role') then
        grant execute on function public.wallet_manual_adjustment(
            uuid, uuid, numeric, text, text, jsonb
        ) to service_role;
    end if;
end;
$$;

commit;
