-- Replaces the payment posting function after qualifying wallet balance columns.
-- Safe to apply after 001_wallet_ledger.sql; it does not recreate any tables.

begin;

create or replace function public.wallet_process_successful_payment(
    p_payment_id uuid,
    p_provider_payment_id text default null,
    p_provider_metadata jsonb default '{}'::jsonb
)
returns table (
    payment_id uuid,
    wallet_id uuid,
    balance_eur numeric(12, 2),
    posted boolean
)
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_payment public.payments%rowtype;
    v_wallet public.wallets%rowtype;
begin
    if p_provider_metadata is null or jsonb_typeof(p_provider_metadata) <> 'object' then
        raise exception 'Provider metadata must be a JSON object' using errcode = '22023';
    end if;

    select p.* into v_payment
      from public.payments as p
     where p.id = p_payment_id
     for update;
    if not found then
        raise exception 'Payment % does not exist', p_payment_id using errcode = 'P0002';
    end if;

    if p_provider_payment_id is not null and btrim(p_provider_payment_id) = '' then
        raise exception 'Provider payment ID cannot be blank' using errcode = '22023';
    end if;
    if v_payment.provider_payment_id is not null
       and p_provider_payment_id is not null
       and v_payment.provider_payment_id <> p_provider_payment_id then
        raise exception 'Provider payment ID does not match stored payment' using errcode = '22023';
    end if;

    if v_payment.posted_at is not null then
        select w.* into strict v_wallet
          from public.wallets as w
         where w.tenant_id = v_payment.tenant_id
           and w.user_id = v_payment.user_id;
        return query select v_payment.id, v_wallet.id, v_wallet.balance_eur, false;
        return;
    end if;

    update public.payments as p
       set provider_payment_id = coalesce(p.provider_payment_id, p_provider_payment_id),
           status = 'paid',
           paid_at = coalesce(p.paid_at, now()),
           last_checked_at = now(),
           error_message = null,
           metadata = p.metadata || p_provider_metadata
     where p.id = v_payment.id
     returning p.* into v_payment;

    if v_payment.provider_payment_id is null then
        raise exception 'A verified provider payment ID is required' using errcode = '23502';
    end if;

    insert into public.wallets (tenant_id, user_id)
    values (v_payment.tenant_id, v_payment.user_id)
    on conflict (tenant_id, user_id) do nothing;

    select w.* into strict v_wallet
      from public.wallets as w
     where w.tenant_id = v_payment.tenant_id
       and w.user_id = v_payment.user_id
     for update;

    insert into public.wallet_transactions (
        wallet_id, tenant_id, user_id, amount_eur, type, payment_id,
        reference_id, idempotency_key, metadata
    ) values (
        v_wallet.id, v_payment.tenant_id, v_payment.user_id, v_payment.amount_eur,
        'payment', v_payment.id, v_payment.provider_payment_id,
        'payment:' || v_payment.provider || ':' || v_payment.provider_payment_id,
        jsonb_build_object('provider', v_payment.provider)
    );

    update public.wallets as w
       set balance_eur = w.balance_eur + v_payment.amount_eur
     where w.id = v_wallet.id
     returning w.* into v_wallet;

    update public.payments as p
       set posted_at = now()
     where p.id = v_payment.id;

    return query select v_payment.id, v_wallet.id, v_wallet.balance_eur, true;
end;
$$;

revoke execute on function public.wallet_process_successful_payment(
    uuid, text, jsonb
) from public;

do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        revoke execute on function public.wallet_process_successful_payment(
            uuid, text, jsonb
        ) from anon;
    end if;
    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        revoke execute on function public.wallet_process_successful_payment(
            uuid, text, jsonb
        ) from authenticated;
    end if;
    if exists (select 1 from pg_roles where rolname = 'service_role') then
        grant execute on function public.wallet_process_successful_payment(
            uuid, text, jsonb
        ) to service_role;
    end if;
end;
$$;

commit;
