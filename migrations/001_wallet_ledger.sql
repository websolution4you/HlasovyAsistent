-- Telio EUR wallet ledger foundation.
-- Portable PostgreSQL migration: Supabase now, Google Cloud SQL later.
-- Do not expose the mutation functions to frontend/database client roles.

begin;

create table public.wallets (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references public.tenants(id) on delete restrict,
    user_id uuid not null references public.booking_users(id) on delete restrict,
    balance_eur numeric(12, 2) not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint wallets_tenant_user_key unique (tenant_id, user_id),
    constraint wallets_identity_key unique (id, tenant_id, user_id),
    constraint wallets_balance_eur_nonnegative check (balance_eur >= 0)
);

create table public.payments (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references public.tenants(id) on delete restrict,
    user_id uuid not null references public.booking_users(id) on delete restrict,
    amount_eur numeric(12, 2) not null,
    provider text not null,
    provider_payment_id text,
    idempotency_key text not null,
    status text not null default 'pending',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    paid_at timestamptz,
    posted_at timestamptz,
    last_checked_at timestamptz,
    error_message text,
    metadata jsonb not null default '{}'::jsonb,
    constraint payments_amount_positive check (amount_eur > 0),
    constraint payments_provider_not_blank check (btrim(provider) <> ''),
    constraint payments_idempotency_not_blank check (btrim(idempotency_key) <> ''),
    constraint payments_status_check check (
        status in ('pending', 'processing', 'paid', 'failed', 'cancelled')
    ),
    constraint payments_metadata_object check (jsonb_typeof(metadata) = 'object'),
    constraint payments_idempotency_key_key unique (idempotency_key),
    constraint payments_provider_payment_key unique (provider, provider_payment_id)
);

create table public.wallet_transactions (
    id uuid primary key default gen_random_uuid(),
    wallet_id uuid not null,
    tenant_id uuid not null,
    user_id uuid not null,
    amount_eur numeric(12, 2) not null,
    type text not null,
    payment_id uuid references public.payments(id) on delete restrict,
    booking_id uuid references public.bookings(id) on delete restrict,
    reference_id text,
    idempotency_key text not null,
    created_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb,
    constraint wallet_transactions_wallet_identity_fkey
        foreign key (wallet_id, tenant_id, user_id)
        references public.wallets(id, tenant_id, user_id)
        on delete restrict,
    constraint wallet_transactions_amount_eur_nonzero check (amount_eur <> 0),
    constraint wallet_transactions_type_check check (
        type in ('payment', 'booking_charge', 'refund', 'manual_adjustment', 'bonus')
    ),
    constraint wallet_transactions_sign_check check (
        (type in ('payment', 'refund', 'bonus') and amount_eur > 0)
        or (type = 'booking_charge' and amount_eur < 0)
        or (type = 'manual_adjustment' and amount_eur <> 0)
    ),
    constraint wallet_transactions_reference_check check (
        (type = 'payment' and payment_id is not null and booking_id is null)
        or (type in ('booking_charge', 'refund') and booking_id is not null and payment_id is null)
        or (type in ('manual_adjustment', 'bonus'))
    ),
    constraint wallet_transactions_idempotency_not_blank check (btrim(idempotency_key) <> ''),
    constraint wallet_transactions_metadata_object check (jsonb_typeof(metadata) = 'object'),
    constraint wallet_transactions_idempotency_key_key unique (idempotency_key)
);

-- Defense in depth against duplicate financial effects.
create unique index wallet_transactions_payment_key
    on public.wallet_transactions (payment_id) where type = 'payment';
create unique index wallet_transactions_booking_charge_key
    on public.wallet_transactions (booking_id) where type = 'booking_charge';
create unique index wallet_transactions_booking_refund_key
    on public.wallet_transactions (booking_id) where type = 'refund';

create index payments_reconciliation_idx
    on public.payments (status, last_checked_at, created_at)
    where status in ('pending', 'processing');
create index payments_wallet_owner_idx
    on public.payments (tenant_id, user_id, created_at desc);
create index wallet_transactions_wallet_created_idx
    on public.wallet_transactions (wallet_id, created_at, id);
create index wallet_transactions_tenant_user_created_idx
    on public.wallet_transactions (tenant_id, user_id, created_at desc);

create or replace function public.wallet_set_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at := now();
    return new;
end;
$$;

create trigger wallets_set_updated_at
before update on public.wallets
for each row execute function public.wallet_set_updated_at();
create trigger payments_set_updated_at
before update on public.payments
for each row execute function public.wallet_set_updated_at();

create or replace function public.wallet_reject_ledger_mutation()
returns trigger language plpgsql as $$
begin
    raise exception 'wallet_transactions is append-only; use a compensating transaction'
        using errcode = '55000';
end;
$$;

create trigger wallet_transactions_append_only
before update or delete on public.wallet_transactions
for each row execute function public.wallet_reject_ledger_mutation();

-- Call only after the backend has verified a successful payment directly
-- with the provider. A browser-provided status is never sufficient.
create or replace function public.wallet_process_successful_payment(
    p_payment_id uuid,
    p_provider_payment_id text default null,
    p_provider_metadata jsonb default '{}'::jsonb
)
returns table (payment_id uuid, wallet_id uuid, balance_eur numeric(12, 2), posted boolean)
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

    select * into v_payment
      from public.payments
     where id = p_payment_id
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
        select * into strict v_wallet
          from public.wallets
         where tenant_id = v_payment.tenant_id and user_id = v_payment.user_id;
        return query select v_payment.id, v_wallet.id, v_wallet.balance_eur, false;
        return;
    end if;

    update public.payments
       set provider_payment_id = coalesce(provider_payment_id, p_provider_payment_id),
           status = 'paid',
           paid_at = coalesce(paid_at, now()),
           last_checked_at = now(),
           error_message = null,
           metadata = metadata || p_provider_metadata
     where id = v_payment.id
     returning * into v_payment;

    if v_payment.provider_payment_id is null then
        raise exception 'A verified provider payment ID is required' using errcode = '23502';
    end if;

    insert into public.wallets (tenant_id, user_id)
    values (v_payment.tenant_id, v_payment.user_id)
    on conflict (tenant_id, user_id) do nothing;

    select * into strict v_wallet
      from public.wallets
     where tenant_id = v_payment.tenant_id and user_id = v_payment.user_id
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

    update public.payments set posted_at = now() where id = v_payment.id;
    return query select v_payment.id, v_wallet.id, v_wallet.balance_eur, true;
end;
$$;

-- Read-only integrity report. Differences must be alerted and investigated,
-- never silently repaired by rewriting the ledger.
create or replace function public.wallet_integrity_report()
returns table (
    wallet_id uuid,
    tenant_id uuid,
    user_id uuid,
    stored_balance_eur numeric(12, 2),
    ledger_balance_eur numeric(12, 2),
    difference_eur numeric(12, 2)
)
language sql
stable
security definer
set search_path = public, pg_temp
as $$
    select w.id, w.tenant_id, w.user_id, w.balance_eur,
           coalesce(sum(t.amount_eur), 0)::numeric(12, 2),
           (w.balance_eur - coalesce(sum(t.amount_eur), 0))::numeric(12, 2)
      from public.wallets w
      left join public.wallet_transactions t on t.wallet_id = w.id
     group by w.id, w.tenant_id, w.user_id, w.balance_eur
    having w.balance_eur <> coalesce(sum(t.amount_eur), 0);
$$;

-- Functions receive EXECUTE for PUBLIC by default; remove it explicitly.
revoke execute on function public.wallet_process_successful_payment(uuid, text, jsonb) from public;
revoke execute on function public.wallet_integrity_report() from public;
revoke execute on function public.wallet_set_updated_at() from public;
revoke execute on function public.wallet_reject_ledger_mutation() from public;

-- Supabase-specific roles do not necessarily exist in Google Cloud SQL, so all
-- grants/revokes for those roles are conditional. A future Cloud SQL migration
-- will grant equivalent rights only to Telio's dedicated backend database role.
do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        revoke all on public.wallets, public.payments, public.wallet_transactions from anon;
        revoke execute on function public.wallet_process_successful_payment(uuid, text, jsonb) from anon;
        revoke execute on function public.wallet_integrity_report() from anon;
    end if;

    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        revoke all on public.wallets, public.payments, public.wallet_transactions from authenticated;
        revoke execute on function public.wallet_process_successful_payment(uuid, text, jsonb) from authenticated;
        revoke execute on function public.wallet_integrity_report() from authenticated;
    end if;

    if exists (select 1 from pg_roles where rolname = 'service_role') then
        -- The backend may create payment intents and read wallet history, but it
        -- cannot directly mutate balances or append financial ledger entries.
        grant select on public.wallets, public.wallet_transactions to service_role;
        grant select, insert on public.payments to service_role;
        grant execute on function public.wallet_process_successful_payment(uuid, text, jsonb) to service_role;
        grant execute on function public.wallet_integrity_report() to service_role;
    end if;
end;
$$;

commit;
