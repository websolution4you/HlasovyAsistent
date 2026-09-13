-- >>> START OF migrations/000_core_from_supabase.sql <<<
-- ==========================================================
-- Core Telio schema and test data exported from Supabase
-- Target: Google Cloud SQL (PostgreSQL)
-- ==========================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Bezpečné prečistenie pred čistým importom
DROP TABLE IF EXISTS public.wallet_transactions CASCADE;
DROP TABLE IF EXISTS public.payments CASCADE;
DROP TABLE IF EXISTS public.wallets CASCADE;
DROP TABLE IF EXISTS public.calendar_connections CASCADE;
DROP TABLE IF EXISTS public.bookings CASCADE;
DROP TABLE IF EXISTS public.booking_users CASCADE;
DROP TABLE IF EXISTS public.tenants CASCADE;

-- 1. TENANTS
CREATE TABLE IF NOT EXISTS public.tenants (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slug text NOT NULL UNIQUE,
    name text NOT NULL,
    project_type text,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- 2. BOOKING USERS
CREATE TABLE IF NOT EXISTS public.booking_users (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL,
    email text,
    password_hash text,
    card_number text,
    role text NOT NULL DEFAULT 'user',
    phone text,
    has_multisport boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_booking_users_phone ON public.booking_users(phone);

-- 3. BOOKINGS
CREATE TABLE IF NOT EXISTS public.bookings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL REFERENCES public.tenants(id) ON DELETE RESTRICT,
    user_id uuid REFERENCES public.booking_users(id) ON DELETE SET NULL,
    court_id text,
    sport text,
    customer_name text,
    customer_phone text,
    start_at timestamptz NOT NULL,
    end_at timestamptz NOT NULL,
    status text NOT NULL DEFAULT 'confirmed',
    notes text,
    call_id text,
    service_id text,
    calendar_event_id text,
    guest_count integer,
    table_id text,
    price_eur numeric(12, 2),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bookings_tenant_time ON public.bookings(tenant_id, start_at, end_at);
CREATE INDEX IF NOT EXISTS idx_bookings_court_time ON public.bookings(court_id, start_at, end_at);

-- 4. CALENDAR CONNECTIONS (if used)
CREATE TABLE IF NOT EXISTS public.calendar_connections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid REFERENCES public.tenants(id) ON DELETE CASCADE,
    provider text NOT NULL,
    calendar_id text,
    access_token text,
    refresh_token text,
    token_expiry timestamptz,
    credentials jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- ==========================================================
-- DATA INSERTS
-- ==========================================================

-- Data for public.tenants (4 rows)
INSERT INTO public.tenants (id, name, project_type, slug, is_active, created_at)
VALUES
    ('3e803e21-3177-42ed-974c-a1a1b260b88b', 'Pizza Levoča', 'pizza', 'pizza-levoca', TRUE, '2026-03-07T16:42:45.153249+00:00'),
    ('6076dbf2-8f14-432d-b27d-504ae30b94b2', 'Taxi Košice', 'taxi', 'taxi-kosice', TRUE, '2026-03-07T16:42:45.153249+00:00'),
    ('ac3b439d-a446-4d67-abf7-5e04f58647fe', 'Peter Dev', 'pizza', 'peter-dev-pizza', TRUE, '2026-03-07T16:45:31.816105+00:00'),
    ('595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', 'NTC', 'pizza', 'ntc', TRUE, '2026-06-20T20:18:08.423805+00:00')
ON CONFLICT DO NOTHING;

-- Data for public.booking_users (16 rows)
INSERT INTO public.booking_users (id, name, email, password_hash, card_number, created_at, updated_at, role, phone, has_multisport)
VALUES
    ('6392e3cf-c8a1-487b-adde-394e1a0ea31f', 'Google Test', 'googletest@ntc.sk', 'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f', '0000', '2026-08-10T20:47:19.373445+00:00', '2026-08-31T19:06:19.511766+00:00', 'trainer', '000000000000', FALSE),
    ('e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'Peter Kaľavský', 'peto@ntc.sk', 'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f', '0002', '2026-07-02T17:55:07.056012+00:00', '2026-09-06T06:11:00.356165+00:00', 'user', '+421905012054', FALSE),
    ('689f3f3b-5596-4859-bc1a-f26cdfa9ba36', 'Kamil Tomčofčík', 'tomcofcik@ntc.sk', 'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f', '0005', '2026-07-27T12:04:41.738365+00:00', '2026-09-06T06:11:07.052598+00:00', 'user', '+421907173176', FALSE),
    ('a72eab08-214c-43ad-b041-d7380ca801a0', 'Štefan Varga', 'varga777.sk@gmail.com', '2b3e753379e6c9a43479e3659fe2e0fd091ee1971364cabd40b4780e984a49fa', '0006', '2026-08-20T10:14:14.278331+00:00', '2026-09-06T06:11:18.288028+00:00', 'user', '0907389210', FALSE),
    ('c6b90673-97cf-4ec2-a9df-644596a2a05a', 'Admin User', 'admin@ntc.sk', 'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f', '0000', '2026-06-30T17:06:43.32569+00:00', '2026-08-23T08:00:48.925614+00:00', 'admin', '+421900000000', FALSE),
    ('6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'Jozef Mrkvička', 'jozef@ntc.sk', 'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f', '1238', '2026-08-03T10:52:38.074699+00:00', '2026-08-23T08:00:48.925614+00:00', 'user', '09159620', FALSE),
    ('2692ab42-4f19-4bd4-9d47-83e50c8fae44', 'Milan Bugan', 'buganovci.lama@gmail.com', '99b8d92580796d7352a57c4025d0ab4b4c5e12e6bc036b6adb9a77a8855cec96', '2503', '2026-08-19T08:30:16.82391+00:00', '2026-08-23T08:00:48.925614+00:00', 'user', '0948291231', FALSE),
    ('c6ff6936-a630-4ecd-8cb8-7bdc2289607f', 'Oliver Urda', 'superstruper1@gmail.com', 'aa0213dcb7d57ed593d2975e73ff852806770a064f96a9b0ff40310bb30f2e16', '1234', '2026-08-19T08:58:46.599726+00:00', '2026-08-23T08:00:48.925614+00:00', 'user', '0944420304', FALSE),
    ('a9f2b336-721f-4046-830d-b6fea9863703', 'Jankovič Ján', 'jankovic@ntc.sk', 'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f', NULL, '2026-09-05T11:51:25.07753+00:00', '2026-09-05T11:51:25.07753+00:00', 'user', '+421900123456', FALSE),
    ('b1fb24b3-3f4b-4b0a-902d-47e65c9f48fe', 'Janka Bartková', 'bartkovajan@gmail.com', 'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f', '0008', '2026-08-21T08:50:39.466334+00:00', '2026-09-06T06:10:49.987052+00:00', 'user', '0908819991', FALSE),
    ('c5fe604e-6b4b-4a20-94da-c9e53cc8802a', 'Lucia Bartková', 'luciabartkova1@gmail.com', 'ef5fdb8b0cb6517ff13cd9c84cc41018f14925093dbde49431652f19df4817db', '0009', '2026-08-21T11:21:53.55021+00:00', '2026-09-06T06:10:54.905225+00:00', 'user', '0907304358', FALSE),
    ('42a0f4f7-0184-4232-8c08-d1d3f591257e', '[Zmazaný používateľ]', 'deleted_42a0f4f7-0184-4232-8c08-d1d3f591257e@deleted.local', 'deleted_2f88b14b-da75-4c05-80c5-ef73b6793c24', NULL, '2026-09-06T06:41:45.519915+00:00', '2026-09-06T06:55:53.538404+00:00', 'user', NULL, FALSE),
    ('0af7d833-6af6-4841-a815-8064aa22ccd0', 'Jozef Priezvisko', 'dsz@654.sk', '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92', '2506', '2026-09-06T07:00:07.203812+00:00', '2026-09-06T07:00:07.203812+00:00', 'user', '+421900555556', FALSE),
    ('e616dfd7-3777-4571-a831-0d47ccfc7618', 'Milan Polanský', 'polansky@ntc.sk', 'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f', '2507', '2026-09-06T07:19:44.431463+00:00', '2026-09-06T07:19:44.431463+00:00', 'user', '+421900543216564', FALSE),
    ('5009e9ad-5bff-4cf5-8190-fd26c576cac6', 'Novy Pouzivatel', 'novy@ntc.sk', 'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f', '5544', '2026-09-06T19:58:30.828714+00:00', '2026-09-06T19:58:30.828714+00:00', 'user', '00421915962067', FALSE),
    ('8b86a015-0924-4780-8264-b9beabc3519f', 'Kamil Bartko', 'kamil@ntc.sk', 'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f', '1235', '2026-07-02T17:55:07.056012+00:00', '2026-09-10T10:30:56.676707+00:00', 'user', '+421915962068', FALSE)
ON CONFLICT DO NOTHING;

-- Data for public.bookings (647 rows)
INSERT INTO public.bookings (id, tenant_id, call_id, service_id, customer_name, customer_phone, start_at, end_at, status, calendar_event_id, notes, guest_count, created_at, table_id, user_id, sport, court_id, price_eur)
VALUES
    ('14460f6d-e3b4-4be6-9b6b-932c2e9b0c1f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kano Bartko', '', '2026-06-24T14:00:00+00:00', '2026-06-24T15:00:00+00:00', 'confirmed', '0c5c637fb5ld5pskd1nkaq3gd4', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "tennis-3"}', NULL, '2026-06-23T08:04:40.659403+00:00', NULL, NULL, 'tennis', 'tennis-3', NULL),
    ('9acb8e91-9990-4fb4-8079-7ef3af8ff592', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartková', '', '2026-06-25T15:00:00+00:00', '2026-06-25T16:00:00+00:00', 'confirmed', 'jv02e3sdnc53r1sefq7ja7oo54', '{"notes": "Standard rate, no membership card", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-06-23T08:31:08.686313+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('93d55104-7f1d-476d-aafb-a10b600e9ef6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-25T09:00:00+00:00', '2026-06-25T10:00:00+00:00', 'confirmed', '3dlkvd8sj9sq635ibu058bjmbk', '{"notes": "Standard rate booking", "source": "azure-realtime-voice-assistant", "courtId": "badminton-3"}', NULL, '2026-06-23T14:03:38.007075+00:00', NULL, NULL, 'badminton', 'badminton-3', NULL),
    ('0e765367-999f-41da-a335-9cea2184a43c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-25T14:00:00+00:00', '2026-06-25T15:00:00+00:00', 'confirmed', '0uftk5hv76i0dk3f3cu0a8ckj8', '{"notes": "Standard rate, no membership card", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-23T15:07:46.009411+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('185913a0-46c5-4b31-9b1f-de66ffe1900a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-25T18:00:00+00:00', '2026-06-25T19:00:00+00:00', 'confirmed', 'b7va92gjateq3na2nmnlmhionk', '{"notes": "Standard rate booking without membership card", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-23T15:12:14.713154+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('97ed35ea-b572-43a3-bb13-882940af632d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-25T11:00:00+00:00', '2026-06-25T12:00:00+00:00', 'confirmed', 'alm0na4knoka2mqjv7h8pjdjok', '{"notes": "Standard booking without membership card", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-06-23T15:02:29.244266+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('f5126e5d-3632-4ebb-ae3f-c0f474306f5f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '', '2026-06-25T17:00:00+00:00', '2026-06-25T18:00:00+00:00', 'confirmed', 'bv0a1h21aa13ni3k4ris36s4is', '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-3"}', NULL, '2026-06-23T16:57:42.657688+00:00', NULL, NULL, 'badminton', 'badminton-3', NULL),
    ('73fbfbd3-a95a-4eb5-9168-3b15b7ee3ace', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '0905654987', '2026-06-23T13:00:00+00:00', '2026-06-23T14:00:00+00:00', 'confirmed', 'mcmhon3vl57uq2ioh5ts5jdpl0', '{"notes": "Stvorhra", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-23T20:05:19.844611+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('0fa5e55f-7ffc-40d3-9f59-586bbb61b3be', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'aaa', '0000', '2026-06-23T10:00:00+00:00', '2026-06-23T11:30:00+00:00', 'confirmed', 'u7shj4ci6dolvr69s20uqmda64', '{"notes": "aaa", "source": "azure-realtime-voice-assistant", "courtId": "badminton-3"}', NULL, '2026-06-23T20:15:33.489378+00:00', NULL, NULL, 'badminton', 'badminton-3', NULL),
    ('14953cad-c4e6-4aa2-ae6b-9838569c6bbc', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartková', '', '2026-06-24T11:00:00+00:00', '2026-06-24T12:00:00+00:00', 'confirmed', 'k4t1m2rkufpoeqrv4ffhffnonk', '{"notes": "Standard rate, no membership card", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-06-23T20:23:44.436383+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('1e79ce51-d583-4080-8e4f-acc394811017', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Líška', '', '2026-06-24T13:00:00+00:00', '2026-06-24T14:00:00+00:00', 'confirmed', 'lu54uqi8b71kftv556phkbdr1s', '{"notes": "Standard booking without membership card", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-06-23T20:27:40.583418+00:00', NULL, NULL, 'tennis', 'tennis-2', NULL),
    ('80e81e5f-aa60-472f-8a39-b2bc63bf2844', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-06-25T14:00:00+00:00', '2026-06-25T15:00:00+00:00', 'confirmed', '3v5fr091pm2fid51p0daltos94', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-06-24T05:16:02.819703+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('a96ee3e5-e458-493c-a499-4984fc7d740c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-06-25T08:00:00+00:00', '2026-06-25T09:00:00+00:00', 'confirmed', '6lphlqq37dmptggfbpvlvignh4', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-24T05:43:03.957139+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('f5864b6e-0f8b-493e-b0a6-dc6bc82cab02', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-06-25T08:00:00+00:00', '2026-06-25T09:00:00+00:00', 'confirmed', 'cgda3av9ntddrtum304k7au200', '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-06-24T05:52:47.240456+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('a7eaa95b-c321-4744-bc23-fedaf0a1f687', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-06-25T08:00:00+00:00', '2026-06-25T09:00:00+00:00', 'confirmed', 'b96irbmkps9q6o8v8ofbva23t4', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-06-24T05:53:53.227279+00:00', NULL, NULL, 'tennis', 'tennis-2', NULL),
    ('89c1d4e8-3795-40ef-9721-5a4fa098e318', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-06-25T08:00:00+00:00', '2026-06-25T09:00:00+00:00', 'confirmed', 'h84c6nbevje90kul2pd8agthfk', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-3"}', NULL, '2026-06-24T05:55:06.623537+00:00', NULL, NULL, 'tennis', 'tennis-3', NULL),
    ('34764202-b202-4bc1-b2c0-4d635a3095be', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-06-25T08:00:00+00:00', '2026-06-25T09:00:00+00:00', 'confirmed', 'v1euvdtgg7iee67aspe9hqv3bk', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-4"}', NULL, '2026-06-24T05:56:17.038739+00:00', NULL, NULL, 'tennis', 'tennis-4', NULL),
    ('3c089f94-dc0b-4e58-bd3c-9f8fcaaae622', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-24T09:00:00+00:00', '2026-06-24T10:00:00+00:00', 'confirmed', 'lerddvhar364447dta3u7c4lp8', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-06-24T06:36:10.124943+00:00', NULL, NULL, 'tennis', 'tennis-2', NULL),
    ('55de8be6-ca7a-4904-ba57-0630aa207876', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-24T12:00:00+00:00', '2026-06-24T13:00:00+00:00', 'confirmed', 'jirbb4q14a6jvf5uj0bhvauvrk', '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-06-24T06:38:11.829805+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('66bf53e5-5301-45e8-8753-5c09705fe792', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-14T12:00:00+00:00', '2026-07-14T13:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-13T06:56:27.748575+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('436652ab-5f2e-46db-ae0a-412ba6666262', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-24T13:00:00+00:00', '2026-06-24T14:30:00+00:00', 'confirmed', 'o97sb74mommo0sumsvutklhlmo', '{"notes": "Bez členskej karty", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-06-24T06:42:17.421715+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('ca5d8fde-1fe5-45ca-8d77-7078646fa9c2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', NULL, '2026-06-25T10:00:00+00:00', '2026-06-25T11:00:00+00:00', 'confirmed', 'fbvctbd2mdu5cp6dkk63gvuc44', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-3"}', NULL, '2026-06-24T07:15:24.587486+00:00', NULL, NULL, 'badminton', 'badminton-3', NULL),
    ('5dc9520a-0fb1-491e-9c58-35ffad56664b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Urda', '', '2026-06-24T14:00:00+00:00', '2026-06-24T15:00:00+00:00', 'confirmed', 'tjm530jdv60p0i5smigcs4d8vk', '{"notes": "Standard booking without membership card", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-06-24T09:40:53.289522+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('ecee6d8b-28e6-44f5-963c-731b25f57e4b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '', '2026-06-26T14:00:00+00:00', '2026-06-26T15:00:00+00:00', 'confirmed', 'jpt76r33cu5d6u00ikjmm1mhvk', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-06-24T15:55:22.385664+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('fbe65f78-49dd-4b4a-b20d-1c966ad7f461', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Barkko', '', '2026-06-27T09:00:00+00:00', '2026-06-27T10:00:00+00:00', 'confirmed', 'oo9phlc32tar4a7r5q388hsqfo', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-06-24T16:00:05.087384+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('b2b59d68-4778-4ddc-9d27-c034d28e0b28', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-04T16:00:00+00:00', '2026-07-04T17:00:00+00:00', 'confirmed', '6gif5uifdh3nuj3vaqf8rfsfbo', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-7"}', NULL, '2026-07-04T15:16:42.315683+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-7', NULL),
    ('2a2f6110-f7be-4ca3-b6f5-619584c1edd5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-26T10:00:00+00:00', '2026-06-26T11:00:00+00:00', 'confirmed', 'vovrtla0pm76gpf932m8oi4ql8', '{"notes": "Standard booking without membership card", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-06-25T13:02:37.5402+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('ae6d8680-2c38-4417-804c-31b3f9faf7db', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-27T08:00:00+00:00', '2026-06-27T09:00:00+00:00', 'confirmed', '0d0hh68fr5o13e383ogj2im7e8', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "tennis-3"}', NULL, '2026-06-25T14:24:19.023476+00:00', NULL, NULL, 'tennis', 'tennis-3', NULL),
    ('3e9bdb56-8126-4cee-b9be-053b3020a544', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-27T11:00:00+00:00', '2026-06-27T12:00:00+00:00', 'confirmed', '7d5737t6lp96sj20f30bqsnf0k', '{"notes": "Standard booking without membership", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-27T09:32:38.27715+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('3641d0df-4f9c-445b-b871-907a783bdd46', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Lutina', '', '2026-06-27T11:00:00+00:00', '2026-06-27T12:00:00+00:00', 'confirmed', 'g026qna36gh8f7fp4ivf78be38', '{"notes": "Standard rate, no membership card", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-06-27T09:36:22.983286+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('c911951a-1de5-4c39-9ac0-ac253307405d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '0905646454', '2026-06-20T05:00:00+00:00', '2026-06-20T06:00:00+00:00', 'confirmed', '8do91u6ut9ond9ibnsvllhik64', '{"notes": "dvojhra", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-06-20T21:36:33.567673+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('8c691fd6-02a5-4214-8ef2-05337fa5d532', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-28T13:00:00+00:00', '2026-06-28T14:00:00+00:00', 'confirmed', 'f95idepoqi9fsuu1l55a8daljs', '{"notes": "NTC členská karta", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-27T09:50:53.699384+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('237bfe45-be2a-4051-8fcf-0d3b008b9d94', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-29T11:00:00+00:00', '2026-06-29T12:00:00+00:00', 'confirmed', 'lcp9lnjagbfld335u53slgndeg', '{"notes": "Klient ma clensku kartu NTC.", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-27T10:10:51.963114+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('1ba30886-4d2d-4cd7-a63a-5ceed1539557', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'LB', '0519198', '2026-06-29T10:00:00+00:00', '2026-06-29T11:00:00+00:00', 'confirmed', 'ip7498mq42uks1rls3nuctmdoc', '{"notes": "dvojhra", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-27T10:11:32.631839+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('718d0e5d-50f7-4881-b4ea-6ec28bcefca1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'LB', '0519198', '2026-06-29T10:00:00+00:00', '2026-06-29T11:00:00+00:00', 'confirmed', '8pmtir0grfhb2os9h3n5hmu67c', '{"notes": "dvojhra", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-27T10:11:35.031735+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('f4a10311-fef8-4eaf-99cc-7cdae0be67f1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', NULL, '2026-06-27T12:00:00+00:00', '2026-06-27T13:00:00+00:00', 'confirmed', 'rvhi1u6hr88s9n3gmddsj10dj4', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-27T15:26:55.669874+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('358ce739-1897-401a-b6cb-cb6f3dab6d77', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, '-', NULL, '2026-06-27T12:00:00+00:00', '2026-06-27T14:00:00+00:00', 'confirmed', 'al93ljcias8ba9et1pkv6de1pg', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-06-27T15:27:54.444937+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('f4e164eb-d45f-476c-95d8-3f4b61a40a43', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Meno Meno', '', '2026-06-27T16:00:00+00:00', '2026-06-27T17:00:00+00:00', 'confirmed', '2gkgrlh5kvqgaf8j08j1clu15s', '{"notes": "Standard booking without membership card", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-06-27T15:47:22.072947+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('25195ac5-5090-4115-be32-bac3b20f22e3', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'pista', '', '2026-06-27T17:00:00+00:00', '2026-06-27T18:00:00+00:00', 'confirmed', 'h77pjinagp3ppfa2qkgb16tls4', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-27T15:50:29.254842+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('7450c310-a3b2-4c2a-8ac4-70225ea9bc97', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Pista Zase', '', '2026-06-27T17:00:00+00:00', '2026-06-27T18:30:00+00:00', 'confirmed', 'tc1s6h1631k86jtu6r3rnub05g', '{"notes": "NTC členská karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-06-27T15:51:27.89243+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('2ae792aa-e4f0-47a8-a547-543bb694ad74', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-01T17:00:00+00:00', '2026-07-01T18:00:00+00:00', 'confirmed', NULL, '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-01T15:00:28.848798+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('27d32546-aeaa-49bd-9a40-49b48e1c6a49', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Žolimsky', '', '2026-07-01T17:00:00+00:00', '2026-07-01T18:00:00+00:00', 'confirmed', 'k13nlq0punj87ci1e36d7bki4s', '{"notes": "Standard rate, no membership card", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-01T15:19:27.851352+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('f0f91add-ec72-451d-baf5-8ae8cb6c5edb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Leščinský', '', '2026-07-03T12:00:00+00:00', '2026-07-03T14:00:00+00:00', 'confirmed', 'tsifg85aaqqrqi6pr6vo8049lo', '{"notes": "Člen NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-02T15:15:57.420328+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('37d5ff98-b549-48d8-a779-b82f9fddad82', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, '-', NULL, '2026-07-02T08:00:00+00:00', '2026-07-02T09:00:00+00:00', 'confirmed', NULL, '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-2"}', NULL, '2026-07-02T17:07:12.8437+00:00', NULL, NULL, 'tennis-clay', 'tennis-clay-2', NULL),
    ('80652ab6-aa5b-43de-9bc8-80c386233277', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto', NULL, '2026-07-02T15:00:00+00:00', '2026-07-02T16:00:00+00:00', 'confirmed', 'burqf7r2h41epknccd705rcl4g', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-02T18:12:29.580927+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', NULL),
    ('703423ba-56d4-45a1-9ab9-5836fac63322', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 's', NULL, '2026-07-04T16:00:00+00:00', '2026-07-04T17:00:00+00:00', 'confirmed', '8le5pm4m545lb0sgsg64bpk0as', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-04T14:45:24.726093+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', NULL),
    ('777250fd-f270-48c7-b54b-28ce7a1420e2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-03T11:00:00+00:00', '2026-07-03T14:00:00+00:00', 'confirmed', 'l573c6r3nsql5dn3fc1iidpi74', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-07-02T20:09:11.604394+00:00', NULL, NULL, 'tennis', 'tennis-2', NULL),
    ('2fa6ebce-50a4-4469-83c9-1ed93036ab70', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jelensky', '', '2026-07-05T13:00:00+00:00', '2026-07-05T14:00:00+00:00', 'confirmed', 'pruao7ku06bbo4ndcdmtv8qhmg', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-02T20:14:14.758449+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('97055e2e-e6bc-4354-b6b1-3d23432db6b3', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Olivia Rodrigová', '', '2026-07-10T14:00:00+00:00', '2026-07-10T15:00:00+00:00', 'confirmed', 'ia4r921as3o21mnl7helcggp4o', '{"notes": "NTC členská karta", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-02T20:17:26.443245+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('709a7ab5-8496-4b05-9c14-4d60695c7b21', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil', '0505505', '2026-07-10T07:00:00+00:00', '2026-07-10T08:00:00+00:00', 'confirmed', 'cvj02phscs4ae494qb73lbtde0', '{"notes": "dvojhra", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-02T20:49:42.713386+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('ca9a859e-d30a-44dd-87c4-a9efb563c667', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil', '0505', '2026-07-03T07:00:00+00:00', '2026-07-03T08:00:00+00:00', 'confirmed', 'j0gs06k65ndcacuquv0fg5k2ck', '{"notes": "dvojhra", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-02T20:53:07.162263+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('829f33ce-9d43-4adc-b886-326cddb2ced5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto', NULL, '2026-07-03T14:00:00+00:00', '2026-07-03T15:00:00+00:00', 'confirmed', 'tp6urjjrma28mt9pvh9gb18rds', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-02T20:53:47.713888+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('d259f19b-c773-4c37-baad-205e90d1f00a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'n', NULL, '2026-07-03T09:00:00+00:00', '2026-07-03T10:00:00+00:00', 'confirmed', 'f8qrlps97u16ff6onh3mr7p9ak', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-03T07:26:49.231029+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('0101f537-b1ec-40d7-b38b-952e842d56a8', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kovalik', '', '2026-07-03T13:00:00+00:00', '2026-07-03T14:00:00+00:00', 'confirmed', 'u7nldbrsnpjbr6039qu2i9p0og', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-03T10:11:53.647332+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('d924ac6b-ebdb-4223-8c6d-e04a9549884b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Marek', '055505', '2026-07-03T14:00:00+00:00', '2026-07-03T15:00:00+00:00', 'confirmed', '4uo2ko5ns99pkinlenk0o15rv4', '{"notes": "dvojhra", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-03T10:13:40.79447+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('544760be-0f6b-4e05-ad86-b119c6cb52c1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Barko', '', '2026-07-03T15:00:00+00:00', '2026-07-03T16:00:00+00:00', 'confirmed', 'lm5q1f5qjvs2cadgb60fjs7v6g', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-03T11:08:42.659221+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('79118dfb-0a1b-4df4-a8e4-83b1d4eeceb8', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Test', NULL, '2026-07-07T12:00:00+00:00', '2026-07-07T13:00:00+00:00', 'confirmed', 'mqu7ha9qeil8ek43rl2q8hkauo', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-04T07:32:14.854945+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('a957365b-8efc-4269-a6e2-2faa128b9d7a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', NULL, '2026-07-04T18:00:00+00:00', '2026-07-04T19:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-04T15:05:29.657792+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', NULL),
    ('a0604a79-01e6-48f4-b266-6771f1d8c83e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', NULL, '2026-07-04T17:00:00+00:00', '2026-07-04T17:30:00+00:00', 'confirmed', 'mr6k4ifmhidoipe784tqls7pd0', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-04T15:05:41.017549+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', NULL),
    ('0e0e2c5d-efcd-48ff-baf0-c543dd4487cb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Líška', '', '2026-06-26T18:00:00+00:00', '2026-06-26T19:00:00+00:00', 'confirmed', 'pl5e6qjil159ejkn951c266gfo', '{"notes": "Standard booking without membership", "source": "azure-realtime-voice-assistant", "courtId": "badminton-7"}', NULL, '2026-06-25T08:27:56.724433+00:00', NULL, NULL, 'badminton', 'badminton-7', NULL),
    ('da62aa27-35ca-42a2-8201-81b5fb789dea', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', NULL, '2026-07-04T17:00:00+00:00', '2026-07-04T18:30:00+00:00', 'confirmed', 'pskuarsm2eusptmkpgf69ih86s', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-04T15:05:47.530508+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-2', NULL),
    ('52d0d533-fb9f-4952-a505-0266e9203398', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', NULL, '2026-07-04T17:00:00+00:00', '2026-07-04T18:00:00+00:00', 'confirmed', 'u9u0j2bve0dnlh5tv9en2i81k8', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-04T15:18:10.857127+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'tennis', 'tennis-1', NULL),
    ('95e058b7-2ff8-497e-9833-d04203c2225d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', NULL, '2026-07-04T18:00:00+00:00', '2026-07-04T20:00:00+00:00', 'confirmed', 's3huvhmq8749r1o4tjsk1jfm44', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-07-04T15:20:38.14154+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'tennis', 'tennis-2', NULL),
    ('abe11b6b-2f51-4e76-b41f-776e7b04e834', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-04T16:00:00+00:00', '2026-07-04T17:00:00+00:00', 'confirmed', 'o8jo136keij2vroqjlnd32rfuo', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-3"}', NULL, '2026-07-04T15:24:18.480273+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', NULL),
    ('a12aa61e-0cfb-4ebb-95f9-f4e016fa9e4e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-04T17:00:00+00:00', '2026-07-04T18:00:00+00:00', 'confirmed', 'r7mesuq15q84jh4764ul4ql73g', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-07-04T15:26:02.856705+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('ff5a335f-82e7-4cee-9ed7-ac71de5cd694', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-04T18:00:00+00:00', '2026-07-04T19:00:00+00:00', 'confirmed', 'vc29mn1njh9v0mrc6vocjos62c', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-04T15:48:41.014728+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('dde34356-1aa7-400d-bad3-ec5075862326', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', NULL, '2026-07-05T11:00:00+00:00', '2026-07-05T12:00:00+00:00', 'confirmed', '4hqajtlvjdkkm0v52bbnljo22g', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-07-04T15:56:13.390237+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-2', NULL),
    ('85989e02-87ec-483a-b4da-db2831902444', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-06T09:00:00+00:00', '2026-07-06T10:00:00+00:00', 'confirmed', '3o3c25e4uium2mgeajnk43bprg', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-07-04T16:00:58.324544+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('e7951189-9e44-4cbc-a232-18f8d99b7b8d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-06T14:00:00+00:00', '2026-07-06T15:00:00+00:00', 'confirmed', 'j08f121atpdlb32r3itkd75j9k', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-07-04T16:01:02.636817+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('cf7ed3f8-7496-4d45-976c-5a6154424df5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-04T18:00:00+00:00', '2026-07-04T19:00:00+00:00', 'confirmed', 'lpr0agkaakl6qadqvrf84c3h28', '{"notes": "NTC clenska karta active", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-04T17:51:57.858355+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('732c9192-83f1-46f9-a6d2-771c9f2bc99f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-05T14:00:00+00:00', '2026-07-05T15:00:00+00:00', 'confirmed', 'rojv7uuuaatv7tpra1qnora7ao', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-4"}', NULL, '2026-07-05T12:47:22.949581+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', NULL),
    ('f9c32395-7af8-4c8d-ae6c-52357662fe52', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-05T11:00:00+00:00', '2026-07-05T12:00:00+00:00', 'confirmed', '1c5b8b5ska63p78nnrv53sjon4', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-05T08:43:07.539531+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('ce99be9a-f048-4cc3-9bcc-c06e9636c0ab', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-05T13:00:00+00:00', '2026-07-05T14:00:00+00:00', 'confirmed', 'ikpd25hr8q8rof2lrdbut7gse4', '{"notes": "dvojhra", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-05T10:59:48.909207+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('ac95e5fb-fbdd-44be-8fe7-c438f5087d41', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-05T15:00:00+00:00', '2026-07-05T16:00:00+00:00', 'confirmed', '4s1fh6la150rvc49r74rf5l64o', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-05T11:14:50.404175+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('4549d0cc-764c-4fe9-bb2f-9248d562ad63', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-05T16:00:00+00:00', '2026-07-05T17:00:00+00:00', 'confirmed', 'koqr860muu667b9jge8qlklcq0', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-05T12:49:26.067139+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('86183e83-ed49-4747-ae76-008c515808c0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', NULL, '2026-07-05T14:00:00+00:00', '2026-07-05T15:00:00+00:00', 'confirmed', 'rtk58bjj48qb30ps53abcanpg4', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-05T13:00:22.815859+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('2fa4d58a-aa44-4781-be90-022c9ac4d8eb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-05T16:00:00+00:00', '2026-07-05T17:00:00+00:00', 'confirmed', '5bfl12jhsdjtcf881bkegc2n48', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-4"}', NULL, '2026-07-05T13:54:55.303441+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', NULL),
    ('6e8cba78-1a00-4f1d-81c0-6d9348358ada', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-05T16:00:00+00:00', '2026-07-05T17:00:00+00:00', 'confirmed', 'efs5d17pa3ru4fco09slco6mgo', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-4"}', NULL, '2026-07-05T13:54:58.140968+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', NULL),
    ('10eb9750-aec8-4899-b7d7-8935b67462ec', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-05T18:00:00+00:00', '2026-07-05T19:00:00+00:00', 'confirmed', 'fldmkl5q65apks86bvppm32f98', '{"notes": "dsklfdsilfjs", "source": "azure-realtime-voice-assistant", "courtId": "badminton-4"}', NULL, '2026-07-05T14:05:34.097653+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', NULL),
    ('1e3ffee8-588b-4692-b1a0-8d6471d1f202', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Michaela Tomčofčíková', '', '2026-07-10T09:00:00+00:00', '2026-07-10T10:00:00+00:00', 'confirmed', '0o2m9lq0j9nn0lvcavinqi6tm8', '{"notes": "Standard booking without membership", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-05T14:13:58.059436+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('d28b8358-d46d-437d-8195-6c830a1bf4fd', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-06T09:00:00+00:00', '2026-07-06T10:00:00+00:00', 'confirmed', 'vok7ubhedejeeil27e77iknei0', '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-05T14:25:09.985172+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('d5ce0980-59fe-4150-bf73-b34be72ed18d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Mokráň', '', '2026-07-06T12:00:00+00:00', '2026-07-06T13:00:00+00:00', 'confirmed', '5vei7rk3ft0ot26a3b0ggklle8', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-05T14:30:33.297494+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('624049ba-0f86-4cb6-8c04-9f3a12827b38', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-06T07:00:00+00:00', '2026-07-06T08:00:00+00:00', 'confirmed', 'k0vsdi01b4b3d83j3lh9qdflqs', '{"notes": "Hkvcgg", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-05T14:34:22.090673+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('fc681828-51c5-4c61-b6da-b7cefebd4e4a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-06T14:00:00+00:00', '2026-07-06T16:00:00+00:00', 'confirmed', '7qoisrl6nvj5s3nu0thd67kfj8', '{"notes": "Digc", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-05T14:36:09.454623+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('5e9f8749-a4cb-4ccd-9fbe-4339aa9f4980', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-07-06T08:00:00+00:00', '2026-07-06T09:00:00+00:00', 'confirmed', 'mfpeg0gg0a4qtec2s9pc2rfdrg', '{"notes": "Standard booking without membership card", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-05T16:10:20.044137+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('7b0fa241-2372-4672-9f58-5c041fc03c20', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Tomčovčík', '', '2026-07-06T16:00:00+00:00', '2026-07-06T17:00:00+00:00', 'confirmed', 'a9m2kjug2i9tqf48cde9risf4o', '{"notes": "Meno s f ako František", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-05T17:33:01.705356+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('63b3a298-58f4-4c4d-a961-06c5d420d968', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-14T08:00:00+00:00', '2026-07-14T09:00:00+00:00', 'cancelled', NULL, '{"courtId": "tennis-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-13T19:38:04.96348+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('3d032658-553c-4ac2-9edc-bc08501dab31', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-14T09:00:00+00:00', '2026-07-14T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-13T19:38:47.931281+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('a1dc35cb-af65-4583-96ab-a03f8877daef', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-04T16:00:00+00:00', '2026-07-04T17:00:00+00:00', 'confirmed', 'cflffeksmnpfb2me8gdqnro9fo', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-7"}', NULL, '2026-07-04T15:16:39.905329+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-7', NULL),
    ('32efe7dc-d604-464c-83a9-845809851102', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', NULL, '2026-07-04T16:00:00+00:00', '2026-07-04T17:00:00+00:00', 'confirmed', 'f2g8fnqomgus3qqovgr06qbrt8', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-6"}', NULL, '2026-07-04T15:17:57.434671+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'tennis', 'tennis-6', NULL),
    ('e83be14c-8380-49d0-9ef4-cc2837a43683', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', NULL, '2026-07-04T18:00:00+00:00', '2026-07-04T19:00:00+00:00', 'confirmed', 'tdi9r3gsoq0d1nvu7t0722mdf4', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-4"}', NULL, '2026-07-04T15:18:04.042494+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'tennis', 'tennis-4', NULL),
    ('d7754286-e07b-48e6-b83d-79e141c6f20d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Tomčovčík', '', '2026-07-06T16:00:00+00:00', '2026-07-06T17:00:00+00:00', 'confirmed', 'b8huanaok5lp4lkp22go1oe0ho', '{"notes": "Meno s f ako František", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-05T17:33:17.899407+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('29b5c4d3-0e53-4634-82e0-e6369aa70277', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-07T14:00:00+00:00', '2026-07-07T15:00:00+00:00', 'confirmed', 'lmi2h831bpimb5qd2pkcidk8hs', '{"notes": "Zákazník nemá členskú kartu NTC.", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-07-05T18:29:02.341766+00:00', NULL, NULL, 'tennis-clay', 'tennis-clay-1', NULL),
    ('463af46f-cca9-499f-9b12-999e25fd91b4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-09T15:00:00+00:00', '2026-07-09T16:00:00+00:00', 'confirmed', 'q431ijvcdadkgrg6u67ql9q19k', '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-05T18:37:20.623124+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('5005e516-3e1f-43ae-a686-353448d14980', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Peter', '', '2026-07-06T08:00:00+00:00', '2026-07-06T09:00:00+00:00', 'confirmed', '1mjt275rh9of237cj8e2102pqs', '{"notes": "Členská karta NTC.", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-07-05T19:15:23.799072+00:00', NULL, NULL, 'tennis', 'tennis-2', NULL),
    ('34f57018-bc79-487e-96a9-2ce34391d0b4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-07T11:00:00+00:00', '2026-07-07T12:00:00+00:00', 'confirmed', '0p5soj96ou5ggdaf4r7u9tljjo', '{"notes": "Standard rate, no membership card", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-06T16:01:46.794912+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('25a467bf-96e8-4821-8a81-749fd325630d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-07T13:00:00+00:00', '2026-07-07T14:00:00+00:00', 'confirmed', 'fmhbjnhgjeu0f4r8i3hq8r5ouk', '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-06T16:10:01.616569+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('47e33ef5-f9e4-4025-8036-0b44defb60b5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Lídiia Kečuchová', '', '2026-07-16T16:00:00+00:00', '2026-07-16T17:00:00+00:00', 'confirmed', 'ekomuho23qd3bll03o7vivdn6s', '{"notes": "Bez členskej karty.", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-06T16:11:56.225219+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('de928874-03db-45e6-94ac-287bbb89efaf', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-06T17:00:00+00:00', '2026-07-06T18:00:00+00:00', 'confirmed', 'les4a1pfrt5n1dv3iuvuvdvcao', '{"notes": "Zákazník má členskú kartu NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-06T16:17:40.841683+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('04aaf348-b9ba-4882-b1c7-05b3b304f20b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-07T14:00:00+00:00', '2026-07-07T15:00:00+00:00', 'confirmed', 'jaguc7okf13eifl1400oada4vo', '{"notes": "Zákazník má členskú kartu NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-07T07:35:48.11532+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('e53cb060-c93f-433e-b338-33063522eea9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-07T15:00:00+00:00', '2026-07-07T16:00:00+00:00', 'confirmed', 'aclge1ha3ntl54tabkdlhme7nk', '{"notes": "Zákazník má aktívnu NTC členskú kartu.", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-07-07T07:39:56.945237+00:00', NULL, NULL, 'tennis', 'tennis-2', NULL),
    ('a77b19b3-84ed-4001-b123-b014fa39cbf9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kličinský', '', '2026-07-07T13:00:00+00:00', '2026-07-07T14:00:00+00:00', 'confirmed', 'lorhnqg2hls3as6vgchk1v0j3o', '{"notes": "Držiteľ členskej karty NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-07-07T07:47:00.84489+00:00', NULL, NULL, 'tennis-clay', 'tennis-clay-1', NULL),
    ('34ffb1e3-3325-4ce6-b87a-4d1e508bba4c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-07T12:00:00+00:00', '2026-07-07T14:00:00+00:00', 'confirmed', 'j0ju56kbakd9pm9dmr85jh8tn8', '{"notes": "Prideme styria", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-07T08:24:10.706491+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('2772f264-e347-43f3-b706-9c9eaede8b72', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-14T06:00:00+00:00', '2026-07-14T07:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-13T19:48:39.478569+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('d5aae286-f3ae-4073-8b28-c22d11c43149', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-07T16:00:00+00:00', '2026-07-07T17:00:00+00:00', 'confirmed', 'uu7c0m1u58mj2cvnsuego29ib0', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-07T08:34:06.677257+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('d40b19ff-fc48-42f2-8a93-1dddab8df543', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-07T17:00:00+00:00', '2026-07-07T18:00:00+00:00', 'confirmed', 'klpjklr0ohd1foq5lagqn27kps', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-07T08:36:17.351359+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('aa04d255-5ca6-41e3-94a4-8d2c37219a8a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-07T16:00:00+00:00', '2026-07-07T17:00:00+00:00', 'confirmed', 'ue2qkaoir1knvml0vrpscbjhpo', '{"notes": "djdkfjsdlôkfj", "source": "azure-realtime-voice-assistant", "courtId": "badminton-3"}', NULL, '2026-07-07T09:01:26.848613+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', NULL),
    ('f8ec2528-8c22-494a-b2c2-124c0e928474', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-08T16:00:00+00:00', '2026-07-08T17:00:00+00:00', 'confirmed', '7fcdu995jn35alcenalede0j0o', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-07T09:02:43.221732+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('dfbe0cf3-08c4-4fc6-b794-72cc5fdcad24', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartková', '', '2026-07-08T15:00:00+00:00', '2026-07-08T16:00:00+00:00', 'confirmed', 'cs3es3g3adv75i5b5gj67k2hsk', '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-07T09:27:34.438478+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('6f6155e4-a016-4215-b3e9-03717c7d63fc', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-07-08T08:00:00+00:00', '2026-07-08T09:00:00+00:00', 'confirmed', 'loib8ohmoo3f3ovg4bp8o8vru0', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-07T09:58:10.829022+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('8892a8c0-3211-4a28-878d-0bb1b3308897', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-07-08T08:00:00+00:00', '2026-07-08T08:30:00+00:00', 'confirmed', 'grjssu523u9nopaj13n7pbhh88', '{"notes": "Rezervácia cez hlasovú asistentku, členská karta áno", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-07-07T11:14:55.460617+00:00', NULL, NULL, 'tennis', 'tennis-2', NULL),
    ('18cb647a-b4d9-46c4-8b3d-1f28d0ddb095', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-08T13:00:00+00:00', '2026-07-08T14:00:00+00:00', 'confirmed', 'ap8vaq40biai8t0ccko88ro6sk', '{"notes": "mkjkj", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-08T09:58:44.184852+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('fd64f4a5-2a7b-4ff1-b4e0-f38a50714bbe', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '+421915962068', '2026-07-08T14:00:00+00:00', '2026-07-08T15:00:00+00:00', 'confirmed', 'idmegbsvsljoru3pi66lm2mf68', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-08T10:43:19.364474+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('76c6b681-cf4d-4393-b81c-217d71331895', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-09T13:00:00+00:00', '2026-07-09T14:00:00+00:00', 'confirmed', NULL, '{"notes": "NTC členská karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-08T19:21:48.997537+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('5243d394-fdbb-4eff-8f21-b1fb8460363e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '+421915962068', '2026-07-08T14:00:00+00:00', '2026-07-08T15:00:00+00:00', 'confirmed', NULL, '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-08T11:29:46.233011+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('e740d5b4-d5fc-4d08-8336-5932b62cad56', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T17:00:00+00:00', '2026-07-16T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T15:59:17.850637+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('4b87eed3-8643-4a6d-9619-2c21ee1b60d5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-19T09:00:00+00:00', '2026-07-19T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "cartesia-voice-assistant", "notes": ""}', NULL, '2026-07-19T07:26:48.372593+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('2133cbda-a034-476f-8941-20a7ff2aaa04', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-06T07:00:00+00:00', '2026-07-06T08:00:00+00:00', 'confirmed', '3p2b35dm7td4s5aafk0ugoomgc', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-05T20:27:05.73156+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('6594f772-0fdc-451e-a16f-5b9a038d9a4f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-06T09:00:00+00:00', '2026-07-06T10:00:00+00:00', 'confirmed', 'hm47gnkuvti9hp1i6s28o1qelg', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-05T20:27:09.076557+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('b8e6016b-cdc0-4f00-8421-32e0656ffa93', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-06T14:00:00+00:00', '2026-07-06T15:00:00+00:00', 'confirmed', 'etin9rggoj8gn0va55m1ogiv70', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-05T20:37:21.127296+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('9e4bbf30-a32a-452d-9c91-2bc6c61ec943', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-06T05:00:00+00:00', '2026-07-06T07:00:00+00:00', 'confirmed', '8rj0lbcrje27olu4la0ohe6d84', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-05T20:38:08.872927+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('276e2df8-87c4-4eba-8e8a-5ff8fdf16b13', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-06T14:00:00+00:00', '2026-07-06T15:00:00+00:00', 'confirmed', 'oi1jjh5bsbe3fpt4hach5tsqu4', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-05T20:38:35.490809+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('c861f276-93ff-4292-98b2-e4b72c5f48a9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', NULL, '2026-07-06T14:00:00+00:00', '2026-07-06T15:00:00+00:00', 'confirmed', '95o50g0hirp7eau7q8q55lhes4', '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-4"}', NULL, '2026-07-05T20:40:08.603713+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-4', NULL),
    ('1bc5b41c-fd28-4a63-8217-7dfc805238dc', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-07T12:00:00+00:00', '2026-07-07T13:00:00+00:00', 'confirmed', '66k7sh5bo7pf721oinhge5rve4', '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-06T15:32:00.871374+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('69b1f669-d43a-4158-b603-ba2cdd7c0e78', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-07T13:00:00+00:00', '2026-07-07T14:00:00+00:00', 'confirmed', '2ovj3ne2jkt8b6l4nucj5gsj4s', '{"notes": "Členská karta NTC, zľava dve eurá na hodinu, dve hodiny parkovania zadarmo.", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-06T15:44:04.86677+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('1f21e13e-63df-4ed8-a65a-7ca545f76437', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-07T09:00:00+00:00', '2026-07-07T10:00:00+00:00', 'confirmed', '0ef3vb8321nadsaglsqj837lak', '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-06T15:48:08.676425+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('241719fd-1d9b-484d-835f-492db723ca5b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-07T10:00:00+00:00', '2026-07-07T11:00:00+00:00', 'confirmed', '9bjej16tivfr3rfqktnhthf7jg', '{"notes": "Na chate", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-06T15:56:24.847918+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('9cf4c378-0a2c-4486-a5f0-48ea3ef3011d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '+421915962068', '2026-07-08T13:00:00+00:00', '2026-07-08T14:00:00+00:00', 'confirmed', NULL, '{"notes": "Držiteľ členskej karty NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-08T11:47:02.677847+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('15281427-050c-4df0-b0ba-72c3c673fd77', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '+421915962068', '2026-07-08T17:00:00+00:00', '2026-07-08T18:00:00+00:00', 'confirmed', NULL, '{"notes": "NTC členská karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-08T12:07:42.878906+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('73bae17c-0cef-463a-8595-ae6206d44033', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '+421915962068', '2026-07-08T16:00:00+00:00', '2026-07-08T17:00:00+00:00', 'confirmed', NULL, '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-08T12:43:07.973197+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('4f572407-0372-41c4-8bed-a27e5e260006', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '+421915962068', '2026-07-08T17:00:00+00:00', '2026-07-08T18:00:00+00:00', 'confirmed', NULL, '{"notes": "Rezervácia cez hlasovú asistentku, členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-08T13:20:34.93811+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('a766c908-de22-4893-96db-82315b5f96aa', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '+421915962068', '2026-07-08T18:00:00+00:00', '2026-07-08T19:00:00+00:00', 'confirmed', NULL, '{"notes": "Rezervácia cez hlasovú asistentku", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-08T15:03:39.669956+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('fdba4838-ba0d-4940-926a-9fea6a2e37fb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', '+421915962068', '2026-07-09T08:00:00+00:00', '2026-07-09T09:00:00+00:00', 'confirmed', NULL, '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-08T19:17:39.144882+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('0fbfff4a-87d6-47bf-883a-ded94f26a678', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421905012054', '2026-07-09T10:00:00+00:00', '2026-07-09T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Rezervácia cez hlasovú asistentku s členskou kartou NTC.", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-08T19:21:25.53748+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('be663886-57e0-415b-8fe9-c813a53ecec8', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-09T10:00:00+00:00', '2026-07-09T11:00:00+00:00', 'confirmed', NULL, '{"notes": "NTC clenska karta", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-08T19:29:07.969845+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('06407fd7-9def-4abe-aacb-68c1b58ecc67', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-09T12:00:00+00:00', '2026-07-09T13:00:00+00:00', 'confirmed', NULL, '{"notes": "NTC členská karta", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-09T06:33:03.412168+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('b9bf04b8-9208-47c1-b011-7086f3ad6afa', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-09T17:00:00+00:00', '2026-07-09T18:00:00+00:00', 'confirmed', NULL, '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-09T14:38:23.83875+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('caed22cd-a0e1-40f3-b140-315f25ee1708', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-10T15:00:00+00:00', '2026-07-10T16:00:00+00:00', 'confirmed', NULL, '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-07-09T14:38:32.910322+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis-clay', 'tennis-clay-1', NULL),
    ('9304e120-d6da-4618-bd11-cf7930e1b5bf', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-09T16:00:00+00:00', '2026-07-09T17:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-09T14:49:46.814977+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('72a999f1-fa7f-4aef-be26-9a1b6b4cd6d9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-14T15:00:00+00:00', '2026-07-14T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "cartesia-voice-assistant", "notes": "Členstvo NTC"}', NULL, '2026-07-14T09:04:01.306527+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('19764fbd-3b63-4586-85af-b76f5c72dead', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T18:00:00+00:00', '2026-07-16T19:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T16:00:28.039279+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('a6fe3c0d-ddd1-4538-8903-372d32e5f27e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-19T09:00:00+00:00', '2026-07-19T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "cartesia-voice-assistant", "notes": "držiteľ členskej karty NTC"}', NULL, '2026-07-19T07:30:13.81041+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('36bf1def-5611-437e-acf9-8fe4e4a4cdf1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-08T12:00:00+00:00', '2026-08-08T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia bez \u010dlenskej karty."}', NULL, '2026-07-23T10:48:00.714233+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('a8ebcaf4-614d-4629-b6af-feb7d2b3be43', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-27T13:00:00+00:00', '2026-07-27T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-07-27T11:38:11.361821+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('4f3c0c38-cfe9-4a05-b34c-3ab0986d18f6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-27T13:00:00+00:00', '2026-07-27T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-07-27T11:38:12.347652+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('1b2389d6-c30d-42e0-b833-60f1da6c1aa9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-28T09:00:00+00:00', '2026-07-28T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-27T14:59:54.11271+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('3a10ea30-5966-4b8a-aa95-1439e34732f1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartkova', '', '2026-07-08T15:00:00+00:00', '2026-07-08T16:00:00+00:00', 'confirmed', '0rddpujlltf3k22onv8hbbc9d0', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-07T13:43:14.955276+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('d5e7686a-d468-40c6-ba8a-9c78040fa086', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-08T11:00:00+00:00', '2026-07-08T12:00:00+00:00', 'confirmed', 'ube8k7tb3fcu21dmgm1v8c2vo4', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-07T13:48:04.272486+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('efaaa54d-516e-4526-9fd9-b9eeb95b4caf', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Test', NULL, '2026-07-10T10:00:00+00:00', '2026-07-10T12:00:00+00:00', 'confirmed', 'cdtpun84fjfoc033n78r4p6dqk', '{"notes": "dvopjhrs", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-07-07T13:55:12.211337+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-2', NULL),
    ('689c37b6-9d65-48e9-9a72-80af771acb05', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', NULL, '2026-07-07T17:00:00+00:00', '2026-07-07T18:00:00+00:00', 'confirmed', 'ttbf6q05a2uve2qcf2s34fsljs', '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-3"}', NULL, '2026-07-07T15:24:53.472096+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-3', NULL),
    ('403a8379-ec93-43b1-b569-10ca8b5e2ba2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-07-07T18:00:00+00:00', '2026-07-07T19:00:00+00:00', 'confirmed', 't2ul2e7091uker3u62lhgereb0', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-07T15:26:22.062437+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('549a4367-ef60-4f05-8d8d-1860abec9bda', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-08T15:00:00+00:00', '2026-07-08T16:00:00+00:00', 'confirmed', '70ids6m4kvg5qfg3t6a5rprskc', '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-08T09:45:28.731353+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('ce32d22b-4f16-4bfd-8535-8e40fb7e8324', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '+421915962068', '2026-07-08T12:00:00+00:00', '2026-07-08T13:00:00+00:00', 'confirmed', 'uqfak3bquvioi1o1qi1huj4760', '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-08T09:57:12.798251+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('1258b8e7-30f1-4648-83fd-e2f621252a3b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-09T16:00:00+00:00', '2026-07-09T17:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-09T14:49:47.521897+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('d1ea3584-c6dc-48dd-9ace-6529b6b68248', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-09T16:00:00+00:00', '2026-07-09T17:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-09T14:49:55.372626+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('10da9a48-77f2-4f73-886c-668b1df1dea7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-09T15:00:00+00:00', '2026-07-09T16:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-09T14:50:17.027282+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('f704dfce-bfe7-41a9-b908-a2f18ed47cb4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-09T17:00:00+00:00', '2026-07-09T18:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-09T14:50:33.912982+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('68d82d33-cae4-4e22-8729-0cdaedbc55ad', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-22T13:00:00+00:00', '2026-07-22T14:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-3"}', NULL, '2026-07-09T16:04:28.573904+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-3', NULL),
    ('4a37561d-5162-4d50-b5a1-c042ee4b7a2c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-22T10:00:00+00:00', '2026-07-22T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-09T16:04:32.143999+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('1e930287-9e93-4e40-866f-e2725759ae5f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-22T16:00:00+00:00', '2026-07-22T17:00:00+00:00', 'confirmed', NULL, '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-07-09T16:04:38.327571+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-2', NULL),
    ('061baca1-7653-4e1a-9ed5-03fa41248d38', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-23T14:00:00+00:00', '2026-07-23T15:00:00+00:00', 'confirmed', NULL, '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-09T16:04:41.839063+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('3f47f15c-d498-4151-9f37-9c86a4030e8b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-23T08:00:00+00:00', '2026-07-23T09:00:00+00:00', 'confirmed', NULL, '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-09T16:04:44.663826+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('d0e87dc7-95ff-40a0-b764-5ea39b6fbe11', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin Údržba', NULL, '2026-07-10T05:00:00+00:00', '2026-07-10T07:00:00+00:00', 'blocked', NULL, '{"notes": "Údržba", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-09T16:09:06.721062+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', NULL),
    ('5b98dba3-6f8d-4a73-b1fa-b9e94a7e61c5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-13T09:00:00+00:00', '2026-07-13T10:00:00+00:00', 'confirmed', NULL, '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-12T17:11:01.157692+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('8a4d9191-75bd-4d35-8545-2b63ccb37057', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', NULL, '2026-07-14T06:00:00+00:00', '2026-07-14T07:00:00+00:00', 'cancelled', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-09T16:41:32.42263+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', NULL),
    ('d60b3efc-9727-4d25-93df-0c86a0256fba', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', NULL, '2026-07-14T06:00:00+00:00', '2026-07-14T07:00:00+00:00', 'confirmed', NULL, '{"notes": "Pista", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-09T16:41:44.765776+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', NULL),
    ('2f6509a4-6501-41b7-b04e-3199b5a7d6b1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', NULL, '2026-07-09T17:00:00+00:00', '2026-07-09T18:00:00+00:00', 'confirmed', NULL, '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-07-09T16:44:44.59391+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'tennis', 'tennis-2', NULL),
    ('7746d175-4272-4caa-b98a-3ea6c6ff32c6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', NULL, '2026-07-09T18:00:00+00:00', '2026-07-09T19:00:00+00:00', 'confirmed', NULL, '{"notes": "Tenis", "source": "azure-realtime-voice-assistant", "courtId": "tennis-2"}', NULL, '2026-07-09T16:44:46.716441+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'tennis', 'tennis-2', NULL),
    ('da2a931f-9142-44f8-8fd5-354f8f4bab5a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin Údržba', NULL, '2026-07-15T05:00:00+00:00', '2026-07-15T13:00:00+00:00', 'blocked', NULL, '{"notes": "Údržba", "source": "azure-realtime-voice-assistant", "courtId": "badminton-3"}', NULL, '2026-07-09T17:14:31.525478+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-3', NULL),
    ('cfb1ae55-c5e7-4860-b946-cf7038d4af32', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-12T13:00:00+00:00', '2026-07-12T14:00:00+00:00', 'confirmed', NULL, '{"notes": "Rezervácia cez hlasovú asistentku bez členskej karty.", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-11T08:07:22.134333+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('f1a95feb-9128-4cc3-b457-d7729b967208', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-14T15:00:00+00:00', '2026-07-14T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-14T09:08:22.523797+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('af9c8890-2cf6-4a5a-ba1a-361e1cedd87d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-16T10:00:00+00:00', '2026-07-16T11:00:00+00:00', 'cancelled', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-09T14:51:05.732024+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('0d887cbd-d637-404f-bc4f-e239d22d54fb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Kaľavský', NULL, '2026-07-16T13:00:00+00:00', '2026-07-16T14:00:00+00:00', 'cancelled', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-09T14:50:58.78919+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('94d6bb44-74c4-4109-82d6-7e97ed910994', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T13:00:00+00:00', '2026-07-16T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T16:15:58.238931+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('3fec8937-3405-47ca-bb94-fc292c0ed128', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-19T17:00:00+00:00', '2026-07-19T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "cartesia-voice-assistant", "notes": "členstvo"}', NULL, '2026-07-19T10:29:55.568457+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('9d86c6d5-0ecb-4899-94ec-66e5b1fee28b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-24T13:00:00+00:00', '2026-07-24T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "\u010clensk\u00e1 karta NTC"}', NULL, '2026-07-23T11:00:46.573622+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('a63c2232-f208-44c5-ba37-c73177cc2551', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-27T17:00:00+00:00', '2026-07-27T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-27T15:58:24.87384+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('f445155a-a621-45d3-96d5-6c1f79a7649a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-11T15:00:00+00:00', '2026-07-11T16:00:00+00:00', 'confirmed', NULL, '{"notes": "sddgreg", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-11T09:28:53.443886+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('15f99684-ef83-4995-ad49-88b852bdcc96', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-11T10:00:00+00:00', '2026-07-11T11:00:00+00:00', 'confirmed', NULL, '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-11T09:30:58.413717+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('95cc2222-2916-43d7-8709-683da62f6da2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-12T09:00:00+00:00', '2026-07-12T10:00:00+00:00', 'confirmed', NULL, '{"notes": "Rezervácia s členskou kartou NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-11T16:46:21.115557+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('0ae1243d-eac6-4543-af05-eeffbb5f71d7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-12T12:00:00+00:00', '2026-07-12T13:00:00+00:00', 'cancelled', NULL, '{"notes": "nova rezervacia", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-12T07:41:34.451635+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('8709bd26-2bf8-4950-866f-4ade6b0071f0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Joško Strelec', '', '2026-07-13T08:00:00+00:00', '2026-07-13T09:00:00+00:00', 'confirmed', NULL, '{"notes": "NTC členská karta", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-12T10:33:46.451468+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('13ba237b-cc7a-402d-aebc-16a2ff677b09', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T15:00:00+00:00', '2026-07-13T16:00:00+00:00', 'confirmed', NULL, '{"notes": "členstvo NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-12T10:49:54.960354+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('ed75263c-fab7-4c81-855b-01b32297655e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-07-13T13:00:00+00:00', '2026-07-13T14:00:00+00:00', 'confirmed', NULL, '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-07-12T11:32:46.977746+00:00', NULL, NULL, 'tennis-clay', 'tennis-clay-1', NULL),
    ('c6b64e99-9131-4e60-a779-34451d593a56', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-12T14:00:00+00:00', '2026-07-12T15:00:00+00:00', 'confirmed', NULL, '{"notes": "Členstvo NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-12T11:33:51.296908+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('be9b4fb9-9db8-48be-ae72-6f28200fec9b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-12T14:00:00+00:00', '2026-07-12T15:00:00+00:00', 'confirmed', NULL, '{"notes": "Členstvo NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-12T11:46:00.351233+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('875ca7de-de6a-4e88-98b3-8522d635220d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T16:00:00+00:00', '2026-07-13T17:00:00+00:00', 'confirmed', NULL, '{"notes": "Členská karta NTC, parkovanie zdarma 2h", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-12T11:56:22.203023+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('097cecda-3dac-4013-a93b-2b0f7739389d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-12T12:00:00+00:00', '2026-07-12T13:00:00+00:00', 'confirmed', NULL, '{"notes": "Členstvo NTC, parkovanie zdarma", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-12T11:58:44.876146+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('d87bb7b2-66b3-4053-a170-2e3420e3d379', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T16:00:00+00:00', '2026-07-13T17:00:00+00:00', 'confirmed', NULL, '{"notes": "Členská karta NTC, parkovanie zdarma", "source": "azure-realtime-voice-assistant", "courtId": "squash-2"}', NULL, '2026-07-12T12:00:26.497804+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-2', NULL),
    ('37c0f04f-75c5-423a-9954-42485dbc7aa2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T07:00:00+00:00', '2026-07-13T08:00:00+00:00', 'confirmed', NULL, '{"notes": "Členstvo NTC, dve hodiny parkovania zdarma", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-12T12:11:21.03722+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('af72e69c-2f95-4124-b2dc-66413caa6c30', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-07-12T15:00:00+00:00', '2026-07-12T16:00:00+00:00', 'confirmed', NULL, '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-12T12:19:42.734148+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('ab20ca42-9e6d-403a-8307-e8028575adfc', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-13T16:00:00+00:00', '2026-07-13T17:00:00+00:00', 'confirmed', NULL, '{"notes": "Členstvo NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-12T13:28:47.253561+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('0c22d281-fe74-4fc0-9cb9-e74030fe396f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-12T15:00:00+00:00', '2026-07-12T16:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-07-12T13:31:33.012585+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('d743579e-bf77-45fa-9f37-1fc9e350cae0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-13T12:00:00+00:00', '2026-07-13T13:00:00+00:00', 'confirmed', NULL, '{"notes": "Členstvo NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-12T17:02:32.635873+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('87efc8af-5c33-439f-aebf-b16af96f66ac', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-13T06:00:00+00:00', '2026-07-13T07:00:00+00:00', 'confirmed', NULL, '{"notes": "Členstvo NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-12T18:07:54.319971+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('25a1e065-21e4-4f4b-ac8a-931548ddf0fb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T13:00:00+00:00', '2026-07-13T14:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-12T18:54:22.998892+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('467ccb68-c0fb-4a75-9a78-24fa4a92523d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T13:00:00+00:00', '2026-07-13T14:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-2"}', NULL, '2026-07-12T19:13:24.905364+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-2', NULL),
    ('c9a64365-a144-42ce-b7ce-46fa8b8dc636', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T13:00:00+00:00', '2026-07-13T14:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-3"}', NULL, '2026-07-12T19:17:01.738422+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-3', NULL),
    ('ac27f19a-d3a5-402c-a1d5-7b0e28bf02f2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T13:00:00+00:00', '2026-07-13T14:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-4"}', NULL, '2026-07-12T19:22:18.600193+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-4', NULL),
    ('dab33968-c657-45b3-958d-f5dd8131dcf7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T14:00:00+00:00', '2026-07-13T15:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-3"}', NULL, '2026-07-12T19:30:55.639775+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-3', NULL),
    ('cd9c731e-a431-4c07-9d48-47b9f91cd33b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-15T13:00:00+00:00', '2026-07-15T14:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-12T19:34:25.357448+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('3638f97f-6abd-4ae9-af09-72d8163ceac9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T13:00:00+00:00', '2026-07-13T14:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-2"}', NULL, '2026-07-12T19:37:49.719969+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis-clay', 'tennis-clay-2', NULL),
    ('deada37e-117b-488a-a289-547362d47c90', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T13:00:00+00:00', '2026-07-13T14:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-3"}', NULL, '2026-07-12T19:39:26.320144+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis-clay', 'tennis-clay-3', NULL),
    ('71f67f5e-360b-4bc6-8d49-4c4fb9e7f231', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T13:00:00+00:00', '2026-07-13T14:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-4"}', NULL, '2026-07-12T19:53:13.245287+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis-clay', 'tennis-clay-4', NULL),
    ('eafb231f-2fcb-42e3-8ffc-52a3f8156b1a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T15:00:00+00:00', '2026-07-13T16:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-07-12T20:06:05.959303+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis-clay', 'tennis-clay-1', NULL),
    ('db4fcc9b-5994-4b22-9c7b-6f6f59b8f857', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-14T16:00:00+00:00', '2026-07-14T17:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-12T20:23:10.114654+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('eafe0a5c-41b7-4117-816a-e87c97b4a911', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T14:00:00+00:00', '2026-07-13T15:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-4"}', NULL, '2026-07-12T20:23:45.270706+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-4', NULL),
    ('ad9ac2cc-5aa2-4a30-bb89-88de780af1be', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T14:00:00+00:00', '2026-07-13T15:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-12T19:25:04.361409+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('842e4f31-ee27-4037-b6fa-8fec62d76e46', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peto Ukážka', '+421905012054', '2026-07-13T14:00:00+00:00', '2026-07-13T15:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-2"}', NULL, '2026-07-12T19:28:46.592941+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-2', NULL),
    ('9b4e0a75-fe83-4529-b0a3-1a11c0363668', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T17:00:00+00:00', '2026-07-13T18:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-12T20:24:38.836436+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('630a0bf1-370b-4547-8445-1cef29dea7cf', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T16:00:00+00:00', '2026-07-13T17:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-3"}', NULL, '2026-07-12T20:26:19.9894+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-3', NULL),
    ('0f2fcda1-40bc-4984-bd97-72f8bd76c159', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-13T08:00:00+00:00', '2026-07-13T09:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-12T20:35:15.687931+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('fbec03e5-0c74-4bc2-8ca5-09659b1cc405', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T12:00:00+00:00', '2026-07-13T13:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-12T20:41:14.165424+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('a3f80342-cd6c-4eec-8993-a79d4d80073a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-15T15:00:00+00:00', '2026-07-15T16:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-07-12T20:44:00.947619+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis-clay', 'tennis-clay-1', NULL),
    ('f6e0dcf1-4100-4922-9818-3fdf63536046', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T13:00:00+00:00', '2026-07-13T14:00:00+00:00', 'cancelled', NULL, '{"notes": "Rezervácia na tenis na tvrdom povrchu.", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-13T06:07:41.95881+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('adb0f807-8658-470f-913d-f4729e9a9fcf', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Držiteľ členskej karty NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-13T06:28:51.920596+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('00881a2e-8509-4e10-97cd-76467b075d79', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-13T06:29:16.711246+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('0902caed-3919-42ba-844e-acd59545f217', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-3"}', NULL, '2026-07-13T06:29:19.675602+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', NULL),
    ('4160f8cb-4dd0-4abf-97d3-a2fbb4a17942', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-4"}', NULL, '2026-07-13T06:29:22.945786+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', NULL),
    ('28a7dc0e-bee2-4cc0-92ed-ed6478f115a2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-5"}', NULL, '2026-07-13T06:29:25.778678+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-5', NULL),
    ('550b555d-bf7d-4a61-a385-5c57d1a7af58', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-6"}', NULL, '2026-07-13T06:29:29.106946+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-6', NULL),
    ('ba6e56cf-b784-4f17-8069-63aecf538398', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-7"}', NULL, '2026-07-13T06:29:32.152497+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-7', NULL),
    ('34cb233b-6ed9-43f8-a535-3dff0f78863b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-8"}', NULL, '2026-07-13T06:29:37.043055+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-8', NULL),
    ('419ff419-97bb-4449-8a01-11166c25eaf4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-9"}', NULL, '2026-07-13T06:29:40.348362+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-9', NULL),
    ('3964df7c-d06f-44ee-bc2a-a1e20f8bfaa8', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Bedminton", "source": "azure-realtime-voice-assistant", "courtId": "badminton-10"}', NULL, '2026-07-13T06:29:43.882191+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-10', NULL),
    ('ada2f6d7-d1ab-4820-9e72-9d259cc6ab30', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "Členstvo", "source": "azure-realtime-voice-assistant", "courtId": "badminton-11"}', NULL, '2026-07-13T06:30:23.888473+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-11', NULL),
    ('4f43b1e6-90e0-4bef-90b1-e268c06604b2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T07:10:00+00:00', '2026-07-13T08:10:00+00:00', 'cancelled', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-2"}', NULL, '2026-07-13T05:58:49.448209+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-2', NULL),
    ('0e45f80a-d8f7-433c-8982-9c04d3c4ec59', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T08:00:00+00:00', '2026-07-13T09:00:00+00:00', 'cancelled', NULL, '{"notes": "NTC Membership Card holder", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-13T06:00:44.831258+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('e1a602d7-dd23-41e1-8a1a-34fe9dab3683', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T12:00:00+00:00', '2026-07-13T13:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-2"}', NULL, '2026-07-13T06:45:11.595655+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-2', NULL),
    ('4458ece6-bcb7-4dc3-9e93-e060db111e7d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-13T11:00:00+00:00', '2026-07-13T12:00:00+00:00', 'confirmed', NULL, '{"notes": "Členská karta NTC", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-13T06:53:17.880136+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('3a009b7f-28bc-44a7-ac6c-d94267f902b8', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T09:00:00+00:00', '2026-07-13T10:00:00+00:00', 'cancelled', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-13T06:04:19.332756+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('ddf3024b-db0a-4b22-9dff-7fc7e93a9ec1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'cancelled', NULL, '{"notes": "Rezervácia na dnešok o dvanástej hodine.", "source": "azure-realtime-voice-assistant", "courtId": "badminton-12"}', NULL, '2026-07-13T06:33:50.03754+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-12', NULL),
    ('58477306-01d8-498d-a758-3e65012f0902', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-13T15:00:00+00:00', '2026-07-13T16:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-13T07:01:26.582642+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('6ff63784-3441-44f4-b459-b3298ad410e7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T12:00:00+00:00', '2026-07-13T13:00:00+00:00', 'cancelled', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-07-13T06:52:38.43788+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('0a128190-ea74-4fd6-b0e1-e9c58a95458e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-15T14:00:00+00:00', '2026-07-15T15:00:00+00:00', 'cancelled', NULL, '{"courtId": "squash-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-14T09:36:40.268508+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('30aebdba-52f1-4796-939a-9daddf86bdc3', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T16:00:00+00:00', '2026-07-16T17:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-2", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T16:17:26.037874+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-2', NULL),
    ('4f33bcde-cf29-4f12-a8e1-13037faab3af', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-19T17:00:00+00:00', '2026-07-19T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "cartesia-voice-assistant", "notes": ""}', NULL, '2026-07-19T16:10:14.386752+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('126a15a5-6719-4b72-90d7-2c4f8a95ea37', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-24T15:00:00+00:00', '2026-07-24T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "\u010clensk\u00e1 karta NTC"}', NULL, '2026-07-23T11:03:10.582143+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('aa835e38-ce7a-410f-a07e-5645ed71a187', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-15T15:00:00+00:00', '2026-07-15T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-14T09:37:25.16409+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('61d39145-b9c4-473d-90c0-25f901085bd7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Petrovičová', '', '2026-06-27T09:00:00+00:00', '2026-06-27T10:00:00+00:00', 'confirmed', '7s4ode7vkcg5vr4mtnau78p4us', '{"notes": "Standard booking without membership card", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-24T16:01:36.565549+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('09ee879c-0405-4e34-a4dc-80863b0c7ef6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil', '5555555', '2026-06-21T10:00:00+00:00', '2026-06-21T11:00:00+00:00', 'confirmed', '264t3pc941uf54eup2ajaja75o', '{"notes": "2hra", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-06-20T21:42:08.961714+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('001d0367-2d76-411b-adc3-f26a57cb07f5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '090538383&3', '2026-06-22T14:00:00+00:00', '2026-06-22T15:00:00+00:00', 'confirmed', 'emja6tr5pqpa7tknb0ra3errsc', '{"notes": "Stvorhra", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-06-21T07:16:18.602781+00:00', NULL, NULL, 'tennis-clay', 'tennis-clay-1', NULL),
    ('9f4e779b-3e88-4ea9-b580-10f68a5cb550', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'ffe', 'efewf', '2026-06-22T07:00:00+00:00', '2026-06-22T08:00:00+00:00', 'confirmed', '6m9tcesgtmkk1cjkv10a37t1v4', '{"notes": "fewf", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-06-21T08:31:47.187467+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('d21374f4-ffcd-46a7-a4ef-fa8a04d59d92', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-23T11:00:00+00:00', '2026-06-23T12:00:00+00:00', 'confirmed', NULL, '{"notes": "Standard rate, no membership card", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-06-22T07:49:26.118371+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('4e091227-9673-4b01-bc21-110ae2a3df3d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-23T09:00:00+00:00', '2026-06-23T10:00:00+00:00', 'confirmed', 'dii8np2liermcgdmge4evt4v1k', '{"notes": "Standard booking without membership card", "source": "azure-realtime-voice-assistant", "courtId": "tennis-1"}', NULL, '2026-06-22T07:59:44.036209+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('d28dd314-70bb-4d98-b06d-5693fd99c87e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamka Bartkova', '0905555555', '2026-06-23T07:00:00+00:00', '2026-06-23T08:00:00+00:00', 'confirmed', 'rrg4e4bkk7s9snrj83f9gssv68', '{"notes": "dvojhra", "source": "azure-realtime-voice-assistant", "courtId": "tennis-3"}', NULL, '2026-06-22T09:56:34.187932+00:00', NULL, NULL, 'tennis', 'tennis-3', NULL),
    ('ecd96e2e-46d4-4d5b-a54d-a5763ab16863', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartkova', '', '2026-06-23T14:00:00+00:00', '2026-06-23T15:00:00+00:00', 'confirmed', 'bjafde119tv10ade9fsr9nf50c', '{"notes": "Standard booking without membership card", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-22T09:58:15.354573+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('cdfd0853-70c0-4700-b3a8-ee157393d445', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-06-24T10:00:00+00:00', '2026-06-24T11:00:00+00:00', 'confirmed', 'u3jvlqrjdfkusp73fosa8dnr7s', '{"notes": "Standard booking without membership card", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-06-23T07:42:58.511632+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('e397cab5-04e7-4714-ae59-56199a1957b7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T16:00:00+00:00', '2026-07-16T17:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-3", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T16:40:11.517064+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-3', NULL),
    ('76b21cb1-2835-4b38-a4a7-3e8c7973a87d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-20T09:00:00+00:00', '2026-07-20T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku, \u010dlensk\u00e1 karta NTC"}', NULL, '2026-07-20T08:12:39.453926+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('028ac00d-36ad-499f-a156-1a5dffb6c1ca', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-24T09:00:00+00:00', '2026-07-24T10:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-1","source":"web","notes":"budeme dvaja"}', NULL, '2026-07-23T17:34:39.070558+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('de46345a-1ec3-4b75-bfa1-c27f98648e0c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-28T12:00:00+00:00', '2026-07-28T13:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "\u010clensk\u00e1 karta NTC"}', NULL, '2026-07-27T11:44:21.320589+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('8af2daf1-2016-4b07-a3b7-36d1148f1047', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-27T18:00:00+00:00', '2026-07-27T19:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-27T16:07:58.78892+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('4fb7f8a1-c2aa-4c08-9e1b-f9cc2be9203a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-28T12:00:00+00:00', '2026-07-28T13:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-2","source":"web","notes":"Skuska cancel"}', NULL, '2026-07-28T10:48:27.257821+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('c5bc5763-b442-4cb2-bae3-ea20d183990e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T10:00:00+00:00', '2026-07-29T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-28T18:22:52.163635+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('37c9438d-839f-483a-88a4-3a61111a66a8', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T09:00:00+00:00', '2026-07-29T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T07:34:40.636991+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('9ac39ecd-cf30-47cc-b998-c31c9c585db6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T08:00:00+00:00', '2026-07-30T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-4", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T18:23:34.431868+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', NULL),
    ('d14d97a6-f7c3-4ae9-b04e-3b96687f37d6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T07:00:00+00:00', '2026-07-30T08:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T20:53:17.47968+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('2aa75469-9ddd-4bef-a920-f1f436508c41', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-30T18:00:00+00:00', '2026-07-30T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Pozn"}', NULL, '2026-07-30T15:37:09.318527+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('9048acd0-9c14-4879-9621-ae8d5eab93df', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-01T13:00:00+00:00', '2026-08-01T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-08-01T05:53:49.261175+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('9f37d098-0093-4e00-9a35-c4d91b1a8d5a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Vážený zákazník', '+421915962068', '2026-08-01T10:00:00+00:00', '2026-08-01T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-01T08:06:03.426238+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('021ce30f-8207-4e0a-82fb-013938193468', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T09:00:00+00:00', '2026-08-03T10:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-02T12:57:06.043754+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('c75f3a4b-6589-4bd2-a22e-9e8a072a56b4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-15T14:00:00+00:00', '2026-07-15T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T08:43:44.596558+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('71e5c3ae-8d89-4278-ae16-5fc5542428e1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T13:00:00+00:00', '2026-07-16T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-clay-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T16:45:20.552355+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis-clay', 'tennis-clay-1', NULL),
    ('d3c692b9-7b32-4636-b139-1f0531db3b23', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-21T11:00:00+00:00', '2026-07-21T12:00:00+00:00', 'cancelled', NULL, '{"courtId": "squash-1", "source": "cartesia-voice-assistant", "notes": "Členská karta NTC"}', NULL, '2026-07-21T08:13:07.042552+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('909681e4-742a-46fc-8656-a5356d21fb05', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-24T13:00:00+00:00', '2026-07-24T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-24T04:59:30.419851+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('5025ea57-fcba-4f33-8a92-af68aff46362', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-28T10:00:00+00:00', '2026-07-28T11:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-27T11:51:55.334209+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('c8885abf-f023-4b41-b70e-675d9ad85e0b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-28T13:00:00+00:00', '2026-07-28T14:00:00+00:00', 'cancelled', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-27T18:16:15.905801+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('bb956fa7-38d5-4fec-b1ce-fae4316d0575', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-28T15:00:00+00:00', '2026-07-28T16:00:00+00:00', 'cancelled', NULL, '{"courtId": "tennis-clay-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-28T12:59:26.349272+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('9a548006-7688-49db-be13-f18ea418c954', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T11:00:00+00:00', '2026-07-29T12:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-07-28T18:23:52.74916+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('31638131-4971-4ed0-beb4-e5a7ffdeb1b8', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T11:00:00+00:00', '2026-07-29T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T07:38:47.581344+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('8598b61d-cfd4-4a79-bed3-9d2f1096b441', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T11:00:00+00:00', '2026-07-29T12:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-2","source":"web","notes":"Squash"}', NULL, '2026-07-29T07:39:09.579602+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-2', NULL),
    ('c926766b-f2d7-4c27-a40d-3d951b1f4a19', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T11:00:00+00:00', '2026-07-29T12:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-3","source":"web","notes":"Squash"}', NULL, '2026-07-29T07:39:12.667869+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-3', NULL),
    ('8aee9efe-33e0-4560-9be8-b8fba155dc28', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T11:00:00+00:00', '2026-07-29T12:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-4","source":"web","notes":"Squash"}', NULL, '2026-07-29T07:39:15.946172+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-4', NULL),
    ('0b40de82-296c-4b7a-b331-dd615a028e68', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T11:00:00+00:00', '2026-07-29T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T07:40:20.105884+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('28959d5e-be80-4fbb-a1ab-a17f4ffa83f9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T09:00:00+00:00', '2026-07-30T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-07-29T18:33:02.025153+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('0d4e54a5-464e-4791-9b9a-0bbcf0a975a8', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T12:00:00+00:00', '2026-07-30T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-30T07:17:58.148751+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('5d43cb1f-2fd9-4936-8c38-f8a701491f27', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Samuel Bártko', '+421915962068', '2026-08-01T10:00:00+00:00', '2026-08-01T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-08-01T07:25:35.868055+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('3b03d266-c14c-416d-9eda-8b0cfb31341e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Novák', '+421915962068', '2026-08-01T11:00:00+00:00', '2026-08-01T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-01T08:12:01.997444+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('05ed1fde-bd47-492e-abfe-8a4b814f3f8d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T09:00:00+00:00', '2026-08-03T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-02T12:59:03.348573+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('fbae137a-d1c7-4bc2-9317-4c0b9ffd2d52', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T11:00:00+00:00', '2026-08-03T12:00:00+00:00', 'cancelled', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-02T13:07:04.729716+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('3c07cdc4-ac89-4a6a-a3c1-c08f70266c66', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T09:00:00+00:00', '2026-08-03T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-03T06:46:28.969298+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('e8506fca-4dd6-4377-a44a-b057d5d22797', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T13:00:00+00:00', '2026-08-03T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-03T10:46:39.981957+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('fb5f0ec0-8a1a-4096-b1a2-6ee2451881ba', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-03T13:00:00+00:00', '2026-08-03T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-03T10:58:59.222996+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-2', NULL),
    ('ac21f3da-c266-41df-9451-5c497d4a6afb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jozef Mrkvička', '+421915962068', '2026-08-03T14:00:00+00:00', '2026-08-03T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-3", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-03T11:06:51.838705+00:00', NULL, '6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'badminton', 'badminton-3', NULL),
    ('94184d69-6636-4d9c-b42e-c14c31478dd6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jozef Mrkvička', '+421915962068', '2026-08-03T13:00:00+00:00', '2026-08-03T14:00:00+00:00', 'cancelled', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-03T10:54:31.877844+00:00', NULL, '6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'squash', 'squash-1', NULL),
    ('661d921f-fab7-4f3e-b192-405e89e97c33', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T14:00:00+00:00', '2026-07-16T15:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T15:34:52.300694+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('d3172569-2711-4b3b-8a6e-5ec7bba2f181', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-20T08:00:00+00:00', '2026-07-20T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T16:46:51.988223+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('f5355cc5-aa84-46f9-a057-9107c052c114', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-22T08:00:00+00:00', '2026-07-22T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "\u010clensk\u00e1 karta NTC"}', NULL, '2026-07-21T19:21:35.673296+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('c1237dd7-a215-402b-aaf7-87b6e47d6271', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-24T14:00:00+00:00', '2026-07-24T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "\u010clensk\u00e1 karta NTC"}', NULL, '2026-07-24T09:25:03.795128+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('b1b3633a-fbea-48f7-8075-027c85dc31ca', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-28T10:00:00+00:00', '2026-07-28T11:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Kamil Bartko"}', NULL, '2026-07-27T11:54:32.090297+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('2e5c9216-9da3-417c-ae90-c9148cb124a3', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-28T14:00:00+00:00', '2026-07-28T15:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-1","source":"web","notes":"Tenis indoor"}', NULL, '2026-07-27T18:44:20.930294+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('65bdb3d3-383d-4d86-8d7a-5fc3b3567af1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T08:00:00+00:00', '2026-07-29T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-28T15:19:11.584276+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('dc93d9dc-37ad-4117-8344-fd43708d8bf7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T11:00:00+00:00', '2026-07-29T12:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-07-28T18:24:54.68569+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('41c9e912-558b-47f8-96f2-425d114cde7e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T17:00:00+00:00', '2026-07-29T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T14:37:11.267482+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('7d0b3e71-898d-40f6-a260-9640c8e932f2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T08:00:00+00:00', '2026-07-30T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-5", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T18:34:19.081778+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-5', NULL),
    ('c3bccbd4-d47e-42a8-b29b-6e0f05bed5dc', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T12:00:00+00:00', '2026-07-30T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-30T09:48:49.500371+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('438f61f4-6614-4c49-b542-a7cac153c890', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Tomčofčík', '+421907173176', '2026-07-30T18:00:00+00:00', '2026-07-30T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Fucgufcv"}', NULL, '2026-07-30T18:00:22.249534+00:00', NULL, '689f3f3b-5596-4859-bc1a-f26cdfa9ba36', 'badminton', 'badminton-3', NULL),
    ('bd9d1c28-395e-4e16-ac56-fed89a841e61', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '', '2026-08-01T10:00:00+00:00', '2026-08-01T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-01T07:28:58.396438+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('07e41281-b03a-4c8e-a6bb-7432ba0001d2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Tomčofčík', '+421907173176', '2026-08-02T17:00:00+00:00', '2026-08-02T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Ahoj speci rezervacia bez online app terminalu"}', NULL, '2026-08-02T08:56:24.177454+00:00', NULL, '689f3f3b-5596-4859-bc1a-f26cdfa9ba36', 'badminton', 'badminton-2', NULL),
    ('348fd412-c9b2-44cf-a514-e7bfb0a31559', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T06:00:00+00:00', '2026-08-03T07:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-clay-1","source":"web","notes":"Tenis antuka"}', NULL, '2026-08-02T13:00:44.028516+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('acfb58d5-a011-41e3-87c0-475c9d34a402', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T14:00:00+00:00', '2026-08-03T15:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-clay-1","source":"web","notes":"Tenis antuka"}', NULL, '2026-08-02T13:00:47.365384+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('54c81896-d219-4149-91aa-83b4b313f7c9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T17:00:00+00:00', '2026-08-03T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-clay-2","source":"web","notes":"Tenis antuka"}', NULL, '2026-08-02T13:00:49.924759+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-2', NULL),
    ('0d3f78a3-6cca-4fca-94c2-3ad82eb9893e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T12:00:00+00:00', '2026-08-03T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-clay-11","source":"web","notes":"Tenis antuka"}', NULL, '2026-08-02T13:01:02.261899+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-11', NULL),
    ('daed0b10-6680-490e-85a7-4633f265fc15', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T08:00:00+00:00', '2026-08-03T09:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-clay-10","source":"web","notes":"Tenis antuka"}', NULL, '2026-08-02T13:01:05.423117+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-10', NULL),
    ('46443604-0fcc-4e2c-9ccc-7a61f3975877', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T09:00:00+00:00', '2026-08-03T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-clay-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-02T13:03:23.730549+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('0ed5a189-f512-4164-b60a-5a539e219a76', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T11:00:00+00:00', '2026-08-03T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-02T13:08:32.989751+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('efe2b72d-5c50-4132-9386-2a71c77f0907', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T10:00:00+00:00', '2026-08-03T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-03T07:02:46.04073+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('b83ef970-e526-41f1-baa1-02cdb4f7684b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T16:00:00+00:00', '2026-07-16T17:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T15:42:23.967105+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('ffa0849d-6bad-412e-8ec5-7075bfd34ec6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T13:00:00+00:00', '2026-07-16T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T15:43:55.365895+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('747350c5-0008-4096-a24f-70854b1fa12f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T16:00:00+00:00', '2026-07-16T17:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T15:43:57.596447+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('eda76199-9931-49d8-a263-46083190c52f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-15T18:00:00+00:00', '2026-07-15T19:00:00+00:00', 'cancelled', NULL, '{"courtId": "squash-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T16:47:48.897442+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('b0fd2ec1-42f6-4979-9130-c63329e5a543', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-22T13:00:00+00:00', '2026-07-22T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "NTC \u010dlensk\u00e1 karta - z\u013eava dve eur\u00e1 na hodinu"}', NULL, '2026-07-22T10:41:43.981353+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('589a3c11-3ce1-4d4b-a053-17c9fbd48ea9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-26T08:00:00+00:00', '2026-07-26T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "\u010clensk\u00e1 karta NTC"}', NULL, '2026-07-25T14:24:17.759487+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('b64e4344-5c15-429b-bf96-aac68dedd451', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-28T10:00:00+00:00', '2026-07-28T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-27T11:55:28.652583+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('78bbe40c-2215-4dae-be33-db4b1a3b1cbf', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-28T18:00:00+00:00', '2026-07-28T19:00:00+00:00', 'cancelled', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-27T18:56:29.499851+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('5dc3771d-0e44-436c-978f-65883fe5fa3c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-29T14:00:00+00:00', '2026-07-29T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alica"}', NULL, '2026-07-28T16:35:06.495843+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('8caa8794-c18e-429f-b390-07fda7122473', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T08:00:00+00:00', '2026-07-29T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-28T18:27:41.681345+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('756be08a-0e4e-42aa-97d6-fcdf557ff4a2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T08:00:00+00:00', '2026-07-30T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T17:03:53.398279+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('af644285-0cbb-4b92-9938-d1244063cb1c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T08:00:00+00:00', '2026-07-30T09:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-6","source":"web","notes":"Bedminton"}', NULL, '2026-07-29T20:00:49.255448+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-6', NULL),
    ('30693918-9548-476e-baee-34bf94d6d041', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T08:00:00+00:00', '2026-07-30T09:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-7","source":"web","notes":"Bedminton"}', NULL, '2026-07-29T20:00:52.692369+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-7', NULL),
    ('37d2ae97-46dc-490a-b0cc-70f2668e9eb4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T08:00:00+00:00', '2026-07-30T09:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-8","source":"web","notes":"Bedminton"}', NULL, '2026-07-29T20:00:55.369148+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-8', NULL),
    ('b2b82c20-1bf3-47ed-8de3-8cb4c8e56485', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T08:00:00+00:00', '2026-07-30T09:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-9","source":"web","notes":"Bedminton"}', NULL, '2026-07-29T20:00:58.179219+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-9', NULL),
    ('2c9fdf46-d4f9-49ff-809f-ae81552063d4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T08:00:00+00:00', '2026-07-30T09:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-10","source":"web","notes":"Bedminton"}', NULL, '2026-07-29T20:01:01.11645+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-10', NULL),
    ('97c76288-0a09-4fe9-9ed4-963565c396b0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T08:00:00+00:00', '2026-07-30T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-11", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T20:01:48.139224+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-11', NULL),
    ('2d0a813c-9ccc-4ddb-b2ff-953f03b26387', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Baco', '+421915962068', '2026-07-30T12:00:00+00:00', '2026-07-30T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-30T09:56:09.995266+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('f9d9b835-99c3-43a7-b281-eee033a375ea', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '+421915962068', '2026-07-31T10:00:00+00:00', '2026-07-31T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-30T18:05:06.014295+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('43711581-1e5a-40b0-b6f7-5d5603f1bd96', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-08-01T09:00:00+00:00', '2026-08-01T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-01T07:57:55.994796+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('f04f0c6d-bb21-46c7-8510-02bf3babf6cf', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-02T13:00:00+00:00', '2026-08-02T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-02T09:05:18.966672+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('dafb946c-2abc-4409-a750-1700b52cfc5e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T18:00:00+00:00', '2026-08-03T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-3","source":"web","notes":"Tenis indoor"}', NULL, '2026-08-02T13:04:04.158168+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-3', NULL),
    ('17faac0a-3e8c-4f1e-a7e5-850f7980eda7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T07:00:00+00:00', '2026-08-03T08:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-3","source":"web","notes":"Tenis indoor"}', NULL, '2026-08-02T13:04:06.316652+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-3', NULL),
    ('0913e831-57d5-4813-84ba-fedb76f0783c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T10:00:00+00:00', '2026-08-03T11:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-5","source":"web","notes":"Tenis indoor"}', NULL, '2026-08-02T13:04:08.911253+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-5', NULL),
    ('db14bf35-5c84-4d79-b8eb-cb6fbf11716f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T16:00:00+00:00', '2026-08-03T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-1","source":"web","notes":"Tenis indoor"}', NULL, '2026-08-02T13:04:11.245503+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('bb7260c3-a7d0-4435-a08e-3e5bd091125e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T08:00:00+00:00', '2026-08-03T09:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-1","source":"web","notes":"Tenis indoor"}', NULL, '2026-08-02T13:04:14.212845+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('b1bcba4b-70b3-437b-8188-b1f0f563eb35', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T16:00:00+00:00', '2026-08-03T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-6","source":"web","notes":"Tenis indoor"}', NULL, '2026-08-02T13:04:17.394642+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-6', NULL),
    ('594f17a4-6622-46e7-b47a-457d5ae4cbab', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-08-02T18:00:00+00:00', '2026-08-02T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Admin"}', NULL, '2026-08-02T16:22:41.443277+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-3', NULL),
    ('422252a4-563a-405e-967b-dd616bdb6ee0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-08-02T18:00:00+00:00', '2026-08-02T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T16:22:57.636408+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-4', NULL),
    ('6afc2906-abd8-41f2-9966-bd4777cce19f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-02T17:00:00+00:00', '2026-08-02T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T16:23:44.486762+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-4', NULL),
    ('58837046-e988-47db-8bfe-44c1fe9a80c7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T11:00:00+00:00', '2026-08-03T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-03T07:54:47.980533+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('d814e7c6-4a29-4d87-814a-c42715c4caa5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', NULL, '2026-07-16T12:00:00+00:00', '2026-07-16T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-1","source":"web","notes":"Bedminton"}', NULL, '2026-07-15T15:47:19.299835+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('e19cb7f5-db43-4f39-bb23-9243849d1c1c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', NULL, '2026-07-16T12:00:00+00:00', '2026-07-16T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-2","source":"web","notes":"Bedminton"}', NULL, '2026-07-15T15:47:21.791776+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-2', NULL),
    ('621f9c23-591d-4810-a2b4-e14d8918f20a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', NULL, '2026-07-16T12:00:00+00:00', '2026-07-16T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-3","source":"web","notes":"Bedminton"}', NULL, '2026-07-15T15:47:23.430239+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-3', NULL),
    ('e1dea4f3-ee55-4a67-be51-0769655bb042', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', NULL, '2026-07-16T12:00:00+00:00', '2026-07-16T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-4","source":"web","notes":"Bedminton"}', NULL, '2026-07-15T15:47:25.319863+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-4', NULL),
    ('0a16ca7e-7031-4b24-9edc-f6f48e40d818', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-20T17:00:00+00:00', '2026-07-20T19:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T16:49:16.623218+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('3cb6a912-c314-4862-9b4e-6101e06a4524', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-07-23T17:00:00+00:00', '2026-07-23T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu."}', NULL, '2026-07-23T09:08:34.95796+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('c96ac2fa-abdb-4d9e-9e03-08741cee877d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-26T17:00:00+00:00', '2026-07-26T18:30:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-clay-1","source":"web","notes":"Dvojhra"}', NULL, '2026-07-26T08:01:36.652529+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('3bbea9c7-19b5-4d74-a64a-96ae124246d4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', NULL, '2026-07-26T17:00:00+00:00', '2026-07-26T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-clay-2","source":"web","notes":"Kamil"}', NULL, '2026-07-26T08:02:19.501005+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-2', NULL),
    ('ec7e0089-4b08-4113-95d2-b24b2a0c4be4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Tomčofčík', '+421907173176', '2026-07-28T13:00:00+00:00', '2026-07-28T17:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-27T12:11:47.02844+00:00', NULL, '689f3f3b-5596-4859-bc1a-f26cdfa9ba36', 'badminton', 'badminton-1', NULL),
    ('b1f737a9-a325-4295-8f09-8cfba5e48dc6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-28T14:00:00+00:00', '2026-07-28T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alica"}', NULL, '2026-07-27T19:01:08.21892+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('bc6f9fdf-6776-4c0c-b266-400bb1d61077', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-29T15:00:00+00:00', '2026-07-29T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Z\u00e1kazn\u00edk m\u00e1 z\u00e1ujem o zapo\u017ei\u010danie rakiet."}', NULL, '2026-07-28T16:50:38.980743+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('2bb496f2-7bba-4bf0-ac27-c2b8b221a25f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T08:00:00+00:00', '2026-07-29T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-28T19:07:15.924936+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('0262da90-7963-47ad-849a-2d033be7d96a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T13:00:00+00:00', '2026-07-29T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku Alicu"}', NULL, '2026-07-28T19:08:11.328131+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('71491a00-2045-4708-9214-ec5f79552f0a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T08:00:00+00:00', '2026-07-30T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T17:05:04.977062+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('dde7316e-97d5-45fe-8c34-0c7890f3ea05', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T09:00:00+00:00', '2026-07-30T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T20:39:16.449145+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('d2b398de-9ed6-4d5e-9910-af344674c4a4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '+421905012054', '2026-07-30T12:00:00+00:00', '2026-07-30T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-3", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-30T09:59:13.887736+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-3', NULL),
    ('f3ac4b2f-a4ae-485a-a9bf-dc48e7dce1a2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil', '+421915962068', '2026-07-31T16:00:00+00:00', '2026-07-31T17:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-31T14:56:44.741414+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('dd1ba4fa-d0c7-4c88-b4c2-c26edb087636', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Neznámy zákazník', '', '2026-08-01T09:00:00+00:00', '2026-08-01T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-01T07:59:34.221581+00:00', NULL, NULL, 'squash', 'squash-2', NULL),
    ('0e72f16f-d9bf-465c-a51f-a7fbca6564ee', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T07:00:00+00:00', '2026-08-03T08:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:52:25.054319+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('a3e36f0d-e268-4adb-b720-1a80501938c7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T08:00:00+00:00', '2026-08-03T09:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:52:28.106403+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', NULL),
    ('73dbbc06-0bcf-45e6-95f3-380a531f93ec', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T12:00:00+00:00', '2026-08-03T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:52:30.615023+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('18de8479-a244-4080-9542-ec4c6b0d70f8', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T14:00:00+00:00', '2026-08-03T15:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:52:33.255582+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', NULL),
    ('e9196559-eac1-4de7-9663-ab2c095765a5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T16:00:00+00:00', '2026-08-03T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:52:36.019004+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('99cd1803-3e7f-459c-a906-fcc36f20a348', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T11:00:00+00:00', '2026-08-03T12:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:52:38.874378+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', NULL),
    ('d7d23147-3f26-4aeb-8572-4873e2ffbe82', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T18:00:00+00:00', '2026-08-03T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:52:41.510766+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', NULL),
    ('52e1fab6-683e-4104-8eb6-2633d478fbbd', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T14:00:00+00:00', '2026-08-03T15:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:52:54.225165+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('c2ecee82-b40e-44d1-82ea-3d311c99f1eb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T13:00:00+00:00', '2026-08-03T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:52:56.878006+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('df3b1d72-5f63-41ee-91f4-5785a9b42844', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T14:00:00+00:00', '2026-08-03T15:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:52:59.829502+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('b2b122b6-61b7-4a21-9d11-9e206eb197be', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T15:00:00+00:00', '2026-08-03T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:53:02.91178+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', NULL),
    ('20341a99-43f8-4c1c-a107-86efed45ac71', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-13T15:00:00+00:00', '2026-07-13T16:00:00+00:00', 'confirmed', NULL, '{"notes": "Členstvo NTC", "source": "azure-realtime-voice-assistant", "courtId": "squash-2"}', NULL, '2026-07-13T07:06:04.726512+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-2', NULL),
    ('0199cd33-d898-4b28-b0d3-3f624e9161ea', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-13T16:00:00+00:00', '2026-07-13T17:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "tennis-clay-1"}', NULL, '2026-07-13T07:15:25.86695+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('d57d8926-8ddb-4625-87a9-dc67d18ff602', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T15:00:00+00:00', '2026-07-13T16:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-3"}', NULL, '2026-07-13T07:58:26.002019+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-3', NULL),
    ('79a7e906-6094-4117-92ff-8dff89494ac2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T11:00:00+00:00', '2026-07-13T12:00:00+00:00', 'cancelled', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-13T06:43:00.005541+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('0c2d36df-6ac1-41d4-ba3a-aa4ff7f6b0bd', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T10:00:00+00:00', '2026-07-13T11:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "squash-1"}', NULL, '2026-07-13T08:10:50.517198+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('72d722e2-55de-48fc-a3c8-70c52c989d8f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T13:00:00+00:00', '2026-07-13T14:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-13T08:15:53.331588+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('23736cae-fb18-4fe1-8295-e5c896f6a7ef', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T12:00:00+00:00', '2026-07-13T13:00:00+00:00', 'cancelled', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-13T08:00:21.41802+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('7f693fac-1c3b-452f-947e-f09f8678746b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-13T14:00:00+00:00', '2026-07-13T15:00:00+00:00', 'confirmed', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "badminton-1"}', NULL, '2026-07-13T08:42:23.999819+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('be69aae6-f957-4fd1-97fe-b8898e372f1e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T13:00:00+00:00', '2026-07-16T14:00:00+00:00', 'cancelled', NULL, '{"notes": "", "source": "azure-realtime-voice-assistant", "courtId": "badminton-2"}', NULL, '2026-07-13T08:09:13.09023+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('7fd06a06-dfce-4ca0-8e63-0953f67b679c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T15:00:00+00:00', '2026-07-16T16:00:00+00:00', 'cancelled', NULL, '{"courtId": "squash-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T15:48:17.991776+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('d987f5ae-dda4-4d83-914d-9f1789b587ac', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-16T10:00:00+00:00', '2026-07-16T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-clay-1", "source": "azure-realtime-voice-assistant", "notes": ""}', NULL, '2026-07-15T17:40:39.703976+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis-clay', 'tennis-clay-1', NULL),
    ('0512d1cd-ee1f-4bc0-be71-72e8ef12fe32', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-23T12:00:00+00:00', '2026-07-23T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "\u010clensk\u00e1 karta NTC"}', NULL, '2026-07-23T09:09:59.973198+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('9ac6f7fb-ca4e-4fd4-a841-0c88507326ca', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-27T08:00:00+00:00', '2026-07-27T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Active TC Membership Card holder"}', NULL, '2026-07-26T16:27:55.577592+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('f8338e7b-4566-4a77-b329-a8d7c0e745b7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Tomčofčík', NULL, '2026-07-28T18:00:00+00:00', '2026-07-28T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Fnv"}', NULL, '2026-07-27T12:16:33.7136+00:00', NULL, '689f3f3b-5596-4859-bc1a-f26cdfa9ba36', 'badminton', 'badminton-1', NULL),
    ('e13760c9-9e28-4f6a-8025-35d6f2186e28', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-07-28T08:00:00+00:00', '2026-07-28T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-27T19:07:02.631499+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('78eaf59e-d069-492f-8b25-e4da77c19d39', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-31T07:00:00+00:00', '2026-07-31T08:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-28T17:44:56.889134+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('af8133f6-05ad-4f48-aa66-6832619eda39', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-29T10:00:00+00:00', '2026-07-29T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T07:14:51.335674+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('18d7a181-c1a0-49d7-90c8-23b0ee3802fd', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T08:00:00+00:00', '2026-07-30T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-3", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T17:06:30.748145+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', NULL),
    ('4c7de631-8a4d-4551-8820-7339bb206a47', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-07-30T09:00:00+00:00', '2026-07-30T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-29T20:50:10.967891+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('20bc03fd-4cb3-41b4-a826-dfdd396fe9b4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '+421915962068', '2026-07-30T17:00:00+00:00', '2026-07-30T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-07-30T15:05:51.854512+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('e2a7bf18-9110-4851-b407-0a0e9ade87fd', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-08-02T13:00:00+00:00', '2026-08-02T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-01T05:37:04.370525+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('d92bb43b-990f-4c39-919d-b92bd0b4b5d4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-08-02T13:00:00+00:00', '2026-08-02T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-01T05:37:05.9867+00:00', NULL, NULL, 'tennis', 'tennis-2', NULL),
    ('7573c161-e4a3-452d-871c-9c0558840ff2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Zákazník', '', '2026-08-01T09:00:00+00:00', '2026-08-01T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-3", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-01T08:02:36.326033+00:00', NULL, NULL, 'squash', 'squash-3', NULL),
    ('6be51199-d448-4092-bdfc-67e2f314663c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-01T09:00:00+00:00', '2026-08-01T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-4", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-01T08:03:22.764023+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-4', NULL),
    ('e378174d-2e74-479a-a9a3-d23588a50b2c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-03T18:00:00+00:00', '2026-08-03T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-03T15:59:32.570736+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('f1c3aca2-9629-4c19-998b-e72a840d58b0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T18:00:00+00:00', '2026-08-03T19:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-03T10:47:32.875403+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('c43f6b87-b744-49e6-a150-2e474112ba75', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-03T19:00:00+00:00', '2026-08-03T20:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-02T12:52:44.192902+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('c9650150-75f0-4e2e-b032-9613adc9ee50', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-03T17:00:00+00:00', '2026-08-03T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-03T16:02:49.989855+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('cd1850d3-0461-4cce-ab99-2ac7af11f63f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peťo', 'client:web-user-1785775893', '2026-08-04T11:00:00+00:00', '2026-08-04T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-03T16:52:51.736724+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('f46cc72e-468e-4d68-984e-f2ea8195128a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jozef Mrkvička', '0915962068', '2026-08-04T07:00:00+00:00', '2026-08-04T08:00:00+00:00', 'cancelled', NULL, '{"courtId":"squash-1","source":"web","notes":"Squash"}', NULL, '2026-08-03T17:11:41.555883+00:00', NULL, '6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'squash', 'squash-1', NULL),
    ('02e4fd85-e734-424c-a78f-57f88353bf6c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jozef Mrkvička', '+421915962068', '2026-08-04T08:00:00+00:00', '2026-08-04T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-03T17:36:54.682863+00:00', NULL, '6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'squash', 'squash-1', NULL),
    ('5e3241fe-d5a1-4b32-8099-dae5cd3063a6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-08-04T10:00:00+00:00', '2026-08-04T12:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Ahoj"}', NULL, '2026-08-03T18:14:41.388545+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('df629ec5-a7ac-4705-8752-1df3fc18d6d6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jozef Mrkvička', '+421915962068', '2026-08-04T08:00:00+00:00', '2026-08-04T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-03T18:27:45.785813+00:00', NULL, '6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'badminton', 'badminton-1', NULL),
    ('0ba778a8-8ad3-4e6f-8939-5daff6724772', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Neznámy zákazník', 'client:web-user-1785826304', '2026-08-04T09:00:00+00:00', '2026-08-04T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-04T06:52:18.872855+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('b42349e6-07b4-4520-86d3-c529904aa39a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jozef Mrkvička', '+421915962068', '2026-08-04T10:00:00+00:00', '2026-08-04T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-04T06:54:07.425892+00:00', NULL, '6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'squash', 'squash-1', NULL),
    ('4d430161-3df4-4f42-855b-a35d80743149', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jozef Mrkvička', '+421915962068', '2026-08-04T09:00:00+00:00', '2026-08-04T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-04T06:55:33.512137+00:00', NULL, '6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'squash', 'squash-1', NULL),
    ('28316cd7-1888-49d0-8158-c7272213dc88', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Neznámy zákazník', '+421915962068', '2026-08-04T10:00:00+00:00', '2026-08-04T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-04T07:02:06.909386+00:00', NULL, '6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'badminton', 'badminton-2', NULL),
    ('1b8bb836-050e-4942-b8f9-7e8551d2e892', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jozef Mrkvička', '+421915962068', '2026-08-04T10:00:00+00:00', '2026-08-04T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-04T07:03:17.546198+00:00', NULL, '6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'tennis', 'tennis-1', NULL),
    ('dbe7721a-430c-43fb-8251-f0f0be395ab2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Neznámy zákazník', '+421915962068', '2026-08-05T09:00:00+00:00', '2026-08-05T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-clay-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-04T07:08:31.548771+00:00', NULL, '6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'tennis-clay', 'tennis-clay-1', NULL),
    ('1a35130e-c4ef-4948-9950-9b827d2ce64d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jozef Mrkvička', '+421915962068', '2026-08-04T13:00:00+00:00', '2026-08-04T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-04T08:26:42.603536+00:00', NULL, '6415c33f-ab6c-4eab-b3ac-7986a1b6ace7', 'squash', 'squash-1', NULL),
    ('e3469672-6865-429e-ae60-6482bdd6c53c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-10T11:00:00+00:00', '2026-08-10T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-clay-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-04T08:57:56.575104+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('22237bdb-5e28-4f82-bea4-8b132bf67ae8', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-04T13:00:00+00:00', '2026-08-04T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-04T10:00:20.388022+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('13e0c304-5979-488e-8f43-199a3b5e4b12', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-04T15:00:00+00:00', '2026-08-04T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-04T11:06:23.152862+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('e2a18613-e6ee-4e84-a616-494e6b266eef', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-05T11:00:00+00:00', '2026-08-05T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-05T09:17:39.902138+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('3b8212fe-2da1-4b4c-9c0a-69688e77e5cf', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-05T12:00:00+00:00', '2026-08-05T13:30:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Chceme si pozicat raketky"}', NULL, '2026-08-05T09:18:55.490252+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('24505ee8-1938-4ca8-95ed-fcfc3fd63d58', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-06T12:00:00+00:00', '2026-08-06T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-06T08:48:08.09432+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('e9fa9057-84f5-45d4-bb9e-d027df1e8897', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-06T19:00:00+00:00', '2026-08-06T20:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-06T15:01:14.642741+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('bf5a6093-7330-4957-8aa2-98dff8ec7aae', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-06T18:00:00+00:00', '2026-08-06T19:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-06T15:16:09.851859+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('7ba68b3a-6cd4-4cdd-8b3b-0bd53e0ea395', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-06T17:00:00+00:00', '2026-08-06T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-06T15:18:32.513031+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('a996d95d-fc32-4e24-b5e6-c755878e67e1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-07T15:00:00+00:00', '2026-08-07T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-06T15:28:46.717818+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('15c11c7f-ac8a-4206-8ce6-6f194d825f5a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-07T16:00:00+00:00', '2026-08-07T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-06T15:28:54.3289+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('35ca446b-f6b1-43fe-9fc9-aa10c7bb1b2e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '+421915962068', '2026-08-07T17:00:00+00:00', '2026-08-07T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant-walkin", "notes": "Telefonická rezervácia návštevníka"}', NULL, '2026-08-07T13:33:06.770293+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('ae44a1bd-2901-4094-b838-bd911ec61f53', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-07T16:00:00+00:00', '2026-08-07T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-1","source":"web","notes":"Squash"}', NULL, '2026-08-07T13:35:24.440443+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('f3ed4205-6188-42ff-af64-878c89f26bf6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-07T17:00:00+00:00', '2026-08-07T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-07T14:41:13.660817+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('33068ecf-4d52-4fd3-a1a0-565e4324c410', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-12T16:00:00+00:00', '2026-08-12T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-1","source":"web","notes":"Tenis indoor"}', NULL, '2026-08-12T10:54:44.505145+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('d58b3fb2-4294-4d8a-84a3-151938269191', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-08T10:00:00+00:00', '2026-08-08T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia vytvoren\u00e1 hlasov\u00fdm asistentom"}', NULL, '2026-08-08T08:03:41.993277+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('2b2894c6-ae35-4ba8-9e74-bc10ef1da4be', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-09T12:00:00+00:00', '2026-08-09T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-1","source":"web","notes":"Squash"}', NULL, '2026-08-09T09:33:04.208312+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('4f571b6b-beae-40c0-a16a-12afadb518d0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-09T16:00:00+00:00', '2026-08-09T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Skuska rezervovania kurtu"}', NULL, '2026-08-09T13:12:47.965281+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('0091dc25-ebaa-4885-a61e-44d796b93d9b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-09T18:00:00+00:00', '2026-08-09T19:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-09T16:27:31.556684+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('e84ced84-514a-4899-bbad-d105212caf26', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-09T18:00:00+00:00', '2026-08-09T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-09T17:02:19.653994+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('9c1f83a9-1179-4408-ac76-67970f72f121', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-12T10:00:00+00:00', '2026-08-12T11:30:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-09T17:03:05.785208+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('f0eed5c9-1ef8-41b2-b53a-7f144740bb34', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-10T11:00:00+00:00', '2026-08-10T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia pre Kamilku"}', NULL, '2026-08-09T17:32:31.095411+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('63296d48-2f35-4444-82f0-209028e21a56', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-10T07:00:00+00:00', '2026-08-10T08:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-09T18:53:52.523536+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('1731ec3e-d701-4f03-920c-f1b61bbc9dd0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-10T07:00:00+00:00', '2026-08-10T08:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-09T18:55:06.425625+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('fd02c914-422e-4a09-9108-f3e7092f36bb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-10T09:00:00+00:00', '2026-08-10T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-09T19:00:49.590755+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('0287687d-28a7-4c76-85a3-019d2f65a216', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-10T11:00:00+00:00', '2026-08-10T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-10T09:45:36.028448+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('d4dfe2d4-4760-4fa6-bc00-75b328671306', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-10T12:00:00+00:00', '2026-08-10T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-10T09:50:59.914109+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('6f9b7c00-d8d3-4f43-bf59-2ef3f9ffdc1b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-10T13:00:00+00:00', '2026-08-10T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-10T09:51:17.766858+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('4b38c7ff-975c-400e-a2af-a515a1c12a3c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-13T14:00:00+00:00', '2026-08-13T15:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-10T09:51:36.790855+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('22045b13-34e6-4c93-bd77-74c0d9e6e1f4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-12T13:00:00+00:00', '2026-08-12T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-10T09:52:27.128283+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('07b5cb3c-8813-4ce0-86e4-bc69b853b388', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-12T13:00:00+00:00', '2026-08-12T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-10T09:52:40.231143+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('66eaccbf-22d7-4701-a822-53112dcfcc60', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-12T14:00:00+00:00', '2026-08-12T15:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-10T09:53:06.119942+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('4fc008a7-2dc1-4dd6-9c94-20bceab2fb73', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-12T15:00:00+00:00', '2026-08-12T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia bedmintonu cez hlasov\u00fa asistentku"}', NULL, '2026-08-10T09:54:10.799241+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('1ec410f8-f655-4cae-ac38-6a63589a024b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-11T09:00:00+00:00', '2026-08-11T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-11T07:28:23.538362+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('8981654a-6e7a-4096-84d7-8c96931d8c1c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-11T09:00:00+00:00', '2026-08-11T10:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Prva hra dnes"}', NULL, '2026-08-11T07:28:55.558265+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('a73d7011-2ccf-4e3f-b8d3-cda96a96308d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-11T11:00:00+00:00', '2026-08-11T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-11T07:49:44.702749+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('67b97cea-02ed-4a2d-ad53-352c86af09bd', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-11T18:00:00+00:00', '2026-08-11T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-11T13:56:01.825989+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('7e5c85f0-dac8-4c8a-acdd-4d1e63a703b1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-13T16:00:00+00:00', '2026-08-13T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-1","source":"web","notes":"Squash"}', NULL, '2026-08-11T14:08:14.698981+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('d53975f7-6959-4098-b1d6-4381eefc36b3', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-20T15:00:00+00:00', '2026-08-20T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-11T14:13:23.933057+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('3d29a2e9-28a6-4139-968d-d865c806ae44', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-12T18:00:00+00:00', '2026-08-12T19:00:00+00:00', 'cancelled', NULL, '{"courtId":"squash-2","source":"web","notes":"Squash"}', NULL, '2026-08-11T13:56:21.146088+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-2', NULL),
    ('dbf028b0-b0a5-4727-9fac-e5ef9ce5e798', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-11T19:00:00+00:00', '2026-08-11T20:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton"}', NULL, '2026-08-11T16:44:33.9174+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', NULL),
    ('874f9005-a532-42a5-a09b-1d6c3b218774', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-18T18:00:00+00:00', '2026-08-18T19:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-11T17:00:17.373888+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('ad23bbf3-a17d-494e-8b68-bd2a26086d66', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-18T15:00:00+00:00', '2026-08-18T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-11T17:00:48.621785+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('afe8418f-9cbe-4181-8726-a503d5296f25', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-12T10:00:00+00:00', '2026-08-12T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-12T07:08:57.045768+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('d9b4ad42-60b3-4828-904d-668be2e7d25f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-12T15:00:00+00:00', '2026-08-12T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T10:36:07.650519+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('bb919617-5694-4503-b195-fa940671cd85', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-12T17:00:00+00:00', '2026-08-12T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T11:02:43.398162+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('5ef3e838-8ef4-4383-9eb2-c1704d1898fa', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-12T17:00:00+00:00', '2026-08-12T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T11:02:52.666161+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('7a1d728c-aeda-49c1-9c85-d18a425588e7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-14T12:00:00+00:00', '2026-08-14T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T11:18:48.728371+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('f7c27ab8-0cbb-43f2-af20-c6a57d76427c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-12T18:00:00+00:00', '2026-08-12T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T12:14:19.653245+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('7871bd62-8167-47e0-a720-2d1abefc8baa', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-12T19:00:00+00:00', '2026-08-12T20:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T14:30:38.424118+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', NULL),
    ('06aa86c3-36bf-46ea-ae91-3c75821600a4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-14T15:00:00+00:00', '2026-08-14T16:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T11:19:27.707135+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('342717c8-4fcd-4cb7-b5d8-73f93cc39699', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-14T15:00:00+00:00', '2026-08-14T16:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T11:18:40.11115+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('14c98e9d-da23-4c82-9d3c-936fc157ba10', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-14T15:00:00+00:00', '2026-08-14T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T14:31:00.353153+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('fd946e9e-f6b9-434c-af85-7f442365848f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-14T13:00:00+00:00', '2026-08-14T14:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T11:19:05.159981+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('ad70f835-c425-48ac-8631-4fd06745cd24', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-14T14:00:00+00:00', '2026-08-14T15:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Chcem si pozicat raketku"}', NULL, '2026-08-12T14:31:30.693627+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', NULL),
    ('7cb4fab0-787c-43bd-932c-03beb0550f3b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-14T15:00:00+00:00', '2026-08-14T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-12T14:33:30.556657+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('9b9e324b-660b-40f4-bc92-4fb1e5efb6fd', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-14T19:00:00+00:00', '2026-08-14T21:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T20:03:51.719258+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('29cfdcc0-0a56-41fa-8f1f-92c5f987772b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-14T09:00:00+00:00', '2026-08-14T10:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T11:19:35.307511+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('5201919b-125f-4894-9619-85de011fa922', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-14T09:00:00+00:00', '2026-08-14T10:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T20:04:42.539404+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', NULL),
    ('6a017dc7-3282-402e-a030-446ab3f9df4f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-13T14:00:00+00:00', '2026-08-13T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-13T08:31:55.568577+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('de438b5c-c656-447a-960c-3848b5795167', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-13T14:00:00+00:00', '2026-08-13T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-13T08:38:19.984441+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('e2b589d5-44e0-4f17-8565-a83324e0a9d2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-13T19:00:00+00:00', '2026-08-13T20:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-13T09:27:45.542746+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('2e808eaf-2be5-4ef3-9918-e5c6cd756877', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Google Test', '000000000000', '2026-12-20T09:00:00+00:00', '2026-12-20T10:00:00+00:00', 'confirmed', NULL, '{"source":"wallet-sql-test"}', NULL, '2026-08-15T10:35:28.313161+00:00', NULL, '6392e3cf-c8a1-487b-adde-394e1a0ea31f', 'badminton', 'badminton-10', NULL),
    ('67212f54-f19e-4474-9400-4b998643fb87', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Google Test', '000000000000', '2026-12-20T09:00:00+00:00', '2026-12-20T10:00:00+00:00', 'confirmed', NULL, '{"source":"wallet-sql-test"}', NULL, '2026-08-15T10:37:38.345698+00:00', NULL, '6392e3cf-c8a1-487b-adde-394e1a0ea31f', 'badminton', 'badminton-1', NULL),
    ('35138f04-2f01-430f-8fc6-4b151f85ebfe', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-16T08:00:00+00:00', '2026-08-16T09:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-1","source":"web","notes":"Squash"}', NULL, '2026-08-15T14:50:56.533635+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', NULL),
    ('730f7c0a-2aa6-4381-b2b4-cdfd782be9ac', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-16T08:00:00+00:00', '2026-08-16T09:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-2","source":"web","notes":"Squash"}', NULL, '2026-08-15T14:53:17.090787+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-2', NULL),
    ('cf998eaf-dafe-460b-b0da-61bc54cd7ef2', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Google Test', '000000000000', '2026-08-15T17:00:00+00:00', '2026-08-15T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-15T15:21:06.037425+00:00', NULL, '6392e3cf-c8a1-487b-adde-394e1a0ea31f', 'badminton', 'badminton-1', NULL),
    ('85d597cf-7afb-4b05-92ee-c7cdf43475aa', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Google Test', '000000000000', '2026-08-16T17:00:00+00:00', '2026-08-16T18:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-15T15:21:17.325574+00:00', NULL, '6392e3cf-c8a1-487b-adde-394e1a0ea31f', 'badminton', 'badminton-2', NULL),
    ('460624dc-9c2a-4e81-8db1-687c3b2593c3', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Google Test', '000000000000', '2026-08-16T12:00:00+00:00', '2026-08-16T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-15T15:26:08.542568+00:00', NULL, '6392e3cf-c8a1-487b-adde-394e1a0ea31f', 'badminton', 'badminton-1', NULL),
    ('6837b2a7-2cb0-4a41-8f72-d18052810ef1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-15T18:00:00+00:00', '2026-08-15T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-15T16:12:47.970463+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('53f20426-7665-4a0d-9983-6b959c1a203f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-16T13:00:00+00:00', '2026-08-16T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-16T06:40:19.579211+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('ae5a1c99-50ea-4699-9c17-6f3daca948c9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-17T14:00:00+00:00', '2026-08-17T15:00:00+00:00', 'cancelled', NULL, '{"courtId":"tennis-1","source":"web","notes":"Tenis indoor"}', NULL, '2026-08-16T06:40:39.627236+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('160487c1-126e-4d55-b8ba-94ebe8a1d5d6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-17T16:00:00+00:00', '2026-08-17T17:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-16T07:04:09.782353+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('a3970233-6911-461a-99d4-f9dd67e4c686', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-17T13:00:00+00:00', '2026-08-17T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-16T07:04:01.898249+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('28a8f0ff-6e78-4862-8d69-c1b78d5e665e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-22T12:00:00+00:00', '2026-08-22T13:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-12T20:09:41.386807+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('9bbf9263-9a5e-449b-ba7f-44bf5d3a8489', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-17T19:00:00+00:00', '2026-08-17T20:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-16T08:19:55.574327+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('bf050c6c-a2a5-44a3-b009-ed42275057c1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-16T15:00:00+00:00', '2026-08-16T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-16T08:50:33.868008+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', NULL),
    ('2f6ba608-99c5-4ec0-b558-d390d08dcff9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-16T17:00:00+00:00', '2026-08-16T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-16T08:50:36.986472+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('d27fbaed-a2ef-416f-b5c9-0afc15a2283b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-17T14:00:00+00:00', '2026-08-17T15:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton"}', NULL, '2026-08-16T08:50:41.507453+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-3', NULL),
    ('93ed146c-deaf-46c6-b486-2171ecf316e1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-17T15:00:00+00:00', '2026-08-17T16:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-16T08:18:16.635437+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('5f9efad0-1a90-4203-92cd-bc5f4b309ee3', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-17T19:00:00+00:00', '2026-08-17T20:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-16T08:50:49.426617+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', NULL),
    ('b1695cc3-51df-49a6-abf7-509ed6b2eeb1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-18T18:00:00+00:00', '2026-08-18T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-clay-1","source":"web","notes":"Tenis antuka"}', NULL, '2026-08-16T09:40:00.046362+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('6d5c3949-96a1-4fea-82d6-50fd01c4fb93', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-18T15:00:00+00:00', '2026-08-18T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-clay-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-16T09:46:18.674923+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('14b3811f-0a12-4130-8945-f3b06d1d2884', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-18T11:00:00+00:00', '2026-08-18T12:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-1","source":"web","notes":"prideme dvaja"}', NULL, '2026-08-16T10:46:42.735601+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', NULL),
    ('c4f7f296-4c04-45f3-89c3-c4f4736ca312', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-18T09:00:00+00:00', '2026-08-18T10:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-16T10:48:49.285342+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('f1237f26-1782-4a07-8c63-3c7c49e7d7bb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Neznámy zákazník', '+421915962068', '2026-08-18T17:00:00+00:00', '2026-08-18T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-18T07:32:49.954253+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('daa1a93f-a604-4706-b1b1-dcf40d21ddd6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-18T14:00:00+00:00', '2026-08-18T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "rezerv\u00e1cia vytvoren\u00e1 telefonicky"}', NULL, '2026-08-18T07:38:31.50631+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('d3f4d0b2-839d-419d-bdaa-4b6c6f050f91', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-08-19T13:00:00+00:00', '2026-08-19T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-1","source":"web","notes":"Tenis indoor"}', NULL, '2026-08-18T07:59:49.339401+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-1', NULL),
    ('715d6a50-3086-405c-9ed7-2015c3d927ce', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-19T17:00:00+00:00', '2026-08-19T18:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-17T15:49:03.68689+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('8925a819-be34-469c-bdda-f51bf4833208', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-08-19T18:00:00+00:00', '2026-08-19T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-18T08:03:20.04863+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('dfc38d94-cb9a-4ef7-baf5-afb10aea16b5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Milan Bugan', '0948291231', '2026-08-19T10:00:00+00:00', '2026-08-19T11:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-19T08:30:30.522328+00:00', NULL, '2692ab42-4f19-4bd4-9d47-83e50c8fae44', 'badminton', 'badminton-2', NULL),
    ('97c11762-3e28-4c9f-8cdf-07e9e4fa540c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Oliver Urda', '0944420304', '2026-08-21T17:00:00+00:00', '2026-08-21T19:00:00+00:00', 'cancelled', NULL, '{"courtId":"tennis-5","source":"web","notes":"Isto prídem"}', NULL, '2026-08-19T08:59:34.219463+00:00', NULL, 'c6ff6936-a630-4ecd-8cb8-7bdc2289607f', 'tennis', 'tennis-5', NULL),
    ('39886058-34e1-4892-9b6b-6052cc42329d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Oliver Urda', '+421944420304', '2026-08-20T16:00:00+00:00', '2026-08-20T17:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "rezerv\u00e1cia cez telef\u00f3n"}', NULL, '2026-08-19T09:04:27.951627+00:00', NULL, 'c6ff6936-a630-4ecd-8cb8-7bdc2289607f', 'badminton', 'badminton-1', NULL),
    ('48222dd3-1da3-443e-90f0-3d77382d76ae', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-20T09:00:00+00:00', '2026-08-20T10:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-19T09:10:38.47707+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('7c09d731-2985-491e-af91-16be4d11779d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-08-20T18:00:00+00:00', '2026-08-20T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton"}', NULL, '2026-08-19T14:43:12.410712+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', NULL),
    ('abcda25b-e727-480c-998f-f8080d45096c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-20T18:00:00+00:00', '2026-08-20T19:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia vytvoren\u00e1 telefonicky"}', NULL, '2026-08-19T14:45:02.03178+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('02278d6d-2492-4413-9da8-e8e8079c99ec', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-21T07:00:00+00:00', '2026-08-21T08:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-19T15:18:19.24635+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('c6fdb69f-87ca-402a-b128-cc6e253500ca', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Štefan Varga', '0907389210', '2026-08-24T10:00:00+00:00', '2026-08-24T11:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-3","source":"web","notes":"Squash"}', NULL, '2026-08-20T10:14:29.876232+00:00', NULL, 'a72eab08-214c-43ad-b041-d7380ca801a0', 'squash', 'squash-3', 13.0),
    ('174fd77e-b6a3-4df2-af5f-a8f70b4308dc', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Štefan Varga', '0907389210', '2026-08-29T16:00:00+00:00', '2026-08-29T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-4","source":"web","notes":"Tenis indoor"}', NULL, '2026-08-20T10:14:40.545447+00:00', NULL, 'a72eab08-214c-43ad-b041-d7380ca801a0', 'tennis', 'tennis-4', 19.0),
    ('645ae7ec-ca17-4875-a4e0-f34d5bfdf5f7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-08-20T17:00:00+00:00', '2026-08-20T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-1","source":"web","notes":"Squash"}', NULL, '2026-08-20T10:31:20.977463+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', 15.0),
    ('0200076d-80c6-4d9f-9465-4a70a3e9c05b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-08-21T16:00:00+00:00', '2026-08-21T16:30:00+00:00', 'cancelled', NULL, '{"courtId":"squash-1","source":"web","notes":"Squash"}', NULL, '2026-08-20T10:31:40.95752+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', 7.5),
    ('6d77f8c8-7317-4afc-a24d-a83f1bb33fe1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-20T17:00:00+00:00', '2026-08-20T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-20T15:38:10.357724+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', 19.0),
    ('52da8408-a0a6-4eef-8c88-8b7fcccacc5f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-08-23T18:00:00+00:00', '2026-08-23T20:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-20T16:36:05.254775+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 26.0),
    ('1ed40436-c44e-46ab-853e-4bd52ea09774', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-08-21T18:00:00+00:00', '2026-08-21T19:30:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-20T15:12:50.233345+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', 28.5),
    ('d3985d26-2b5e-429f-8e3b-fffdcd4f6869', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-08-23T10:00:00+00:00', '2026-08-23T11:30:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-2","source":"web","notes":"Tenis indoor"}', NULL, '2026-08-21T08:01:04.230139+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis', 'tennis-2', 25.5),
    ('848d68d2-9943-4883-9008-4bfb08282df4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Janka Bartková', '+421908819991', '2026-08-21T14:00:00+00:00', '2026-08-21T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "zarezervovane cez telefonat"}', NULL, '2026-08-21T08:57:42.581381+00:00', NULL, 'b1fb24b3-3f4b-4b0a-902d-47e65c9f48fe', 'badminton', 'badminton-1', NULL),
    ('897363b7-0e69-4113-8a98-1642c8c4a4d6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-21T17:00:00+00:00', '2026-08-21T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia vytvoren\u00e1 telefonicky"}', NULL, '2026-08-21T09:02:22.405737+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('b2f525ca-853d-4241-9765-f8914eb0314a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-21T18:00:00+00:00', '2026-08-21T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton"}', NULL, '2026-08-21T13:34:49.479957+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-3', 19.0),
    ('0d035f6e-aa78-457e-9026-ed7a01011b3d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-28T13:00:00+00:00', '2026-08-28T14:00:00+00:00', 'cancelled', NULL, '{"courtId":"squash-1","source":"web","notes":"Squash"}', NULL, '2026-08-21T13:35:22.540377+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', 11.0),
    ('efd00a09-9910-496c-9197-e0893b7034b4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-22T15:00:00+00:00', '2026-08-22T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-22T08:23:10.619678+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 13.0),
    ('075c42ff-433f-4b19-9429-1da46098faf4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-22T13:00:00+00:00', '2026-08-22T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-22T10:56:19.183937+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 13.0),
    ('77da839a-43c6-473d-9828-5ac880c7537a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-23T14:00:00+00:00', '2026-08-23T15:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-22T10:56:34.155692+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', 13.0),
    ('0e95d40c-384e-412a-aea0-abb8a0738088', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-23T13:00:00+00:00', '2026-08-23T14:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-23T09:31:09.487666+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', 11.0),
    ('bf9f8ef8-297f-4458-906a-3fbba3f1e4e7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-24T17:00:00+00:00', '2026-08-24T18:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-23T13:20:09.577557+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 17.0),
    ('93392940-1dfa-45da-8510-11148424959a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-23T17:00:00+00:00', '2026-08-23T18:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-23T13:07:25.859462+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 11.0),
    ('aafee5a1-26dd-44e9-979f-b058f6217401', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-25T13:00:00+00:00', '2026-08-25T14:30:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton"}', NULL, '2026-08-23T13:34:58.615499+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-4', 19.5),
    ('41d1d789-ef33-4e53-b0bb-8c7248aed6fb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-25T13:00:00+00:00', '2026-08-25T14:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-23T09:31:24.751721+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', 11.0),
    ('0093d957-9ca7-474d-b383-ee057d6e8444', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin Údržba', NULL, '2026-08-25T11:00:00+00:00', '2026-08-25T12:00:00+00:00', 'blocked', NULL, '{"courtId":"badminton-1","source":"admin","notes":"Údržba"}', NULL, '2026-08-23T14:13:30.081248+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', 13.0),
    ('93de872b-4d0e-4481-953a-15239d149745', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-08-26T13:00:00+00:00', '2026-08-26T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-24T17:03:12.841887+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', 13.0),
    ('ff0466a2-6c62-4905-96d6-71edcedf678e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-08-26T17:00:00+00:00', '2026-08-26T18:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-24T17:03:37.036572+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', 19.0),
    ('359576fc-2156-42d4-86bf-beeacfb2d54a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-08-26T14:00:00+00:00', '2026-08-26T15:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-24T16:56:32.594939+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-2', 19.0),
    ('84f27eb4-d580-4980-aa28-994f4a9a2956', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-24T18:00:00+00:00', '2026-08-24T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-24T17:08:57.53664+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', 17.0),
    ('122152c2-bf14-4c96-aa95-084bb5a2ce67', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-08-26T15:00:00+00:00', '2026-08-26T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton"}', NULL, '2026-08-24T17:09:21.372922+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-4', 19.0),
    ('9594c4e8-e5bf-4ff1-bd37-657799540c7e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-25T17:00:00+00:00', '2026-08-25T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-24T17:09:28.778443+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', 17.0),
    ('33523563-5323-4eaf-a6c2-8be06ea9f99e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-26T17:00:00+00:00', '2026-08-26T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-24T17:09:43.02555+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', 17.0),
    ('0764bb9b-d682-4466-a072-15cd1cc83e59', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-08-26T15:00:00+00:00', '2026-08-26T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"admin","notes":"Údržba"}', NULL, '2026-08-24T17:24:41.495735+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', 0.0),
    ('61f7a887-624f-4ff7-9cc8-4b012e48c9dc', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-08-26T10:00:00+00:00', '2026-08-26T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"admin","notes":"Údržba"}', NULL, '2026-08-24T17:35:54.941138+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-3', 0.0),
    ('2007f634-d4dd-4b67-ae38-904e632209b6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-09-03T08:00:00+00:00', '2026-09-03T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"admin","notes":"Údržba"}', NULL, '2026-08-24T17:37:48.817207+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', 0.0),
    ('c6d851fd-d0ca-46e7-a610-f71f3ecf0ad5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-08-24T18:00:00+00:00', '2026-08-24T20:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"admin","notes":"Údržba"}', NULL, '2026-08-24T17:40:08.769107+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-2', 0.0),
    ('b7c87346-6ced-4937-bd80-3cc0c547ebdc', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-25T18:00:00+00:00', '2026-08-25T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-24T18:16:50.671163+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', 17.0),
    ('675c03ee-9d24-4f00-afc2-4dac6e30b485', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-25T18:00:00+00:00', '2026-08-25T19:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia pre p\u00e1na Bartka"}', NULL, '2026-08-24T18:17:56.484556+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('fadb22cf-7725-47af-b3c5-d4fe6920894e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-08-26T08:00:00+00:00', '2026-08-26T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez telef\u00f3n"}', NULL, '2026-08-25T07:19:30.890414+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('6c6dccdb-3d53-406c-be10-63cbedd5a2c0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-08-25T13:00:00+00:00', '2026-08-25T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-25T07:21:46.838654+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('6579797c-529a-4102-bfa8-ff5b07e73a35', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-26T14:00:00+00:00', '2026-08-26T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia vytvoren\u00e1 cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-25T09:18:24.319485+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('38d9606f-9638-4fef-82c2-9722c0dc76a1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-26T15:00:00+00:00', '2026-08-26T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton"}', NULL, '2026-08-25T09:21:04.093415+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', 17.0),
    ('eab18b27-db2e-4940-8870-cd333eb631a1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Bartko', '', '2026-08-25T17:00:00+00:00', '2026-08-25T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia vytvoren\u00e1 telefonicky"}', NULL, '2026-08-25T12:44:32.504879+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('6d98767c-1777-4cdb-97f1-7b737f0ed16e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-25T17:00:00+00:00', '2026-08-25T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-3", "source": "voice-assistant", "notes": "Rezerv\u00e1cia vytvoren\u00e1 telefonicky"}', NULL, '2026-08-25T12:46:00.595477+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', NULL),
    ('da709295-b3be-4e07-8dbe-b5ba1168fe84', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamilo Bartko', '', '2026-08-25T18:00:00+00:00', '2026-08-25T19:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-3", "source": "voice-assistant", "notes": "Rezerv\u00e1cia vytvoren\u00e1 cez hlasov\u00e9ho asistenta."}', NULL, '2026-08-25T15:15:40.734739+00:00', NULL, NULL, 'badminton', 'badminton-3', NULL),
    ('8d9fe4eb-b956-4809-856b-0bde9f170944', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-26T17:00:00+00:00', '2026-08-26T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez telef\u00f3n"}', NULL, '2026-08-26T12:46:23.568326+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('a814c774-5bb2-4a94-9788-053fe4d12c55', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-27T16:00:00+00:00', '2026-08-27T17:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia vytvoren\u00e1 telefonicky"}', NULL, '2026-08-27T07:18:43.052361+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('6fe7cae0-79e4-467a-bd71-8cc74fb72f0b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-28T17:00:00+00:00', '2026-08-28T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-27T17:24:06.896443+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('a4cb3be6-dd3e-4657-a5db-fffa1ddbb5b4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-29T11:00:00+00:00', '2026-08-29T12:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-27T20:30:26.57413+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 11.0),
    ('c6fa6daa-c8c3-4d61-851c-8a6c8fef0903', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-29T08:00:00+00:00', '2026-08-29T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez telef\u00f3n"}', NULL, '2026-08-28T09:31:52.712342+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('27a71e75-163e-4734-b877-2005b3119e8c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-08-28T15:00:00+00:00', '2026-08-28T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez telef\u00f3n"}', NULL, '2026-08-28T10:03:38.134588+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', NULL),
    ('e3da30ee-da88-4c97-8397-4b2fb297a99d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-01T13:00:00+00:00', '2026-09-01T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-clay-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-31T09:48:54.608291+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', NULL),
    ('d1672ad0-6357-48df-b2ae-d0b13e481662', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-01T18:00:00+00:00', '2026-09-01T19:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-08-31T09:37:21.477743+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('3b86f870-ed52-49dd-9748-ea09b71f68e0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-01T17:00:00+00:00', '2026-09-01T18:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistentka"}', NULL, '2026-08-31T09:44:31.705682+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('2f03609e-f281-42ab-b69a-d9705f85d30f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-31T15:00:00+00:00', '2026-08-31T16:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia bedmintonu"}', NULL, '2026-08-31T11:50:21.854823+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('93fca382-b090-48c7-b2fe-685f57d9968d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-01T14:00:00+00:00', '2026-09-01T15:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez telef\u00f3n"}', NULL, '2026-08-31T07:08:45.51865+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', NULL),
    ('c2fc1a6f-6179-4bc6-8202-7239d9727bfd', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-01T17:00:00+00:00', '2026-09-01T18:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-08-31T09:44:32.110819+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', NULL),
    ('9704c641-ed3b-4666-b889-36509f9db2a6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-08-31T18:00:00+00:00', '2026-08-31T19:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-31T12:41:04.480131+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 17.0),
    ('b060a7b8-d3f8-47d3-9d7d-42f7cf34ee7d', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-01T19:00:00+00:00', '2026-09-01T20:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-31T12:44:34.757384+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 17.0),
    ('be4c864d-f658-49fd-9728-794889e35370', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-01T18:00:00+00:00', '2026-09-01T19:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton"}', NULL, '2026-08-31T12:44:38.67621+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', 17.0),
    ('a81e43a3-58eb-4fa5-b83d-285edf3dc802', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-01T18:00:00+00:00', '2026-09-01T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-31T12:46:32.717444+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 17.0),
    ('0e9ab747-24ba-4356-8ec2-7595fd2fe7d1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-01T16:00:00+00:00', '2026-09-01T17:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton"}', NULL, '2026-08-31T12:44:29.173081+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', 17.0),
    ('648cc77b-0b96-42f8-a299-2721d086025f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Lucia Bartková', '0907304358', '2026-09-01T13:00:00+00:00', '2026-09-01T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-31T17:35:38.366605+00:00', NULL, 'c5fe604e-6b4b-4a20-94da-c9e53cc8802a', 'badminton', 'badminton-1', 11.0),
    ('23a1778f-f0c6-40e6-8920-c2c8a0bcf3d9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-09-01T12:00:00+00:00', '2026-09-01T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-31T19:00:18.526875+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', 11.0),
    ('b0b696cf-2480-4d9a-b038-d824394a7ec3', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Google Test', '000000000000', '2026-09-01T15:00:00+00:00', '2026-09-01T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton"}', NULL, '2026-08-31T19:36:47.92796+00:00', NULL, '6392e3cf-c8a1-487b-adde-394e1a0ea31f', 'badminton', 'badminton-2', 15.0),
    ('52e19df4-9aa4-457e-82d4-aefce2a42906', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-01T17:00:00+00:00', '2026-09-01T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia badmintonu"}', NULL, '2026-09-01T07:33:11.675015+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 17.0),
    ('0593111b-b23f-4148-9751-01917c12db34', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-09-01T11:00:00+00:00', '2026-09-01T12:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia squashu"}', NULL, '2026-09-01T07:38:03.493334+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', 9.0),
    ('b826a62d-64d3-41c4-9148-906d458ec7e7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Ernest Hamburbažo', '+421915962068', '2026-09-01T14:00:00+00:00', '2026-09-01T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-09-01T07:58:25.247973+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('3647c7f8-5d4c-460d-ae59-eef84fa14f94', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Novák', '+421915962068', '2026-09-01T16:00:00+00:00', '2026-09-01T17:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-01T07:43:44.184555+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('f895804b-4858-4b05-8b13-9751f1e5adc9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-01T15:00:00+00:00', '2026-09-01T16:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton"}', NULL, '2026-08-31T12:44:21.097871+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 17.0),
    ('e2e0aa3e-dc94-4493-9ca4-cf0957db96b9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Laura Kanderová', '+421915962068', '2026-09-01T18:00:00+00:00', '2026-09-01T19:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-01T16:46:55.362258+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('762607ab-2850-4cd8-bde4-209b5e6b63d6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Trženský', '+421915962068', '2026-09-01T18:00:00+00:00', '2026-09-01T19:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-01T16:50:11.994338+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('27729659-26ca-447e-b8cf-cec60a3fe963', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Michal Zelina', '+421915962068', '2026-09-01T18:00:00+00:00', '2026-09-01T19:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-3", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-01T16:52:09.434989+00:00', NULL, NULL, 'badminton', 'badminton-3', NULL),
    ('9785fbc6-8282-413e-a4f6-93fe7edce468', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421915962068', '2026-09-03T15:00:00+00:00', '2026-09-03T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-02T07:49:08.885931+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('1051778b-4ee0-45bb-955a-b87a3b358009', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kanov Bartko', '+421915962068', '2026-09-02T15:00:00+00:00', '2026-09-02T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-02T07:56:43.759847+00:00', NULL, NULL, 'tennis', 'tennis-1', NULL),
    ('c3b40848-98ac-44a8-acbc-d31a91ddd2a7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-09-04T14:00:00+00:00', '2026-09-04T15:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton","multisportCardsCount":1}', NULL, '2026-09-03T06:58:11.227256+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 8.5),
    ('a30fb829-c32e-4a1b-821c-3cd752256f09', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Milan Tešbir', '+421915962068', '2026-09-03T10:00:00+00:00', '2026-09-03T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-03T07:07:22.962257+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('872ebf24-d349-4b82-9b3b-465fd2b83c5b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', 'client:web-user-1788419448', '2026-09-03T15:00:00+00:00', '2026-09-03T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-03T07:11:31.58024+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('c0b00a01-11ac-4d1b-9e97-227f697737bb', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Rastislav Harder', '+421915962068', '2026-09-04T17:00:00+00:00', '2026-09-04T18:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-03T17:16:00.296264+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('1df04e0e-7a92-4117-a455-649cd06455e4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Štefan Vrábel', '+421915962068', '2026-09-04T12:00:00+00:00', '2026-09-04T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-03T17:18:13.971006+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('c18a793c-648e-404a-a453-7e1fde7d89f5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Štefan Vrábel', '+421915962068', '2026-09-04T15:00:00+00:00', '2026-09-04T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-03T17:41:12.824534+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('4247edb4-4566-4f7c-b243-626576d43d5f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Diego Sánchez Serrano', '+421915962068', '2026-09-04T13:00:00+00:00', '2026-09-04T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-03T17:42:36.513946+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('9f0c3c4a-6a9e-49a5-a145-d63f1d9c4486', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-09-04T13:00:00+00:00', '2026-09-04T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-2","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-03T17:42:54.395338+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'squash', 'squash-2', 0.0),
    ('c89761ea-ea6e-4828-85f3-805fe346151e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-09-04T13:00:00+00:00', '2026-09-04T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-3","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-03T17:42:57.508008+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'squash', 'squash-3', 0.0),
    ('e8712b1a-ee34-42ab-9b5e-24b23b327a8c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-09-04T13:00:00+00:00', '2026-09-04T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-4","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-03T17:43:00.752796+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'squash', 'squash-4', 0.0),
    ('cb818391-9ab4-40c3-b5ff-cdd587ed5349', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Oliver Barabás', '+421915962068', '2026-09-04T14:00:00+00:00', '2026-09-04T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-03T17:44:08.598134+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('6e7254c0-ae67-4189-a190-ae24a75ddd90', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Villy Šartenleib', '+421915962068', '2026-09-04T12:00:00+00:00', '2026-09-04T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-03T17:45:20.738461+00:00', NULL, NULL, 'squash', 'squash-1', NULL),
    ('12f0b3ea-85d1-4606-ad33-971338c71ad9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Jesenjak', '+421915962068', '2026-09-05T10:00:00+00:00', '2026-09-05T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-04T09:38:40.35094+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('a532c7a7-84ea-4ad6-9713-774f6753fdfd', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-09-06T08:00:00+00:00', '2026-09-06T09:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-05T06:23:58.321391+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('8cd6cb69-61e3-49f2-bd16-ea1255ccbc5a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-09-06T10:00:00+00:00', '2026-09-06T11:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-05T06:33:24.799004+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', 0.0),
    ('857d05e0-2d5a-41fc-96e1-c1600c08e32f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jankovič Ján', '+421900123456', '2026-09-06T18:00:00+00:00', '2026-09-06T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton","multisportCardsCount":1}', NULL, '2026-09-05T12:00:52.959736+00:00', NULL, 'a9f2b336-721f-4046-830d-b6fea9863703', 'badminton', 'badminton-1', 6.5),
    ('17e62401-4201-4e68-9728-4bfb89a38d8c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter', '', '2026-09-06T08:00:00+00:00', '2026-09-06T09:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-05T18:36:16.756287+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('a63bb726-4df9-4f27-80de-a86b3a2fa4bc', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-09-06T15:00:00+00:00', '2026-09-06T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-06T08:27:47.385372+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-1', 0.0),
    ('41c81bf0-48a9-40a5-8059-337b7c1e0992', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-09-06T11:00:00+00:00', '2026-09-06T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-06T09:57:13.464534+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-4', 22.0),
    ('10c35109-40de-467e-88e2-4ef1d921f88f', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Admin User', '+421900000000', '2026-09-06T12:00:00+00:00', '2026-09-06T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-06T09:58:03.353654+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-3', 0.0),
    ('b4b40b50-cb14-4832-9a19-df2e9ae1e029', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Údržba', NULL, '2026-09-06T17:00:00+00:00', '2026-09-06T18:00:00+00:00', 'blocked', NULL, '{"courtId":"badminton-6","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-06T10:07:01.450374+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-6', 0.0),
    ('36e2827e-77f6-4a5e-9db5-68db01ee44d0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-09-06T14:00:00+00:00', '2026-09-06T15:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-06T10:08:36.347071+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', 11.0),
    ('959196e3-9c2b-4183-9ab4-476980675fd5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Milan Polanský', '+421900543216564', '2026-09-06T18:00:00+00:00', '2026-09-06T19:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-06T10:31:00.210031+00:00', NULL, 'e616dfd7-3777-4571-a831-0d47ccfc7618', 'badminton', 'badminton-3', 11.0),
    ('77d3dc0b-0f8e-4372-80b1-78dd944d85ed', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-09-07T13:00:00+00:00', '2026-09-07T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-06T10:40:14.353093+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', 11.0),
    ('9b1c78bd-fa92-4df9-910f-2e4e18914f23', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-09-08T15:00:00+00:00', '2026-09-08T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-06T10:48:32.383881+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', 17.0),
    ('29f4c9b4-2a43-42a0-8570-9958506d2931', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-09-07T17:00:00+00:00', '2026-09-07T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-09-06T11:07:10.780597+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'squash', 'squash-1', 13.0),
    ('dab8d753-9a51-4502-99d4-58e64690ea47', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '0905012054', '2026-09-06T15:00:00+00:00', '2026-09-06T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistentka"}', NULL, '2026-09-06T11:19:22.506463+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', 15.0),
    ('b2e2f85e-a955-440d-a862-b177a0fc6d3b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-09-06T13:00:00+00:00', '2026-09-06T15:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-06T09:57:20.264986+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-4', 22.0),
    ('4bf84aad-c0d5-4125-b7a3-1ecff0efca62', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '0905012054', '2026-09-06T18:00:00+00:00', '2026-09-06T19:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez AI hlasov\u00e9ho asistenta"}', NULL, '2026-09-06T11:31:32.876301+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', 11.0),
    ('5cffc70f-f03c-4362-a2f5-9b6f5953c5f4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Lucia Bartková', '0915962068', '2026-09-06T17:00:00+00:00', '2026-09-06T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-09-06T14:06:52.910519+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('b01a293d-ad0c-40a4-8905-94d888f3f099', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Lucia Bartková', '0915962068', '2026-09-07T17:00:00+00:00', '2026-09-07T18:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-06T14:15:25.361346+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('3dea5424-689a-4554-aafb-be83f1c90781', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Novy Pouzivatel', '00421915962067', '2026-09-07T16:00:00+00:00', '2026-09-07T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton","multisportCardsCount":1}', NULL, '2026-09-06T21:00:06.272782+00:00', NULL, '5009e9ad-5bff-4cf5-8190-fd26c576cac6', 'badminton', 'badminton-2', 8.5),
    ('c04053f2-3c11-4c1f-96db-cdec54b59ad6', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Novy Pouzivatel', '00421915962067', '2026-09-07T12:00:00+00:00', '2026-09-07T13:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton","multisportCardsCount":2}', NULL, '2026-09-06T21:00:17.083855+00:00', NULL, '5009e9ad-5bff-4cf5-8190-fd26c576cac6', 'badminton', 'badminton-4', 0.0),
    ('ccbd79a0-f30a-4051-9fd8-53af1772bb6b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Novy Pouzivatel', '00421915962067', '2026-09-07T19:00:00+00:00', '2026-09-07T20:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-5","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-06T21:00:27.38226+00:00', NULL, '5009e9ad-5bff-4cf5-8190-fd26c576cac6', 'badminton', 'badminton-5', 17.0),
    ('5b5d659d-6580-498a-9bcc-93ab990d462c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-09-07T15:00:00+00:00', '2026-09-07T16:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-06T21:02:11.477502+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', 17.0),
    ('d51a772e-0a1e-4d18-b6bd-8d7aaffd762c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-09-07T14:00:00+00:00', '2026-09-07T15:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-5","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-06T21:02:19.551229+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-5', 17.0),
    ('03355988-2c89-4ba4-a1bb-7be4cdeefb65', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-09-07T15:00:00+00:00', '2026-09-07T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-4","source":"web","notes":"Bedminton","multisportCardsCount":1}', NULL, '2026-09-06T21:02:37.526363+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-4', 17.0),
    ('bfcb2712-b0ea-4c98-8428-9836ec9aaba8', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-07T14:00:00+00:00', '2026-09-07T15:00:00+00:00', 'cancelled', NULL, '{"courtId": "squash-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia squashu pre Kamil Bartko"}', NULL, '2026-09-06T21:08:30.74925+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', 13.0),
    ('b632785d-7b53-4a29-a309-28ee98ac2fc1', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-09-08T10:00:00+00:00', '2026-09-08T11:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-06T21:43:53.992389+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 11.0),
    ('57cbc9a1-0afb-4e7c-ba67-5aabe59fa701', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-08T12:00:00+00:00', '2026-09-08T13:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-06T21:46:31.994563+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 11.0),
    ('a7dfe1e7-a5e9-4a46-9c19-2e60bc563638', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-09-07T18:00:00+00:00', '2026-09-07T19:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-06T21:02:16.136531+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 17.0),
    ('d0533bd8-b6a9-4fdf-a976-d690380324fd', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '0915962068', '2026-09-08T12:00:00+00:00', '2026-09-08T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-07T09:55:04.721706+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 11.0),
    ('1f1f335f-7257-4459-a353-4fb7aa0cf95a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-08T13:00:00+00:00', '2026-09-08T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"squash-1","source":"web","notes":"Squash","multisportCardsCount":0}', NULL, '2026-09-07T18:55:36.043786+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'squash', 'squash-1', 9.0),
    ('8b342c17-bd24-43b6-8693-17a961e9d398', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-08T16:00:00+00:00', '2026-09-08T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"tennis-clay-1","source":"web","notes":"Tenis antuka","multisportCardsCount":0}', NULL, '2026-09-07T18:55:44.140011+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', 13.0),
    ('8fc31770-00ef-4c83-be5e-ed7dcab9d15e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '0915962068', '2026-09-08T15:00:00+00:00', '2026-09-08T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-clay-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia antuka 17:00"}', NULL, '2026-09-07T18:58:46.576227+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'tennis-clay', 'tennis-clay-1', 13.0),
    ('a44338ca-e986-4c87-8f64-2ff773696d84', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '0915962068', '2026-09-09T10:00:00+00:00', '2026-09-09T11:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-09-07T19:16:11.702953+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 11.0),
    ('74f1754d-6200-45de-b240-89adcd4cda0e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '0915962068', '2026-09-09T12:00:00+00:00', '2026-09-09T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-09-07T19:17:55.904538+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 11.0),
    ('d60c52e3-c875-4966-ad89-37ee88c098f7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jozef Holbovič', '0915962068', '2026-09-10T12:00:00+00:00', '2026-09-10T13:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia bedmintonu"}', NULL, '2026-09-08T07:30:36.563418+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('b2601bc1-b2b0-472a-90f3-ea1351ecf38e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Milankovič', '0915962068', '2026-09-09T14:00:00+00:00', '2026-09-09T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-09T08:02:41.133699+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('22a00d93-6f2d-43b3-93af-c4155576f589', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Údržba', NULL, '2026-09-09T11:00:00+00:00', '2026-09-09T12:00:00+00:00', 'blocked', NULL, '{"courtId":"badminton-2","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-09T08:22:44.905472+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-2', 0.0),
    ('33755236-752c-4ed7-81de-448f26e64d16', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Údržba', NULL, '2026-09-09T11:00:00+00:00', '2026-09-09T20:00:00+00:00', 'blocked', NULL, '{"courtId":"badminton-3","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-09T08:23:03.633107+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-3', 0.0),
    ('901a194e-71a3-4c67-b3e1-97e3009d820a', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Údržba', NULL, '2026-09-09T15:00:00+00:00', '2026-09-09T20:00:00+00:00', 'blocked', NULL, '{"courtId":"badminton-4","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-09T08:23:19.916238+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-4', 0.0),
    ('04540c84-3df7-4c41-83c6-4f32596f402b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Jan Jankovič', '0915962068', '2026-09-09T14:00:00+00:00', '2026-09-09T15:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-2", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-09-09T08:46:12.759732+00:00', NULL, NULL, 'badminton', 'badminton-2', NULL),
    ('89ee42b4-48dd-4d68-a394-9990f81e24f9', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Petr Nový Vrt', '0915962068', '2026-09-09T11:00:00+00:00', '2026-09-09T12:00:00+00:00', 'cancelled', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-09T08:47:45.080588+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('e694312d-5c92-4654-b089-2cb23684b2b4', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko, Slovakia GREPA', '0915962068', '2026-09-10T16:00:00+00:00', '2026-09-10T17:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Reservation via voice call"}', NULL, '2026-09-10T08:21:58.509598+00:00', NULL, NULL, 'badminton', 'badminton-1', NULL),
    ('9d6b06c3-4395-42bd-91bf-8d9800c68175', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-09-10T16:00:00+00:00', '2026-09-10T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-10T10:28:39.554216+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-2', 17.0),
    ('3c7ecc3f-6e0f-46bf-8ff2-164327802c83', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-09-10T13:00:00+00:00', '2026-09-10T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-10T10:28:43.872483+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', 11.0),
    ('5eb0d41b-3980-4e63-9fa4-5cb6207f173b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-09-10T18:00:00+00:00', '2026-09-10T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-10T10:28:49.248888+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', 17.0),
    ('d1b8ac4e-c4e1-4d2a-99f6-85404c126ae0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-09-10T13:00:00+00:00', '2026-09-10T14:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-10T10:28:55.456731+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 11.0),
    ('9a0cf690-2a64-4195-9e00-b31b9d59af3e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '0915962068', '2026-09-10T15:00:00+00:00', '2026-09-10T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-09-10T10:32:00.010016+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 17.0),
    ('852c7d13-1b51-43c4-9f5b-2d27d46732b7', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-09-10T17:00:00+00:00', '2026-09-10T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-10T13:30:59.037502+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', 17.0),
    ('94890d79-b7c0-4367-8d55-1cafdfd82d88', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-09-11T18:00:00+00:00', '2026-09-11T19:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-10T13:35:37.561307+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-1', 17.0),
    ('ce23ad48-f671-45c2-8e3f-f009ae9e73c0', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '0905012054', '2026-09-11T13:00:00+00:00', '2026-09-11T14:00:00+00:00', 'confirmed', NULL, '{"courtId": "tennis-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00fa asistentku"}', NULL, '2026-09-10T15:15:19.320898+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'tennis', 'tennis-1', 15.0),
    ('db629082-0a31-4b9a-8733-034ac91925d5', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+42191596206', '2026-09-11T16:00:00+00:00', '2026-09-11T17:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-6","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-10T18:02:31.873525+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-6', 17.0),
    ('e7ce3c54-bea9-4ff8-b00d-3b1fcd38f430', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-11T17:00:00+00:00', '2026-09-11T18:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-3","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-10T19:03:30.812222+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-3', 17.0),
    ('40f3c985-5136-404f-b0f5-282384302d70', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Údržba', NULL, '2026-09-11T15:00:00+00:00', '2026-09-11T16:00:00+00:00', 'blocked', NULL, '{"courtId":"badminton-2","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-10T19:06:09.164594+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-2', 0.0),
    ('bba50aad-4881-49d2-a0c8-d8d4179bc674', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Údržba', NULL, '2026-09-11T16:00:00+00:00', '2026-09-11T19:00:00+00:00', 'blocked', NULL, '{"courtId":"badminton-2","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-10T19:06:24.429697+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-2', 0.0),
    ('f3134da4-e25e-4c1e-a775-1afc47c5535e', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Údržba', NULL, '2026-09-11T11:00:00+00:00', '2026-09-11T12:00:00+00:00', 'blocked', NULL, '{"courtId":"badminton-3","source":"admin","notes":"Údržba","multisportCardsCount":0}', NULL, '2026-09-11T08:40:14.731667+00:00', NULL, 'c6b90673-97cf-4ec2-a9df-644596a2a05a', 'badminton', 'badminton-3', 0.0),
    ('746fbc7a-b2f8-464e-8900-075566da917b', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '+421915962068', '2026-09-12T18:00:00+00:00', '2026-09-12T19:00:00+00:00', 'cancelled', NULL, '{"courtId":"badminton-1","source":"web","notes":"Bedminton","multisportCardsCount":0}', NULL, '2026-09-11T13:49:49.792816+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 11.0),
    ('d324a0cc-2ab1-4ad6-ac97-474123811e8c', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Kamil Bartko', '0915962068', '2026-09-12T15:00:00+00:00', '2026-09-12T16:00:00+00:00', 'confirmed', NULL, '{"courtId": "badminton-1", "source": "voice-assistant", "notes": "Rezerv\u00e1cia cez hlasov\u00e9ho asistenta"}', NULL, '2026-09-11T13:56:02.93842+00:00', NULL, '8b86a015-0924-4780-8264-b9beabc3519f', 'badminton', 'badminton-1', 11.0),
    ('2f034e9b-38db-4f55-a14c-1fac5af735cc', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', NULL, NULL, 'Peter Kaľavský', '+421905012054', '2026-09-12T11:00:00+00:00', '2026-09-12T12:00:00+00:00', 'confirmed', NULL, '{"courtId":"badminton-2","source":"web","notes":"Bedminton","multisportCardsCount":1}', NULL, '2026-09-11T17:30:41.506012+00:00', NULL, 'e9e3b12e-5e08-46ce-bd6d-9da8096db37d', 'badminton', 'badminton-2', 5.5)
ON CONFLICT DO NOTHING;

-- Data for public.calendar_connections (3 rows)
INSERT INTO public.calendar_connections (id, tenant_id, provider, calendar_id, access_token, refresh_token, token_expiry)
VALUES
    ('b68f6cb8-1db3-4c17-9147-1006a0fbf3b2', 'ac3b439d-a446-4d67-abf7-5e04f58647fe', 'google', 'primary', 'REDACTED_ACCESS_TOKEN', 'REDACTED_REFRESH_TOKEN', '2026-02-25T17:44:03.502+00:00'),
    ('56b7fef5-ce33-4f9a-a6ca-664cb83976fe', '6076dbf2-8f14-432d-b27d-504ae30b94b2', 'google', 'primary', 'REDACTED_ACCESS_TOKEN', 'REDACTED_REFRESH_TOKEN', '2026-04-04T08:29:24.526+00:00'),
    ('e237e8f0-d577-4607-9b92-8e2c2d973a12', '595cbb6c-1019-41ae-b1c2-a60c13c8dcdf', 'google', 'primary', 'REDACTED_ACCESS_TOKEN', 'REDACTED_REFRESH_TOKEN', '2026-07-13T07:00:12.311057+00:00')
ON CONFLICT DO NOTHING;

COMMIT;

-- >>> END OF migrations/000_core_from_supabase.sql <<<

-- >>> START OF migrations/001_wallet_ledger.sql <<<
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

-- >>> END OF migrations/001_wallet_ledger.sql <<<

-- >>> START OF migrations/002_wallet_bookings.sql <<<
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

-- >>> END OF migrations/002_wallet_bookings.sql <<<

-- >>> START OF migrations/003_wallet_manual_adjustment.sql <<<
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

-- >>> END OF migrations/003_wallet_manual_adjustment.sql <<<

-- >>> START OF migrations/004_wallet_payment_hotfix.sql <<<
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

-- >>> END OF migrations/004_wallet_payment_hotfix.sql <<<

