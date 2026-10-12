-- Bin Buddy initial schema. Owner: WS4. Enum values and the vector size are checked
-- against the contracts package by contracts/tests/test_supabase_schema.py.

create extension if not exists vector;

create type material as enum (
  'paper', 'plastic', 'glass', 'metal', 'beverage_carton',
  'styrofoam', 'e_waste', 'textile', 'food_waste', 'other_general'
);

create type condition as enum ('clean', 'food_residue', 'leftover_liquid', 'item_inside');

create type channel as enum ('recycle', 'general_waste', 'e_waste', 'donation', 'abstain');

-- One row per signed-in (anonymous) user; only a pseudonym is stored.
create table profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  pseudonym text not null unique check (char_length(pseudonym) between 3 and 24),
  created_at timestamptz not null default now()
);

-- Demonstration bins (QR payload: bin_id, site_policy, bin_key, location_label).
create table bins (
  bin_id text primary key,
  site_policy text not null,
  bin_key text not null,
  location_label text not null
);

create table scans (
  id uuid primary key,
  user_id uuid not null references profiles (id) on delete cascade,
  site_policy text not null,
  material material,
  condition condition,
  channel channel not null,
  bin_key text,
  confidence real not null check (confidence between 0 and 1),
  questions jsonb not null default '[]'::jsonb,
  model_revision text not null,
  embedding vector(256),
  phash text check (phash ~ '^[0-9a-f]{16}$'),
  created_at timestamptz not null default now()
);

create table disposals (
  id uuid primary key default gen_random_uuid(),
  scan_id uuid not null unique references scans (id) on delete cascade, -- one reward per item
  user_id uuid not null references profiles (id) on delete cascade,
  bin_id text not null references bins (bin_id),
  similarity real not null,
  passed boolean not null,
  points integer not null default 0 check (points >= 0),
  phash text not null check (phash ~ '^[0-9a-f]{16}$'),
  created_at timestamptz not null default now()
);

-- Leaderboard: pseudonyms and points only, with the daily cap (DAILY_POINTS_CAP = 10) applied.
create view leaderboard as
select p.pseudonym, coalesce(sum(d.day_points), 0)::integer as total_points
from profiles p
left join (
  select user_id, least(sum(points), 10) as day_points
  from disposals
  where passed
  group by user_id, (created_at at time zone 'Asia/Singapore')::date
) d on d.user_id = p.id
group by p.pseudonym;

-- Row Level Security: users see and write only their own rows.
alter table profiles enable row level security;
alter table scans enable row level security;
alter table disposals enable row level security;
alter table bins enable row level security;

create policy "own profile" on profiles for all using (auth.uid() = id) with check (auth.uid() = id);
create policy "own scans" on scans for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own disposals read" on disposals for select using (auth.uid() = user_id);
create policy "bins are public" on bins for select using (true);
-- Disposals are inserted only by the server (service role) after /verify passes.
