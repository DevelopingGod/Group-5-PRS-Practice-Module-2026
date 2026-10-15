-- Demonstration bins with Bin Buddy QR codes. Owner: WS4.
-- Must match apps/web/src/lib/demo-bins.ts (checked by apps/web/src/lib/demo-bins.test.ts).
insert into bins (bin_id, site_policy, bin_key, location_label) values
  ('demo-hdb-01', 'hdb_default', 'blue_bin', 'Demo blue recycling bin'),
  ('demo-hdb-02', 'hdb_default', 'general_waste', 'Demo general waste bin'),
  ('demo-campus-01', 'campus_example', 'paper_bin', 'Demo campus paper bin'),
  ('demo-campus-02', 'campus_example', 'plastic_bin', 'Demo campus plastic bin'),
  ('demo-campus-03', 'campus_example', 'cans_glass_bin', 'Demo campus cans and glass bin'),
  ('demo-campus-04', 'campus_example', 'general_waste', 'Demo campus general waste bin')
on conflict (bin_id) do update set
  site_policy = excluded.site_policy,
  bin_key = excluded.bin_key,
  location_label = excluded.location_label;
