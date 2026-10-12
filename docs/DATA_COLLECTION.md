# Data collection guide (everyone)

Photos live on **Google Drive**, in the team folder `BinBuddy_Data` (shared with the four members only). The repository holds only the manifest `data/labels.csv`, which CI validates on every push.

## Drive layout and names

```
BinBuddy_Data/
  local/
    ws2-0017/
      ws2-0017-a.jpg    # as used (with residue, liquid or item inside)
      ws2-0017-b.jpg    # after emptying and rinsing (paired capture)
      ws2-0017-c.jpg    # optional extra angle
```

- `item_id` = your workstream + a 4-digit number you increment yourself (`ws1-0001`, `ws1-0002`, ...). No clashes, because each person has their own prefix.
- `photo_id` = `item_id` + a letter.
- In Colab, add the shared folder to "My Drive" as a shortcut named `BinBuddy_Data` (see `ml/binbuddy_ml/paths.py`).

## Privacy rules

- Photograph the **item only**: no faces, hands with rings or tattoos, letters, parcels with addresses, or screens.
- Turn **off location tagging** in your phone camera before collecting (phone photos embed GPS in EXIF).
- Never commit photos to GitHub; CI rejects image files.

## One row per photo in `data/labels.csv`

| Column | Values |
|---|---|
| `capture` | `as_used`, `cleaned` (a `cleaned` photo must have condition `clean`) |
| `material` | paper, plastic, glass, metal, beverage_carton, styrofoam, e_waste, textile, food_waste, other_general |
| `condition` | clean, food_residue, leftover_liquid, item_inside |
| `hdb_bin_key` | Must equal what `policies/hdb_default.yaml` says for that material and condition (CI checks) |
| `q1_recycling_symbol`, `q2_cleaned` | yes, no, not_sure (`q2` may be `na`) |
| `q3_sitting_time` | under_1_hour, same_day, more_than_1_day, or `na` |
| `split` | train, val, test, field. **All photos of one item share the same split** (CI checks) |

Default split rule (keeps splits balanced and leak-free): look at the last digit of the item number: `0` = field, `1` = test, `2` = val, anything else = train. That reserves about 10% of items for the field test; with ~500+ items this meets the proposal's 50-item minimum.

## Labelling examples (proposal 1.3)

| Item | material | condition | hdb_bin_key |
|---|---|---|---|
| Banana peel inside a PP container | plastic | item_inside | general_waste |
| Milk carton with milk left | beverage_carton | leftover_liquid | general_waste |
| Same carton, rinsed | beverage_carton | clean | blue_bin |
| Oily pizza box | paper | food_residue | general_waste |
| Styrofoam box with a recycling symbol | styrofoam | clean | general_waste |
| Old phone charger, light bulb | e_waste | clean | e_waste_point |
| Old T-shirt | textile | clean | donation |
| Clean aluminium can | metal | clean | blue_bin |

Check your rows before pushing: `uv run python scripts/validate_data.py`.
