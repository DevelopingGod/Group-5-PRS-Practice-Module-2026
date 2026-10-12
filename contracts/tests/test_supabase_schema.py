import re
from pathlib import Path

from binbuddy_contracts.constants import DAILY_POINTS_CAP, EMBEDDING_DIM
from binbuddy_contracts.enums import Channel, Condition, Material

MIGRATIONS = Path(__file__).resolve().parents[2] / "supabase" / "migrations"


def _sql() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(MIGRATIONS.glob("*.sql")))


def _enum_values(sql: str, name: str) -> list[str]:
    match = re.search(rf"create type {name} as enum \((.*?)\);", sql, re.S)
    assert match, f"enum {name} not found in migrations"
    return re.findall(r"'([^']+)'", match.group(1))


def test_enums_match_contracts() -> None:
    sql = _sql()
    assert _enum_values(sql, "material") == [m.value for m in Material]
    assert _enum_values(sql, "condition") == [c.value for c in Condition]
    assert _enum_values(sql, "channel") == [c.value for c in Channel]


def test_vector_dimension_matches_embedding_dim() -> None:
    dims = {int(d) for d in re.findall(r"vector\((\d+)\)", _sql())}
    assert dims == {EMBEDDING_DIM}


def test_daily_cap_matches_contract() -> None:
    assert f"least(sum(points), {DAILY_POINTS_CAP})" in _sql()
