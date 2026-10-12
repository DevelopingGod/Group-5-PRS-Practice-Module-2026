from pathlib import Path

import pytest

from binbuddy_contracts.dataset import COLUMNS, validate_labels
from binbuddy_contracts.policy import SitePolicy, load_policies

POLICIES = Path(__file__).resolve().parents[2] / "policies"


@pytest.fixture(scope="module")
def hdb() -> SitePolicy:
    return load_policies(POLICIES)["hdb_default"]


def row(**overrides: str) -> dict[str, str]:
    base = {
        "photo_id": "ws2-0001-a",
        "item_id": "ws2-0001",
        "collector": "ws2",
        "capture": "as_used",
        "drive_path": "local/ws2-0001/ws2-0001-a.jpg",
        "material": "plastic",
        "condition": "item_inside",
        "hdb_bin_key": "general_waste",
        "q1_recycling_symbol": "yes",
        "q2_cleaned": "no",
        "q3_sitting_time": "same_day",
        "split": "train",
        "notes": "banana peel inside a PP container",
    }
    base.update(overrides)
    assert tuple(base) == COLUMNS
    return base


def cleaned_twin() -> dict[str, str]:
    return row(
        photo_id="ws2-0001-b",
        drive_path="local/ws2-0001/ws2-0001-b.jpg",
        capture="cleaned",
        condition="clean",
        hdb_bin_key="blue_bin",
        q2_cleaned="yes",
        q3_sitting_time="na",
        notes="",
    )


def test_paired_capture_is_valid(hdb: SitePolicy) -> None:
    assert validate_labels([row(), cleaned_twin()], hdb) == []


def test_split_leakage_is_rejected(hdb: SitePolicy) -> None:
    twin = cleaned_twin()
    twin["split"] = "test"
    errors = validate_labels([row(), twin], hdb)
    assert any("share split" in e for e in errors)


def test_label_must_agree_with_policy(hdb: SitePolicy) -> None:
    errors = validate_labels([row(hdb_bin_key="blue_bin")], hdb)
    assert any("disagrees with the HDB policy" in e for e in errors)


def test_cleaned_capture_must_be_clean(hdb: SitePolicy) -> None:
    errors = validate_labels([row(capture="cleaned")], hdb)
    assert any("'cleaned' capture" in e for e in errors)


def test_duplicate_photo_rejected(hdb: SitePolicy) -> None:
    errors = validate_labels([row(), row()], hdb)
    assert any("duplicate photo_id" in e for e in errors)


def test_bad_enum_value_rejected(hdb: SitePolicy) -> None:
    errors = validate_labels([row(material="cardboard")], hdb)
    assert any("material" in e for e in errors)
