import json
from pathlib import Path

import pytest
from estimate_likelihoods import estimate, observations, smoothed

from app import likelihoods
from binbuddy_contracts.dataset import LabelRow


def row(n: int, material: str, condition: str, q1: str, q2: str, q3: str) -> LabelRow:
    return LabelRow(
        photo_id=f"ws3-{n:04d}-a",
        item_id=f"ws3-{n:04d}",
        collector="ws3",
        capture="as_used",
        drive_path=f"local/ws3-{n:04d}/ws3-{n:04d}-a.jpg",
        material=material,
        condition=condition,
        hdb_bin_key="general_waste" if condition != "clean" else "blue_bin",
        q1_recycling_symbol=q1,
        q2_cleaned=q2,
        q3_sitting_time=q3,
        split="train",
    )


def test_observations_group_answers_by_row() -> None:
    rows = [
        row(1, "plastic", "clean", "yes", "yes", "under_1_hour"),
        row(2, "plastic", "food_residue", "yes", "no", "same_day"),
        row(3, "paper", "food_residue", "no", "na", "na"),
    ]
    seen = observations(rows)
    assert seen["q1/plastic"] == ["yes", "yes"]
    assert seen["q2/food_residue"] == ["no"]  # "na" skipped
    assert seen["q3/residue_perishable"] == ["same_day"]
    assert "q3/residue_dry" not in seen


@pytest.mark.parametrize("n", [5, 200, 10_000])
def test_smoothing_never_rules_an_answer_out(n: int) -> None:
    # Many identical answers leave the others below the floor before it is applied.
    hand_set = likelihoods.all_rows()["q1/plastic"]
    row_ = smoothed(["yes"] * n, list(hand_set))
    likelihoods.check_row("q1/plastic", row_, hand_set)
    assert row_["no"] == row_["not_sure"] >= likelihoods.LIKELIHOOD_FLOOR


def test_rows_need_enough_answers() -> None:
    rows = [row(i, "plastic", "clean", "yes", "yes", "under_1_hour") for i in range(1, 5)]
    assert estimate(rows, min_count=5) == {}
    counted = estimate(rows, min_count=4)
    assert set(counted) == {"q1/plastic", "q2/clean", "q3/clean"}
    for name, r in counted.items():
        likelihoods.check_row(name, r, likelihoods.all_rows()[name])


def test_counted_rows_replace_hand_set_rows(tmp_path: Path) -> None:
    rows = {k: dict(v) for k, v in likelihoods.all_rows().items()}
    path = tmp_path / "counted.json"
    path.write_text(json.dumps({"rows": {"q2/clean": {"yes": 0.9, "no": 0.05, "not_sure": 0.05}}}))
    assert likelihoods.load_counted(path, rows) == ["q2/clean"]
    assert rows["q2/clean"]["yes"] == 0.9
    assert rows["q2/food_residue"] == likelihoods.all_rows()["q2/food_residue"]


def test_bad_counted_rows_are_rejected(tmp_path: Path) -> None:
    rows = {k: dict(v) for k, v in likelihoods.all_rows().items()}
    path = tmp_path / "counted.json"
    path.write_text(json.dumps({"rows": {"q2/clean": {"yes": 1.0, "no": 0.0, "not_sure": 0.0}}}))
    with pytest.raises(ValueError, match="below"):
        likelihoods.load_counted(path, rows)
    assert likelihoods.load_counted(tmp_path / "missing.json", rows) == []
