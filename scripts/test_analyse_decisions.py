import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from analyse_decisions import (
    bootstrap_weights,
    expected_calibration_error,
    intervals,
    item_count,
    main,
    noisy,
    outcomes,
    paired_difference,
    perturbed,
    tempered,
)
from simulate_gate import load_items, make_cases
from test_simulate_gate import HDB, write_predictions

from app import decision, likelihoods
from binbuddy_contracts.enums import Material


@pytest.mark.parametrize("rate", [0.0, 0.2, 1.0])
def test_noisy_tables_are_distributions(rate: float) -> None:
    tables = noisy(likelihoods.all_rows(), rate)
    for name, row in tables.items():
        assert sum(row.values()) == pytest.approx(1.0)
        if rate == 0.0:
            assert row == pytest.approx(likelihoods.all_rows()[name])
        if rate == 1.0:
            assert all(v == pytest.approx(1.0 / len(row)) for v in row.values())


def test_perturbed_tables_are_distributions() -> None:
    same = perturbed(likelihoods.all_rows(), 0.0, random.Random(0))
    for name, row in same.items():
        assert row == pytest.approx(likelihoods.all_rows()[name])
    for row in perturbed(likelihoods.all_rows(), 0.5, random.Random(0)).values():
        assert sum(row.values()) == pytest.approx(1.0)
        assert min(row.values()) > 0.0


def test_other_worlds_keep_the_same_true_conditions(tmp_path: Path) -> None:
    # Every setting must see the same items and conditions; only the answers differ.
    write_predictions(tmp_path / "p.csv")
    items = load_items(tmp_path / "p.csv", "val")
    base = make_cases(items, 30, 0.5, 3)
    other = make_cases(items, 30, 0.5, 3, noisy(likelihoods.all_rows(), 1.0))
    assert [c.condition for c in base] == [c.condition for c in other]
    assert [c.answers for c in base] != [c.answers for c in other]


def test_temperature_sharpens_and_flattens(tmp_path: Path) -> None:
    write_predictions(tmp_path / "p.csv")
    item = load_items(tmp_path / "p.csv", "val")[0]
    top = item.probabilities[Material.PLASTIC]
    assert tempered([item], 1.0)[0].probabilities[Material.PLASTIC] == pytest.approx(top)
    assert tempered([item], 0.5)[0].probabilities[Material.PLASTIC] > top
    assert tempered([item], 2.0)[0].probabilities[Material.PLASTIC] < top


def test_calibration_error() -> None:
    assert expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    assert expected_calibration_error([0.9, 0.9], [False, False]) == pytest.approx(0.9)
    assert expected_calibration_error([0.75] * 4, [True, True, True, False]) == pytest.approx(0.0)


def test_bootstrap_brackets_the_estimate_and_pairs_cancel(tmp_path: Path) -> None:
    write_predictions(tmp_path / "p.csv")
    cases = make_cases(load_items(tmp_path / "p.csv", "val"), 25, 0.5, 0)
    outs = outcomes(cases, HDB, decision.DEFAULT_GATE)
    weights = bootstrap_weights(item_count(cases), 200, 0)
    accuracy = sum(o.correct for o in outs) / len(outs)
    lo, hi = intervals(cases, outs, weights)["verdict_accuracy"]
    assert lo <= accuracy <= hi
    same = paired_difference(cases, outs, outs, weights)
    assert all(v == pytest.approx([0.0, 0.0, 0.0]) for v in same.values())


def test_main_writes_results_tables_and_figures(tmp_path: Path) -> None:
    write_predictions(tmp_path / "p.csv")
    out = tmp_path / "out"
    args = ["--predictions", str(tmp_path / "p.csv"), "--out", str(out)]
    assert main([*args, "--repeats", "3", "--worlds", "2", "--resamples", "20"]) == 0
    results = json.loads((out / "results.json").read_text())
    assert len([k for k in results if k[0].isdigit()]) == 9
    tables = (out / "tables.md").read_text()
    assert "## 9. Condition prior" in tables
    svgs = sorted(out.glob("*.svg"))
    assert len(svgs) == 3
    for svg in svgs:
        ET.fromstring(svg.read_text())  # well-formed
