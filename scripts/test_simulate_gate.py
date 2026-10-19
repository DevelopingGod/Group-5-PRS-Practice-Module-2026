import csv
import json
from pathlib import Path

from simulate_gate import load_items, main, make_cases, run_gate, sweep

from app import decision
from binbuddy_contracts.constants import DEFAULT_SITE_POLICY
from binbuddy_contracts.enums import Condition, Material
from binbuddy_contracts.policy import load_policies
from binbuddy_ml.labels import REPO_ROOT

HDB = load_policies(REPO_ROOT / "policies")[DEFAULT_SITE_POLICY]


def write_predictions(path: Path) -> None:
    rows = [
        ("val", Material.PLASTIC, Material.PLASTIC, 0.9),
        ("val", Material.STYROFOAM, Material.PLASTIC, 0.5),
        ("val", Material.E_WASTE, Material.METAL, 0.6),
        ("val", Material.TEXTILE, Material.TEXTILE, 0.95),
        ("test", Material.GLASS, Material.GLASS, 0.9),
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["path", "source", "split", "material", *(f"p_{m.value}" for m in Material)]
        )
        for i, (split, true, top, p) in enumerate(rows):
            rest = (1.0 - p) / (len(Material) - 1)
            probs = [p if m is top else rest for m in Material]
            source = "local" if i % 2 == 0 else "trashnet"
            writer.writerow([f"img{i}.jpg", source, split, true.value, *probs])


def test_load_items_keeps_one_split(tmp_path: Path) -> None:
    write_predictions(tmp_path / "p.csv")
    items = load_items(tmp_path / "p.csv", "val")
    assert [i.material for i in items] == [
        Material.PLASTIC,
        Material.STYROFOAM,
        Material.E_WASTE,
        Material.TEXTILE,
    ]
    assert all(abs(sum(i.probabilities.values()) - 1.0) < 1e-9 for i in items)


def test_cases_are_reproducible_and_mix_conditions(tmp_path: Path) -> None:
    write_predictions(tmp_path / "p.csv")
    items = load_items(tmp_path / "p.csv", "val")
    a = make_cases(items, repeats=25, clean_share=0.5, seed=1)
    b = make_cases(items, repeats=25, clean_share=0.5, seed=1)
    assert a == b and len(a) == 100
    assert {c.condition for c in a} == set(Condition)


def test_sweep_reports_every_setting(tmp_path: Path) -> None:
    write_predictions(tmp_path / "p.csv")
    cases = make_cases(load_items(tmp_path / "p.csv", "val"), 10, 0.5, 0)
    results = sweep(cases, HDB, [0.6, 0.8])
    assert [(r.system, r.threshold, r.e_waste_rule) for r in results] == [
        ("adaptive", 0.6, True),
        ("adaptive", 0.6, False),
        ("static", 0.6, True),
        ("adaptive", 0.8, True),
        ("adaptive", 0.8, False),
        ("static", 0.8, True),
    ]
    assert all(r.avg_questions == 3.0 for r in results if r.system == "static")
    for r in results:
        assert r.runs == 40
        assert 0.0 <= r.avg_questions <= 3.0
        assert 0.0 <= r.costly_error_rate <= 1.0
        # Every costly error is also a wrong verdict.
        costly_runs = r.costly_error_rate * r.costly_error_base
        assert costly_runs <= (1.0 - r.verdict_accuracy) * r.runs + 1e-9


def test_styrofoam_misread_as_plastic_is_not_sent_to_the_blue_bin(tmp_path: Path) -> None:
    write_predictions(tmp_path / "p.csv")
    styrofoam = [
        i for i in load_items(tmp_path / "p.csv", "val") if i.material is Material.STYROFOAM
    ]
    result = run_gate(make_cases(styrofoam, 50, 1.0, 0), HDB, decision.DEFAULT_GATE)
    assert result.costly_error_rate < 0.5


def test_main_writes_json(tmp_path: Path) -> None:
    write_predictions(tmp_path / "p.csv")
    out = tmp_path / "gate.json"
    assert (
        main(["--predictions", str(tmp_path / "p.csv"), "--repeats", "2", "--out", str(out)]) == 0
    )
    assert len(json.loads(out.read_text())["results"]) == 12


def test_source_filter(tmp_path: Path) -> None:
    write_predictions(tmp_path / "p.csv")
    local = load_items(tmp_path / "p.csv", "val", "local")
    assert [i.material for i in local] == [Material.PLASTIC, Material.E_WASTE]


def test_costly_errors_use_the_evaluator_definition(tmp_path: Path) -> None:
    # As in evaluate_baselines.py: only runs whose true bin is not a recycling bin count.
    write_predictions(tmp_path / "p.csv")
    plastic = [i for i in load_items(tmp_path / "p.csv", "val") if i.material is Material.PLASTIC]
    clean = run_gate(make_cases(plastic, 20, 1.0, 0), HDB, decision.DEFAULT_GATE)
    assert clean.costly_error_base == 0 and clean.costly_error_rate == 0.0
    dirty = run_gate(make_cases(plastic, 20, 0.0, 0), HDB, decision.DEFAULT_GATE)
    assert dirty.costly_error_base == dirty.runs
