from typing import Any

import pytest
from evaluate_baselines import (
    Post,
    Prediction,
    Record,
    evaluate,
    material_macro_f1,
    parse_vlm_reply,
    recorded_answer,
    run_cnn,
    run_full,
    run_vlm,
    summarise,
)
from fastapi.testclient import TestClient

from app.main import app
from binbuddy_contracts.constants import DEFAULT_SITE_POLICY
from binbuddy_contracts.dataset import LabelRow
from binbuddy_contracts.enums import Condition, Material, QuestionId
from binbuddy_contracts.policy import SitePolicy, load_policies
from binbuddy_ml.labels import REPO_ROOT
from tests.conftest import image_b64


@pytest.fixture(scope="module")
def hdb() -> SitePolicy:
    return load_policies(REPO_ROOT / "policies")[DEFAULT_SITE_POLICY]


@pytest.fixture(scope="module")
def stub_post() -> Post:
    client = TestClient(app)

    def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        res = client.post(path, json=body)
        assert res.status_code == 200, res.text
        data: dict[str, Any] = res.json()
        return data

    return post


def row(n: int, material: str = "plastic", condition: str = "clean", bin_key: str = "blue_bin",
        q2: str = "na", q3: str = "na") -> LabelRow:  # fmt: skip
    return LabelRow(
        photo_id=f"ws4-{n:04d}-a",
        item_id=f"ws4-{n:04d}",
        collector="ws4",
        capture="as_used",
        drive_path=f"local/ws4-{n:04d}/ws4-{n:04d}-a.jpg",
        material=material,
        condition=condition,
        hdb_bin_key=bin_key,
        q1_recycling_symbol="yes",
        q2_cleaned=q2,
        q3_sitting_time=q3,
        split="test",
    )


def rec(material: Material, condition: Condition, true_bin: str, pred: Prediction) -> Record:
    return Record("p", "i", material, condition, true_bin, pred)


def test_cnn_and_full_run_against_the_stub_api(stub_post: Post, hdb: SitePolicy) -> None:
    rows = [row(i) for i in range(1, 7)]

    def load(r: LabelRow) -> str:
        return image_b64(int(r.item_id[-4:]))

    for run in (run_cnn, run_full):

        def system(img: str, r: LabelRow, run: Any = run) -> Prediction:
            pred: Prediction = run(stub_post, img, r, hdb)
            return pred

        records = evaluate(rows, load, system)
        assert all(not r.error for r in records)
        assert all(r.pred.material is not None for r in records)
        assert all(r.pred.questions is not None and r.pred.questions <= 3 for r in records)
        assert summarise(records, hdb)["scored"] == len(rows)


def test_cnn_never_asks_questions(stub_post: Post, hdb: SitePolicy) -> None:
    pred = run_cnn(stub_post, image_b64(3), row(3), hdb)
    assert pred.questions == 0
    assert pred.bin_key in {b.key for b in hdb.bins}


def test_na_answers_add_no_contamination_evidence() -> None:
    r = row(1)
    assert recorded_answer(r, QuestionId.Q1_RECYCLING_SYMBOL) == "yes"
    assert recorded_answer(r, QuestionId.Q2_CLEANED) == "not_sure"
    assert recorded_answer(r, QuestionId.Q3_SITTING_TIME) == "under_1_hour"
    dirty = row(2, condition="food_residue", bin_key="general_waste", q2="no", q3="same_day")
    assert recorded_answer(dirty, QuestionId.Q2_CLEANED) == "no"
    assert recorded_answer(dirty, QuestionId.Q3_SITTING_TIME) == "same_day"


def test_summary_metrics(hdb: SitePolicy) -> None:
    p, gw = Material.PLASTIC, "general_waste"
    records = [
        rec(p, Condition.CLEAN, "blue_bin", Prediction(p, Condition.CLEAN, "blue_bin", 0)),
        # costly error: a dirty item sent to the blue bin, contamination missed
        rec(p, Condition.FOOD_RESIDUE, gw, Prediction(p, Condition.CLEAN, "blue_bin", 1)),
        # caught and right
        rec(p, Condition.FOOD_RESIDUE, gw, Prediction(p, Condition.FOOD_RESIDUE, gw, 2)),
        # abstained
        rec(Material.STYROFOAM, Condition.CLEAN, gw, Prediction(Material.STYROFOAM, None, None, 0)),
        Record("x", "x", p, Condition.CLEAN, "blue_bin", error="RuntimeError: down"),
    ]
    m = summarise(records, hdb)
    assert (m["photos"], m["scored"], m["failed_calls"]) == (5, 4, 1)
    assert m["verdict_accuracy"] == pytest.approx(2 / 4)
    assert m["accuracy_on_answered"] == pytest.approx(2 / 3)
    assert m["abstain_rate"] == pytest.approx(1 / 4)
    assert m["contamination_recall"] == pytest.approx(1 / 2)
    assert (m["costly_error_rate"], m["costly_error_base"]) == (pytest.approx(1 / 3), 3)
    assert m["avg_questions"] == pytest.approx(3 / 4)
    assert m["material_macro_f1"] == pytest.approx(1.0)


def test_macro_f1_counts_a_missing_guess_as_a_miss() -> None:
    records = [
        rec(Material.PAPER, Condition.CLEAN, "blue_bin", Prediction(Material.PAPER)),
        rec(Material.GLASS, Condition.CLEAN, "blue_bin", Prediction(None)),
    ]
    assert material_macro_f1(records) == pytest.approx(0.5)


def test_parse_vlm_reply(hdb: SitePolicy) -> None:
    text = 'Sure!\n```json\n{"material": "metal", "condition": "clean", "bin": "blue_bin"}\n```'
    assert parse_vlm_reply(text, hdb) == Prediction(Material.METAL, Condition.CLEAN, "blue_bin")
    odd = parse_vlm_reply('{"material": "aluminium", "condition": "clean", "bin": "bin"}', hdb)
    assert (odd.material, odd.condition, odd.bin_key) == (None, Condition.CLEAN, None)
    assert parse_vlm_reply("I cannot tell.", hdb).note == "unparsed reply"


def test_run_vlm_sends_the_photo_and_allowed_labels(hdb: SitePolicy) -> None:
    sent: list[dict[str, Any]] = []

    def fake(path: str, body: dict[str, Any]) -> dict[str, Any]:
        sent.append({"path": path, **body})
        reply = '{"material": "styrofoam", "condition": "clean", "bin": "general_waste"}'
        return {"choices": [{"message": {"content": reply}}]}

    pred = run_vlm(fake, "some-vlm", "QUJD", hdb)
    assert pred.bin_key == "general_waste" and pred.material is Material.STYROFOAM
    content = sent[0]["messages"][0]["content"]
    assert sent[0]["path"] == "/chat/completions" and sent[0]["model"] == "some-vlm"
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert "e_waste_point" in content[0]["text"] and "beverage_carton" in content[0]["text"]
