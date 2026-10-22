from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient

from binbuddy_contracts.api import AnswerResponse, PredictResponse, VerifyResponse
from binbuddy_contracts.constants import EMBEDDING_DIM
from binbuddy_contracts.enums import QuestionId, ResponseStatus
from binbuddy_contracts.questions import OPENING_QUESTION, QUESTION_BANK
from tests.conftest import image_b64


def test_health_and_version(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    v = client.get("/version").json()
    assert v["embedding_dim"] == EMBEDDING_DIM
    assert {p["site"] for p in v["policies"]} >= {"hdb_default"}


def test_blurry_photo_asks_for_retake(client: TestClient) -> None:
    body = client.post("/predict", json={"image_base64": image_b64(flat=True)}).json()
    res = PredictResponse.model_validate(body)
    assert res.status is ResponseStatus.RETAKE


def test_unknown_policy_is_an_error(client: TestClient) -> None:
    r = client.post("/predict", json={"image_base64": image_b64(), "site_policy": "mars_base"})
    assert r.status_code == 400
    assert r.json()["code"] == "unknown_policy"


def _run_to_verdict(client: TestClient, seed: int) -> PredictResponse:
    res = PredictResponse.model_validate(
        client.post("/predict", json={"image_base64": image_b64(seed)}).json()
    )
    if res.status is ResponseStatus.QUESTION:
        assert res.next_question is not None
        assert res.next_question.id in {OPENING_QUESTION, QuestionId.Q2_CLEANED}
    return res


def test_question_flow_reaches_a_verdict(client: TestClient) -> None:
    for seed in range(40):
        res = _run_to_verdict(client, seed)
        state, question = res.state, res.next_question
        steps = 0
        while question is not None:
            assert state is not None
            ans = {
                "question_id": question.id.value,
                "answer": QUESTION_BANK[question.id].options[0],
            }
            body = client.post(
                "/answer", json={"state": state.model_dump(mode="json"), "answer": ans}
            ).json()
            step = AnswerResponse.model_validate(body)
            state, question = step.state, step.next_question
            steps += 1
            assert steps <= 3, "never more than three questions"
            if step.status is ResponseStatus.VERDICT:
                assert step.verdict is not None


def test_tampered_state_is_rejected(client: TestClient) -> None:
    for seed in range(40):
        res = _run_to_verdict(client, seed)
        if res.state is None:
            continue
        state: dict[str, Any] = res.state.model_dump(mode="json")
        state["confidence"] = 0.99
        assert res.next_question is not None
        ans = {"question_id": res.next_question.id.value, "answer": "yes"}
        r = client.post("/answer", json={"state": state, "answer": ans})
        assert r.status_code == 400
        assert r.json()["code"] == "invalid_state"
        return
    raise AssertionError("no seed produced a question")


def test_verify_same_photo_right_bin_passes(client: TestClient) -> None:
    img = image_b64(7)
    res = PredictResponse.model_validate(client.post("/predict", json={"image_base64": img}).json())
    assert res.analysis is not None
    now = datetime.now(UTC)
    payload = {
        "scan_id": str(res.scan_id),
        "image_base64": img,
        "qr": {
            "v": 1,
            "bin_id": "demo-hdb-01",
            "site_policy": "hdb_default",
            "bin_key": "blue_bin",
            "location_label": "Demo block",
        },
        "recommended_site_policy": "hdb_default",
        "recommended_bin_key": "blue_bin",
        "original_embedding": res.analysis.embedding,
        "original_phash": res.analysis.phash,
        "original_created_at": now.isoformat(),
        "submitted_at": (now + timedelta(minutes=2)).isoformat(),
    }
    ok = VerifyResponse.model_validate(client.post("/verify", json=payload).json())
    assert ok.passed

    wrong_bin = {**payload, "recommended_bin_key": "general_waste"}
    bad = VerifyResponse.model_validate(client.post("/verify", json=wrong_bin).json())
    assert not bad.passed and not bad.checks.bin_matches_verdict

    late = {**payload, "submitted_at": (now + timedelta(hours=2)).isoformat()}
    assert not VerifyResponse.model_validate(client.post("/verify", json=late).json()).passed

    reused = {**payload, "known_phashes": [res.analysis.phash]}
    dup = VerifyResponse.model_validate(client.post("/verify", json=reused).json())
    assert not dup.checks.not_duplicate


def test_root_shows_a_welcome_page_with_links(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert 'href="/docs"' in r.text
