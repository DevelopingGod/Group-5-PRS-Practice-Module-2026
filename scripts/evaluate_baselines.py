"""Score Bin Buddy and its baselines on the same photos (context.md Section 9, WS4).

Systems (pick with --system):
  cnn   Baseline 1: the vision model alone, no questions. Calls the model API's /predict, takes
        the top material and condition and applies the HDB policy (the first matching rule).
  full  Bin Buddy end to end: /predict, then /answer with the collector's recorded Q1-Q3
        answers from data/labels.csv, until a verdict.
  vlm   Baseline 3: a general-purpose vision-language model given the same photo, as a proxy for
        chatbot-style tools. Any OpenAI-compatible chat-completions server works (a local
        Ollama or LM Studio server, or a hosted API): --vlm-base-url and --vlm-model; the key,
        if one is needed, comes from the VLM_API_KEY environment variable.

    uv run python scripts/evaluate_baselines.py --system cnn \
        --model-api http://localhost:7860 --split test --out results/cnn
    uv run python scripts/evaluate_baselines.py --system vlm \
        --vlm-base-url http://localhost:11434/v1 --vlm-model qwen2.5vl:7b --out results/vlm

Photos are read from the Drive folder (BINBUDDY_DATA_ROOT, see ml/binbuddy_ml/paths.py) and
shrunk like the web app does before upload. Outputs in --out: predictions.csv (one row per
photo) and summary.json (the metrics and which model produced them). Keep results out of git.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

# The ml helpers are not installed as a package; put them on the path like the tests do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from binbuddy_contracts.api import (
    AnswerResponse,
    DecisionState,
    PredictResponse,
    QuestionAnswer,
    VerdictResult,
)
from binbuddy_contracts.constants import DEFAULT_SITE_POLICY, MAX_IMAGE_BYTES
from binbuddy_contracts.dataset import LabelRow
from binbuddy_contracts.enums import (
    Channel,
    Condition,
    Material,
    QuestionId,
    ResponseStatus,
    SittingTime,
    Split,
    YesNoUnsure,
)
from binbuddy_contracts.policy import SitePolicy, load_policies
from binbuddy_ml.labels import REPO_ROOT, load_labels
from binbuddy_ml.paths import photo_path

# Same as apps/web/src/lib/image.ts, so every system sees the photo the app would send.
MAX_SIDE = 1024
JPEG_QUALITIES = (85, 70, 50)

Post = Callable[[str, dict[str, Any]], dict[str, Any]]


# ------------------------------------------------------------------------------ records


@dataclass
class Prediction:
    """One system's answer for one photo. bin_key None means no verdict (abstain or retake)."""

    material: Material | None = None
    condition: Condition | None = None
    bin_key: str | None = None
    questions: int | None = None
    note: str = ""


@dataclass
class Record:
    photo_id: str
    item_id: str
    true_material: Material
    true_condition: Condition
    true_bin: str
    pred: Prediction = field(default_factory=Prediction)
    error: str = ""  # the call failed; the photo is left out of the metrics


# ------------------------------------------------------------------------------ metrics


def _f1(tp: int, fp: int, fn: int) -> float:
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def material_macro_f1(records: Sequence[Record]) -> float | None:
    """Macro-F1 over the materials present in the ground truth (a missing guess is a miss)."""
    if not records:
        return None
    classes = sorted({r.true_material for r in records})
    scores = []
    for c in classes:
        tp = sum(r.true_material is c and r.pred.material is c for r in records)
        fp = sum(r.true_material is not c and r.pred.material is c for r in records)
        fn = sum(r.true_material is c and r.pred.material is not c for r in records)
        scores.append(_f1(tp, fp, fn))
    return sum(scores) / len(scores)


def _rate(hits: Iterable[bool]) -> float | None:
    values = list(hits)
    return sum(values) / len(values) if values else None


def summarise(records: Sequence[Record], policy: SitePolicy) -> dict[str, Any]:
    """Section 9 metrics. Abstaining counts as a wrong verdict in end-to-end accuracy."""
    scored = [r for r in records if not r.error]
    answered = [r for r in scored if r.pred.bin_key is not None]
    dirty = [r for r in scored if r.true_condition is not Condition.CLEAN]

    def recycles(key: str | None) -> bool:
        return key is not None and policy.bin(key).channel is Channel.RECYCLE

    not_blue = [r for r in scored if not recycles(r.true_bin)]
    asked = [r.pred.questions for r in scored if r.pred.questions is not None]
    return {
        "photos": len(records),
        "scored": len(scored),
        "failed_calls": len(records) - len(scored),
        "material_macro_f1": material_macro_f1(scored),
        "contamination_recall": _rate(
            r.pred.condition is not None and r.pred.condition is not Condition.CLEAN for r in dirty
        ),
        "verdict_accuracy": _rate(r.pred.bin_key == r.true_bin for r in scored),
        "costly_error_rate": _rate(recycles(r.pred.bin_key) for r in not_blue),
        "costly_error_base": len(not_blue),
        "abstain_rate": _rate(r.pred.bin_key is None for r in scored),
        "accuracy_on_answered": _rate(r.pred.bin_key == r.true_bin for r in answered),
        "avg_questions": sum(asked) / len(asked) if asked else None,
    }


# ------------------------------------------------------------------------------ helpers


def image_base64(path: Path) -> str:
    """Shrink and re-encode as JPEG (drops EXIF), like the web app before upload."""
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    for quality in JPEG_QUALITIES:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return base64.b64encode(buf.getvalue()).decode()
    raise ValueError(f"{path.name} is too large even after shrinking")


def http_post(base_url: str, timeout_s: float, headers: dict[str, str] | None = None) -> Post:
    def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json", **(headers or {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as res:
                data: dict[str, Any] = json.loads(res.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{path} returned {exc.code}: {exc.read()[:200]!r}") from exc
        return data

    return post


def http_get_json(url: str, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as res:
        data: dict[str, Any] = json.loads(res.read())
    return data


def _top(scores: Sequence[Any]) -> Any:
    return max(scores, key=lambda s: s.probability).label


def _from_verdict(verdict: VerdictResult, questions: int) -> Prediction:
    return Prediction(bin_key=verdict.bin_key, questions=questions, note=verdict.rule_id)


# ------------------------------------------------------------------------------ systems


def run_cnn(post: Post, image_b64: str, row: LabelRow, policy: SitePolicy) -> Prediction:
    res = PredictResponse.model_validate(
        post("/predict", {"image_base64": image_b64, "site_policy": policy.site})
    )
    if res.status is ResponseStatus.RETAKE or res.analysis is None:
        return Prediction(questions=0, note="retake")
    material = _top(res.analysis.material_scores)
    condition = _top(res.analysis.condition_scores)
    if res.analysis.is_ood:
        return Prediction(material, condition, None, 0, "ood")
    rule = policy.first_match(material, condition)
    return Prediction(material, condition, rule.then, 0, rule.id)


def recorded_answer(row: LabelRow, qid: QuestionId) -> str:
    """The collector's answer from labels.csv. 'na' (no food or drink involved) is mapped to the
    answer that adds no contamination evidence: 'not_sure' for Q2, 'under_1_hour' for Q3."""
    value = {
        QuestionId.Q1_RECYCLING_SYMBOL: row.q1_recycling_symbol,
        QuestionId.Q2_CLEANED: row.q2_cleaned,
        QuestionId.Q3_SITTING_TIME: row.q3_sitting_time,
    }[qid]
    if value == "na":
        if qid is QuestionId.Q3_SITTING_TIME:
            return SittingTime.UNDER_1_HOUR.value
        return YesNoUnsure.NOT_SURE.value
    return str(value)


def run_full(post: Post, image_b64: str, row: LabelRow, policy: SitePolicy) -> Prediction:
    res = PredictResponse.model_validate(
        post("/predict", {"image_base64": image_b64, "site_policy": policy.site})
    )
    if res.status is ResponseStatus.RETAKE or res.analysis is None:
        return Prediction(questions=0, note="retake")
    material = _top(res.analysis.material_scores)
    condition = _top(res.analysis.condition_scores)
    step: PredictResponse | AnswerResponse = res
    asked = 0
    while step.status is ResponseStatus.QUESTION:
        assert step.next_question is not None and step.state is not None
        state: DecisionState = step.state
        answer = QuestionAnswer(
            question_id=step.next_question.id,
            answer=recorded_answer(row, step.next_question.id),
        )
        step = AnswerResponse.model_validate(
            post(
                "/answer",
                {"state": state.model_dump(mode="json"), "answer": answer.model_dump(mode="json")},
            )
        )
        asked += 1
    assert step.verdict is not None
    pred = _from_verdict(step.verdict, asked)
    pred.material, pred.condition = material, condition
    return pred


VLM_PROMPT = """You are helping a resident in Singapore decide where one household item goes,
following NEA guidance for HDB estates. Look at the single item in the photo.

Answer with JSON only, no other text, using exactly these keys and allowed values:
{{"material": one of {materials},
  "condition": one of {conditions},
  "bin": one of {bins}}}

"condition" describes contamination: food_residue includes oil- or food-stained paper;
item_inside means something else is left inside the container (e.g. a fruit peel).
"bin" is where the item should go now, as it is: {bin_help}."""


def vlm_prompt(policy: SitePolicy) -> str:
    return VLM_PROMPT.format(
        materials=json.dumps([m.value for m in Material]),
        conditions=json.dumps([c.value for c in Condition]),
        bins=json.dumps([b.key for b in policy.bins]),
        bin_help="; ".join(f"{b.key} = {b.label}" for b in policy.bins),
    )


def parse_vlm_reply(text: str, policy: SitePolicy) -> Prediction:
    """Read the first JSON object in the reply. Values outside the allowed sets become None, so
    an unusable reply counts as no verdict instead of a guess."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    material = data.get("material")
    condition = data.get("condition")
    bin_key = data.get("bin")
    keys = {b.key for b in policy.bins}
    return Prediction(
        material=Material(material) if material in set(Material) else None,
        condition=Condition(condition) if condition in set(Condition) else None,
        bin_key=bin_key if bin_key in keys else None,
        note="" if data else "unparsed reply",
    )


def run_vlm(post: Post, model: str, image_b64: str, policy: SitePolicy) -> Prediction:
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vlm_prompt(policy)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    reply = post("/chat/completions", body)
    text = str(reply["choices"][0]["message"]["content"])
    return parse_vlm_reply(text, policy)


# ------------------------------------------------------------------------------ main


def evaluate(
    rows: Sequence[LabelRow],
    load_image: Callable[[LabelRow], str],
    system: Callable[[str, LabelRow], Prediction],
) -> list[Record]:
    records = []
    t0 = time.time()
    for i, row in enumerate(rows, start=1):
        rec = Record(row.photo_id, row.item_id, row.material, row.condition, row.hdb_bin_key)
        try:
            rec.pred = system(load_image(row), row)
        except (OSError, RuntimeError, ValueError) as exc:  # network, file and shape errors
            rec.error = f"{type(exc).__name__}: {exc}"[:300]
        records.append(rec)
        if i % 25 == 0:
            print(f"{i}/{len(rows)} photos, {(time.time() - t0) / i:.1f} s each")
    return records


def write_outputs(out: Path, records: Sequence[Record], summary: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["photo_id", "item_id", "true_material", "true_condition", "true_bin",
             "pred_material", "pred_condition", "pred_bin", "questions", "note", "error"]
        )  # fmt: skip
        for r in records:
            p = r.pred
            writer.writerow(
                [r.photo_id, r.item_id, r.true_material, r.true_condition, r.true_bin,
                 p.material or "", p.condition or "", p.bin_key or "", p.questions, p.note,
                 r.error]
            )  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--system", choices=["cnn", "full", "vlm"], required=True)
    ap.add_argument("--split", choices=[s.value for s in Split], default=Split.TEST.value)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N photos")
    ap.add_argument("--model-api", default="http://localhost:7860", help="for cnn and full")
    ap.add_argument("--vlm-base-url", help="OpenAI-compatible base URL, e.g. .../v1")
    ap.add_argument("--vlm-model", help="model name on that server")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds per call")
    args = ap.parse_args(argv)

    policy = load_policies(REPO_ROOT / "policies")[DEFAULT_SITE_POLICY]
    rows = load_labels(Split(args.split))[: args.limit]
    if not rows:
        print(f"no photos in split '{args.split}' of data/labels.csv yet")
        return 1

    meta: dict[str, Any] = {"system": args.system, "split": args.split, "policy": policy.site}
    meta["policy_version"] = policy.version
    system: Callable[[str, LabelRow], Prediction]
    if args.system == "vlm":
        if not (args.vlm_base_url and args.vlm_model):
            ap.error("--system vlm needs --vlm-base-url and --vlm-model")
        key = os.getenv("VLM_API_KEY")
        headers = {"authorization": f"Bearer {key}"} if key else {}
        vlm_post = http_post(args.vlm_base_url, args.timeout, headers)
        meta["vlm_model"] = args.vlm_model
        meta["vlm_base_url"] = args.vlm_base_url
        system = lambda img, _row: run_vlm(vlm_post, args.vlm_model, img, policy)  # noqa: E731
    else:
        api_post = http_post(args.model_api, args.timeout)
        meta["model_api"] = args.model_api
        meta["model_version"] = http_get_json(args.model_api.rstrip("/") + "/version", 120.0)
        run = run_cnn if args.system == "cnn" else run_full
        system = lambda img, row: run(api_post, img, row, policy)  # noqa: E731

    records = evaluate(rows, lambda row: image_base64(photo_path(row.drive_path)), system)
    summary = {**meta, "metrics": summarise(records, policy)}
    write_outputs(args.out, records, summary)
    print(json.dumps(summary["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
