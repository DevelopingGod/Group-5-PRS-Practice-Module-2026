"""Answer likelihoods for the question bank (WS3): P(answer | class).

PROVISIONAL, HAND-SET. The local photos were collected without Q1-Q3 answers, so these tables
are set by domain judgement, not counted from data/labels.csv. Every row sums to 1 and no cell is
below LIKELIHOOD_FLOOR, so one unexpected answer can never rule a class out completely. When
recorded answers exist, scripts/estimate_likelihoods.py counts them with Laplace smoothing,
P(answer | class) = (count + 1) / (rows of that class + number of answers),
and writes likelihoods_counted.json next to this file; its rows replace the hand-set ones at import.

Which class each question informs (naive-Bayes evidence fusion, proposal 1.2):
  Q1 recycling symbol -> material   (a symbol is evidence for packaging, never a verdict)
  Q2 cleaned          -> condition
  Q3 sitting time     -> condition, weighted by material (see Q3 below)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from binbuddy_contracts.enums import Condition, Material, QuestionId, SittingTime, YesNoUnsure

LIKELIHOOD_FLOOR: Final = 0.01

Row = dict[str, float]

_YES, _NO, _NOT_SURE = YesNoUnsure.YES.value, YesNoUnsure.NO.value, YesNoUnsure.NOT_SURE.value
_UNDER_1H = SittingTime.UNDER_1_HOUR.value
_SAME_DAY = SittingTime.SAME_DAY.value
_OVER_1D = SittingTime.MORE_THAN_1_DAY.value


def _yes_no(yes: float, no: float, not_sure: float) -> Row:
    return {_YES: yes, _NO: no, _NOT_SURE: not_sure}


def _time(under_1_hour: float, same_day: float, more_than_1_day: float) -> Row:
    return {_UNDER_1H: under_1_hour, _SAME_DAY: same_day, _OVER_1D: more_than_1_day}


# Q1 "Do you see a recycling symbol?" given the material.
Q1_GIVEN_MATERIAL: Final[dict[Material, Row]] = {
    Material.PAPER: _yes_no(0.80, 0.10, 0.10),
    Material.PLASTIC: _yes_no(0.85, 0.05, 0.10),  # resin codes are moulded on most packaging
    Material.GLASS: _yes_no(0.60, 0.25, 0.15),  # bottles and jars often carry no symbol
    Material.METAL: _yes_no(0.50, 0.30, 0.20),  # cans often carry no symbol
    Material.BEVERAGE_CARTON: _yes_no(0.75, 0.15, 0.10),
    Material.STYROFOAM: _yes_no(0.60, 0.20, 0.20),  # resin code 6; the policy still says general
    Material.E_WASTE: _yes_no(0.15, 0.60, 0.25),
    Material.TEXTILE: _yes_no(0.05, 0.80, 0.15),
    Material.FOOD_WASTE: _yes_no(0.05, 0.85, 0.10),
    Material.OTHER_GENERAL: _yes_no(0.10, 0.75, 0.15),
}

# Q2 "Did you clean or rinse it?" given the condition. The questions tell clean from not clean,
# not which kind of contamination, so the three contaminated conditions share one row. "Not sure"
# is kept close to neutral (proposal 1.2: Q3 follows when Q2 was "No or Not sure").
_Q2_CONTAMINATED: Final[Row] = _yes_no(0.05, 0.75, 0.20)
Q2_GIVEN_CONDITION: Final[dict[Condition, Row]] = {
    Condition.CLEAN: _yes_no(0.82, 0.03, 0.15),
    Condition.FOOD_RESIDUE: dict(_Q2_CONTAMINATED),
    Condition.LEFTOVER_LIQUID: dict(_Q2_CONTAMINATED),
    Condition.ITEM_INSIDE: dict(_Q2_CONTAMINATED),
}

# Q3 "How long has it been sitting (with food or drink in it)?" given the condition.
# The material is a proxy for the contents: cartons, cans, bottles and plastic cups or containers
# mostly hold drinks or wet food, which spoil within hours, so a long wait is strong evidence of
# residue. Paper and other packaging more often held dry food, so the wait says less.
Q3_GIVEN_CLEAN: Final[Row] = _time(0.45, 0.30, 0.25)

PERISHABLE_CONTENTS: Final = frozenset(
    {Material.PLASTIC, Material.GLASS, Material.METAL, Material.BEVERAGE_CARTON}
)

# As for Q2, one row covers all three contaminated conditions.
Q3_GIVEN_RESIDUE_PERISHABLE: Final[Row] = _time(0.15, 0.35, 0.50)
Q3_GIVEN_RESIDUE_DRY: Final[Row] = _time(0.30, 0.35, 0.35)


def q1(material: Material, answer: str) -> float:
    """P(Q1 answer | material)."""
    return Q1_GIVEN_MATERIAL[material][answer]


def _q3(condition: Condition, answer: str, perishable_share: float) -> float:
    if condition is Condition.CLEAN:
        return Q3_GIVEN_CLEAN[answer]
    return (
        perishable_share * Q3_GIVEN_RESIDUE_PERISHABLE[answer]
        + (1.0 - perishable_share) * Q3_GIVEN_RESIDUE_DRY[answer]
    )


def condition_likelihood(
    qid: QuestionId, answer: str, materials: Mapping[Material, float]
) -> dict[Condition, float]:
    """P(answer | condition) for Q2 or Q3.

    Q3 mixes the perishable and dry tables by the current material belief. The state keeps
    material and condition as separate distributions, so this mixture stands in for a joint
    P(answer | material, condition).
    """
    if qid is QuestionId.Q2_CLEANED:
        return {c: Q2_GIVEN_CONDITION[c][answer] for c in Condition}
    if qid is QuestionId.Q3_SITTING_TIME:
        share = sum(p for m, p in materials.items() if m in PERISHABLE_CONTENTS)
        return {c: _q3(c, answer, share) for c in Condition}
    raise ValueError(f"{qid} is evidence about the material, not the condition")


def all_rows() -> dict[str, Row]:
    """Every likelihood row by name, for validation and tests."""
    rows = {f"q1/{m.value}": r for m, r in Q1_GIVEN_MATERIAL.items()}
    rows |= {f"q2/{c.value}": r for c, r in Q2_GIVEN_CONDITION.items()}
    rows |= {
        "q3/clean": Q3_GIVEN_CLEAN,
        "q3/residue_perishable": Q3_GIVEN_RESIDUE_PERISHABLE,
        "q3/residue_dry": Q3_GIVEN_RESIDUE_DRY,
    }
    return rows


COUNTED_PATH: Final = Path(__file__).with_name("likelihoods_counted.json")


def check_row(name: str, row: Mapping[str, float], options: Mapping[str, float]) -> None:
    if set(row) != set(options):
        raise ValueError(f"{name}: answers must be {sorted(options)}, got {sorted(row)}")
    if abs(sum(row.values()) - 1.0) > 1e-6:
        raise ValueError(f"{name}: probabilities must sum to 1")
    if min(row.values()) < LIKELIHOOD_FLOOR - 1e-9:
        raise ValueError(f"{name}: no probability may be below {LIKELIHOOD_FLOOR}")


def load_counted(path: Path = COUNTED_PATH, rows: Mapping[str, Row] | None = None) -> list[str]:
    """Replace hand-set rows with counted ones from `path`, if it exists. Returns the row names.

    `rows` defaults to the live tables; tests pass copies.
    """
    if not path.exists():
        return []
    targets = all_rows() if rows is None else rows
    counted: dict[str, dict[str, float]] = json.loads(path.read_text(encoding="utf-8"))["rows"]
    for name, row in counted.items():
        if name not in targets:
            raise ValueError(f"{path.name}: unknown row '{name}'")
        check_row(name, row, targets[name])
    for name, row in counted.items():
        targets[name].clear()
        targets[name].update(row)
    return sorted(counted)


COUNTED_ROWS: Final = load_counted()
