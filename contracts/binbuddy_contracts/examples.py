"""Valid example objects for tests, mocks and the frontend's dummy mode."""

from __future__ import annotations

import math

from binbuddy_contracts.api import ConditionScore, MaterialScore, VisionAnalysis
from binbuddy_contracts.constants import EMBEDDING_DIM
from binbuddy_contracts.enums import Condition, Material


def unit_embedding(seed: int = 0) -> list[float]:
    raw = [math.sin(seed + i + 1.0) for i in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(v * v for v in raw))
    return [v / norm for v in raw]


def material_scores(top: Material, p_top: float = 0.9) -> list[MaterialScore]:
    rest = (1.0 - p_top) / (len(Material) - 1)
    return [MaterialScore(label=m, probability=p_top if m is top else rest) for m in Material]


def condition_scores(top: Condition, p_top: float = 0.9) -> list[ConditionScore]:
    rest = (1.0 - p_top) / (len(Condition) - 1)
    return [ConditionScore(label=c, probability=p_top if c is top else rest) for c in Condition]


def analysis(
    material: Material = Material.METAL,
    condition: Condition = Condition.CLEAN,
    confidence: float = 0.9,
    is_ood: bool = False,
) -> VisionAnalysis:
    return VisionAnalysis(
        material_scores=material_scores(material),
        condition_scores=condition_scores(condition),
        confidence=confidence,
        is_ood=is_ood,
        embedding=unit_embedding(),
        phash="0f0f0f0f0f0f0f0f",
    )
