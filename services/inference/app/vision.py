"""STUB vision model (WS1 + WS2 replace this). Returns deterministic, contract-valid output
so WS3 and WS4 can build against the API from day one."""

from __future__ import annotations

import hashlib

from PIL import Image

from app.quality import average_hash
from binbuddy_contracts import examples
from binbuddy_contracts.api import VisionAnalysis
from binbuddy_contracts.enums import Condition, Material


def analyse(img: Image.Image) -> VisionAnalysis:
    digest = hashlib.sha256(img.tobytes()).digest()
    materials, conditions = list(Material), list(Condition)
    material = materials[digest[0] % len(materials)]
    condition = conditions[digest[1] % len(conditions)]
    confidence = 0.5 + (digest[2] % 50) / 100.0  # 0.50-0.99
    return VisionAnalysis(
        material_scores=examples.material_scores(material, 0.7),
        condition_scores=examples.condition_scores(condition, 0.7),
        confidence=confidence,
        is_ood=digest[3] % 20 == 0,
        embedding=examples.unit_embedding(seed=digest[4]),
        phash=average_hash(img),
    )
