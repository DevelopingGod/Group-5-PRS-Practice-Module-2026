"""Disposal verification (WS4). Thresholds come from the contracts constants."""

from __future__ import annotations

import math
from datetime import timedelta

from binbuddy_contracts.api import VerifyChecks, VerifyRequest, VerifyResponse
from binbuddy_contracts.constants import VERIFY_MIN_SIMILARITY, VERIFY_TIME_WINDOW_MINUTES

DUPLICATE_MAX_HAMMING = 4


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return max(-1.0, min(1.0, dot / (na * nb)))


def hamming(h1: str, h2: str) -> int:
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


def verify(req: VerifyRequest, embedding: list[float], phash: str) -> VerifyResponse:
    similarity = cosine(req.original_embedding, embedding)
    elapsed = req.submitted_at - req.original_created_at
    checks = VerifyChecks(
        image_match=similarity >= VERIFY_MIN_SIMILARITY,
        bin_matches_verdict=(
            req.qr.site_policy == req.recommended_site_policy
            and req.qr.bin_key == req.recommended_bin_key
        ),
        within_time_window=timedelta(0) <= elapsed <= timedelta(minutes=VERIFY_TIME_WINDOW_MINUTES),
        not_duplicate=all(hamming(phash, h) > DUPLICATE_MAX_HAMMING for h in req.known_phashes),
    )
    failed = [name for name, ok in checks.model_dump().items() if not ok]
    return VerifyResponse(
        passed=not failed,
        similarity=similarity,
        checks=checks,
        reason="verified" if not failed else "failed: " + ", ".join(failed),
        phash=phash,
    )
