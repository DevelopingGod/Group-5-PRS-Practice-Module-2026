"""Dataset manifest contract: data/labels.csv (photos themselves live on Google Drive).

One row per photo. Validation enforces the proposal's data rules (4.2): paired capture,
split by item (never by photo) and labels consistent with the HDB policy.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from binbuddy_contracts.api import Strict
from binbuddy_contracts.enums import (
    Capture,
    Condition,
    Material,
    SittingTime,
    Split,
    YesNoUnsure,
)
from binbuddy_contracts.policy import SitePolicy

COLUMNS: tuple[str, ...] = (
    "photo_id",
    "item_id",
    "collector",
    "capture",
    "drive_path",
    "material",
    "condition",
    "hdb_bin_key",
    "q1_recycling_symbol",
    "q2_cleaned",
    "q3_sitting_time",
    "split",
    "notes",
)

NotApplicable = Literal["na"]


class LabelRow(Strict):
    photo_id: str = Field(pattern=r"^ws[1-4]-\d{4}-[a-z]$", description="e.g. ws2-0017-b")
    item_id: str = Field(pattern=r"^ws[1-4]-\d{4}$", description="e.g. ws2-0017")
    collector: Literal["ws1", "ws2", "ws3", "ws4"]
    capture: Capture
    drive_path: str = Field(pattern=r"^local/ws[1-4]-\d{4}/ws[1-4]-\d{4}-[a-z]\.(jpg|jpeg|png)$")
    material: Material
    condition: Condition
    hdb_bin_key: str
    q1_recycling_symbol: YesNoUnsure
    q2_cleaned: YesNoUnsure | NotApplicable
    q3_sitting_time: SittingTime | NotApplicable
    split: Split
    notes: str = ""

    @model_validator(mode="after")
    def _row_rules(self) -> LabelRow:
        if not self.photo_id.startswith(self.item_id + "-"):
            raise ValueError("photo_id must start with item_id")
        if not self.item_id.startswith(self.collector + "-"):
            raise ValueError("item_id must start with the collector's workstream (e.g. ws2-)")
        if not self.drive_path.startswith(f"local/{self.item_id}/{self.photo_id}."):
            raise ValueError("drive_path must be local/<item_id>/<photo_id>.<ext>")
        if self.capture is Capture.CLEANED and self.condition is not Condition.CLEAN:
            raise ValueError("a 'cleaned' capture must have condition 'clean'")
        return self


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: header must be exactly {','.join(COLUMNS)}")
        return list(reader)


def validate_labels(rows: list[dict[str, str]], hdb_policy: SitePolicy) -> list[str]:
    """Return a list of human-readable errors (empty list = valid)."""
    errors: list[str] = []
    parsed: list[LabelRow] = []
    for i, raw in enumerate(rows, start=2):  # line 1 is the header
        try:
            parsed.append(LabelRow.model_validate(raw))
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"]) or "row"
                errors.append(f"line {i}: {loc}: {err['msg']}")

    seen: set[str] = set()
    by_item: dict[str, list[LabelRow]] = defaultdict(list)
    for row in parsed:
        if row.photo_id in seen:
            errors.append(f"{row.photo_id}: duplicate photo_id")
        seen.add(row.photo_id)
        by_item[row.item_id].append(row)
        expected = hdb_policy.first_match(row.material, row.condition).then
        if row.hdb_bin_key != expected:
            errors.append(
                f"{row.photo_id}: hdb_bin_key '{row.hdb_bin_key}' disagrees with the HDB policy "
                f"('{expected}' for {row.material.value}/{row.condition.value})"
            )

    for item_id, item_rows in by_item.items():
        for field in ("material", "split", "collector"):
            values = sorted({str(getattr(r, field)) for r in item_rows})
            if len(values) > 1:
                got = ", ".join(values)
                errors.append(f"{item_id}: all photos of an item must share {field}; got {got}")
    return errors
