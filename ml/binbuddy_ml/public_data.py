"""Map the public datasets onto the contract's Material taxonomy, and split leak-free (WS1).

Pure Python (no PyTorch) so it is type-checked and tested in CI and imported by the Colab
notebooks in ml/notebooks/. The vision model predicts **material only**; cleanliness comes from
the question bank (Q2/Q3), so no public image needs a condition label.

Sources (proposal Section 4.1):
- TrashNet  (MIT)               6 classes, one item per photo on a white background
- RealWaste (CC BY-NC-SA 4.0)   9 classes, real landfill items; non-commercial academic use only
- TACO      (CC BY 4.0)         litter in the wild with boxes; each box is cropped to one item
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from binbuddy_contracts.enums import Material, Split

M = Material

TRASHNET: Mapping[str, Material] = {
    "cardboard": M.PAPER,
    "glass": M.GLASS,
    "metal": M.METAL,
    "paper": M.PAPER,
    "plastic": M.PLASTIC,
    "trash": M.OTHER_GENERAL,
}

REALWASTE: Mapping[str, Material] = {
    "Cardboard": M.PAPER,
    "Food Organics": M.FOOD_WASTE,
    "Glass": M.GLASS,
    "Metal": M.METAL,
    "Miscellaneous Trash": M.OTHER_GENERAL,
    "Paper": M.PAPER,
    "Plastic": M.PLASTIC,
    "Textile Trash": M.TEXTILE,
    "Vegetation": M.FOOD_WASTE,
}

# TACO category name -> material. None = deliberately excluded (reason in TACO_EXCLUDED).
TACO: Mapping[str, Material | None] = {
    "Aluminium foil": M.METAL,
    "Battery": M.E_WASTE,
    "Aluminium blister pack": None,
    "Carded blister pack": M.OTHER_GENERAL,
    "Other plastic bottle": M.PLASTIC,
    "Clear plastic bottle": M.PLASTIC,
    "Glass bottle": M.GLASS,
    "Plastic bottle cap": M.PLASTIC,
    "Metal bottle cap": M.METAL,
    "Broken glass": None,
    "Food Can": M.METAL,
    "Aerosol": M.METAL,
    "Drink can": M.METAL,
    "Toilet tube": M.PAPER,
    "Other carton": M.PAPER,
    "Egg carton": M.PAPER,
    "Drink carton": M.BEVERAGE_CARTON,
    "Corrugated carton": M.PAPER,
    "Meal carton": M.PAPER,
    "Pizza box": M.PAPER,
    "Paper cup": M.PAPER,
    "Disposable plastic cup": M.PLASTIC,
    "Foam cup": M.STYROFOAM,
    "Glass cup": M.GLASS,
    "Other plastic cup": M.PLASTIC,
    "Food waste": M.FOOD_WASTE,
    "Glass jar": M.GLASS,
    "Plastic lid": M.PLASTIC,
    "Metal lid": M.METAL,
    "Other plastic": M.PLASTIC,
    "Magazine paper": M.PAPER,
    "Tissues": M.OTHER_GENERAL,
    "Wrapping paper": M.PAPER,
    "Normal paper": M.PAPER,
    "Paper bag": M.PAPER,
    "Plastified paper bag": None,
    "Plastic film": M.PLASTIC,
    "Six pack rings": M.PLASTIC,
    "Garbage bag": M.PLASTIC,
    "Other plastic wrapper": M.PLASTIC,
    "Single-use carrier bag": M.PLASTIC,
    "Polypropylene bag": M.PLASTIC,
    "Crisp packet": M.OTHER_GENERAL,
    "Spread tub": M.PLASTIC,
    "Tupperware": M.PLASTIC,
    "Disposable food container": M.PLASTIC,
    "Foam food container": M.STYROFOAM,
    "Other plastic container": M.PLASTIC,
    "Plastic glooves": M.PLASTIC,
    "Plastic utensils": M.PLASTIC,
    "Pop tab": M.METAL,
    "Rope & strings": M.OTHER_GENERAL,
    "Scrap metal": M.METAL,
    "Shoe": M.TEXTILE,
    "Squeezable tube": M.OTHER_GENERAL,
    "Plastic straw": M.PLASTIC,
    "Paper straw": M.PAPER,
    "Styrofoam piece": M.STYROFOAM,
    "Unlabeled litter": None,
    "Cigarette": None,
}

TACO_EXCLUDED: Mapping[str, str] = {
    "Aluminium blister pack": "foil-plastic laminate: no single material",
    "Broken glass": "shards are not a household item a resident photographs; disposal differs",
    "Plastified paper bag": "no annotations in TACO",
    "Unlabeled litter": "unknown material",
    "Cigarette": "tiny litter, not a household item; 667 boxes would swamp other_general",
}

# A TACO box is kept only if it is big enough to look like a phone photo of one item.
TACO_MIN_SIDE_PX = 96
TACO_MIN_AREA_FRACTION = 0.01


def taco_box_usable(box_w: float, box_h: float, img_w: int, img_h: int) -> bool:
    """True if a TACO bounding box is large enough to crop as a single-item photo."""
    if min(box_w, box_h) < TACO_MIN_SIDE_PX:
        return False
    return (box_w * box_h) / (img_w * img_h) >= TACO_MIN_AREA_FRACTION


@dataclass(frozen=True)
class SplitFractions:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")


PUBLIC_SPLIT = SplitFractions(train=0.70, val=0.15, test=0.15)
# Local photos are the domain we care about, so a larger share is held out for testing.
LOCAL_SPLIT = SplitFractions(train=0.50, val=0.20, test=0.30)


def split_for_group(group_id: str, fractions: SplitFractions, seed: str = "binbuddy") -> Split:
    """Deterministic split by group (item / source photo), never by image.

    All crops of one TACO photo, and all photos of one local item, share a group id, so the
    same object can never appear in both train and test (proposal 4.2 leakage control).
    """
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < fractions.train:
        return Split.TRAIN
    if u < fractions.train + fractions.val:
        return Split.VAL
    return Split.TEST


def coverage(*mappings: Mapping[str, Material | None]) -> dict[Material, list[str]]:
    """Which source classes feed each material (empty list = no public training data)."""
    out: dict[Material, list[str]] = {m: [] for m in Material}
    for mapping in mappings:
        for name, material in mapping.items():
            if material is not None:
                out[material].append(name)
    return out
