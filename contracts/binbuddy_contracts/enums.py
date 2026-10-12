"""Label taxonomy, framed on the submitted proposal (section references in comments).

Values are lowercase snake_case so they are identical in Python, JSON, TypeScript
and the Postgres enums in supabase/migrations.
"""

from enum import StrEnum


class Material(StrEnum):
    """What the item is made of (proposal 1.3, 4.2)."""

    PAPER = "paper"  # paper and cardboard ("clean paper", "takeaway box")
    PLASTIC = "plastic"  # e.g. PP container, bubble tea cup
    GLASS = "glass"
    METAL = "metal"  # e.g. aluminium can
    BEVERAGE_CARTON = "beverage_carton"  # e.g. milk or drink carton (Q2 example)
    STYROFOAM = "styrofoam"  # never blue bin, even with a symbol (1.3)
    E_WASTE = "e_waste"  # small electronic appliances, light bulbs (1.3)
    TEXTILE = "textile"  # old clothing (1.3)
    FOOD_WASTE = "food_waste"  # the item itself is food (e.g. a banana peel on its own)
    OTHER_GENERAL = "other_general"  # anything else that is general waste (e.g. tissues)


class Condition(StrEnum):
    """Contamination state (proposal: food residue, leftover liquid, items left inside)."""

    CLEAN = "clean"
    FOOD_RESIDUE = "food_residue"  # includes oil- or food-stained paper
    LEFTOVER_LIQUID = "leftover_liquid"
    ITEM_INSIDE = "item_inside"  # e.g. a banana peel inside a plastic container


class Channel(StrEnum):
    """Disposal channel a site bin belongs to. Site policies map channels to concrete bins."""

    RECYCLE = "recycle"  # blue bin at HDB; colour-coded recycling bins at multi-stream sites
    GENERAL_WASTE = "general_waste"
    E_WASTE = "e_waste"
    DONATION = "donation"
    ABSTAIN = "abstain"  # no verdict: unfamiliar item, link to NEA Check My Item


class QuestionId(StrEnum):
    """The fixed question bank (proposal 1.2)."""

    Q1_RECYCLING_SYMBOL = "q1_recycling_symbol"
    Q2_CLEANED = "q2_cleaned"
    Q3_SITTING_TIME = "q3_sitting_time"


class YesNoUnsure(StrEnum):
    YES = "yes"
    NO = "no"
    NOT_SURE = "not_sure"


class SittingTime(StrEnum):
    UNDER_1_HOUR = "under_1_hour"
    SAME_DAY = "same_day"
    MORE_THAN_1_DAY = "more_than_1_day"


class QualityIssue(StrEnum):
    """Why a photo needs a retake (proposal 1.1: too blurred or dark)."""

    TOO_BLURRY = "too_blurry"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"


class ResponseStatus(StrEnum):
    RETAKE = "retake"  # photo failed the quality check
    QUESTION = "question"  # the model is uncertain; ask next_question
    VERDICT = "verdict"  # final answer (may be the abstain channel)


class Capture(StrEnum):
    """Dataset only: paired capture (proposal 4.2)."""

    AS_USED = "as_used"
    CLEANED = "cleaned"


class Split(StrEnum):
    """Dataset only: split by item, never by photo (proposal 4.2)."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    FIELD = "field"  # held-back field-test items (>= 50)


class ErrorCode(StrEnum):
    INVALID_IMAGE = "invalid_image"
    IMAGE_TOO_LARGE = "image_too_large"
    INVALID_STATE = "invalid_state"
    INVALID_ANSWER = "invalid_answer"
    UNKNOWN_POLICY = "unknown_policy"
    MODEL_UNAVAILABLE = "model_unavailable"
    INTERNAL = "internal"
