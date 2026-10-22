"""The fixed question bank (proposal 1.2). Texts are shown to users verbatim."""

from typing import Final

from pydantic import BaseModel, ConfigDict

from binbuddy_contracts.enums import QuestionId, SittingTime, YesNoUnsure


class QuestionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: QuestionId
    text: str
    options: tuple[str, ...]


QUESTION_BANK: Final[dict[QuestionId, QuestionSpec]] = {
    QuestionId.Q1_RECYCLING_SYMBOL: QuestionSpec(
        id=QuestionId.Q1_RECYCLING_SYMBOL,
        text="Do you see a recycling symbol on the item (usually at the bottom)?",
        options=tuple(v.value for v in YesNoUnsure),
    ),
    QuestionId.Q2_CLEANED: QuestionSpec(
        id=QuestionId.Q2_CLEANED,
        text="Is it empty, and did you rinse it? (e.g. a milk carton)",
        options=tuple(v.value for v in YesNoUnsure),
    ),
    QuestionId.Q3_SITTING_TIME: QuestionSpec(
        id=QuestionId.Q3_SITTING_TIME,
        text="How long has it been sitting (with food or drink in it)?",
        options=tuple(v.value for v in SittingTime),
    ),
}

# Opening question when the top material is not recyclable. For recyclables Q2 opens (D-004): the
# camera cannot see whether the item is clean.
OPENING_QUESTION: Final = QuestionId.Q1_RECYCLING_SYMBOL
