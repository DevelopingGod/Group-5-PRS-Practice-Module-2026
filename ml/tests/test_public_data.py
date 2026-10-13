from collections import Counter

import pytest

from binbuddy_contracts.enums import Material, Split
from binbuddy_ml import experiments, public_data
from binbuddy_ml.public_data import (
    LOCAL_SPLIT,
    PUBLIC_SPLIT,
    SplitFractions,
    split_for_group,
    taco_box_usable,
)


def test_taco_mapping_covers_all_60_categories() -> None:
    assert len(public_data.TACO) == 60


def test_every_taco_exclusion_has_a_reason() -> None:
    excluded = {name for name, m in public_data.TACO.items() if m is None}
    assert excluded == set(public_data.TACO_EXCLUDED)


def test_e_waste_has_almost_no_public_data() -> None:
    cov = public_data.coverage(public_data.TRASHNET, public_data.REALWASTE, public_data.TACO)
    assert cov[Material.E_WASTE] == ["Battery"]  # why local photos and zero-shot CLIP matter
    assert all(cov[m] for m in Material)


def test_taco_box_filter() -> None:
    assert taco_box_usable(200, 300, 2000, 1500)
    assert not taco_box_usable(50, 300, 2000, 1500)  # too thin
    assert not taco_box_usable(100, 100, 4000, 3000)  # too small a share of the photo


def test_split_is_deterministic_and_grouped() -> None:
    assert split_for_group("taco-123", PUBLIC_SPLIT) is split_for_group("taco-123", PUBLIC_SPLIT)


def test_split_proportions() -> None:
    counts = Counter(split_for_group(f"g{i}", LOCAL_SPLIT) for i in range(20_000))
    assert counts[Split.TRAIN] / 20_000 == pytest.approx(0.50, abs=0.02)
    assert counts[Split.TEST] / 20_000 == pytest.approx(0.30, abs=0.02)
    assert Split.FIELD not in counts


def test_bad_fractions_rejected() -> None:
    with pytest.raises(ValueError):
        SplitFractions(0.5, 0.5, 0.5)


def test_model_registry() -> None:
    keys = [m.key for m in experiments.MODELS]
    assert len(keys) == len(set(keys))
    fine_tuned = [m for m in experiments.MODELS if m.mode is experiments.Mode.FINE_TUNE]
    assert len(fine_tuned) >= 5
    assert all(m.backbone_lr > 0 for m in fine_tuned)
    assert experiments.model("efficientnet_b0").params_m == 5.3


def test_zero_shot_prompts_cover_every_material() -> None:
    assert set(experiments.ZERO_SHOT_PROMPTS) == {m.value for m in Material}
