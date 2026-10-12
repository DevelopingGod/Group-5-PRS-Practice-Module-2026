from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from binbuddy_contracts.enums import Channel, Condition, Material
from binbuddy_contracts.policy import SitePolicy, load_policies

POLICIES = Path(__file__).resolve().parents[2] / "policies"


@pytest.fixture(scope="module")
def hdb() -> SitePolicy:
    return load_policies(POLICIES)["hdb_default"]


def test_all_policies_valid() -> None:
    assert {"hdb_default", "campus_example"} <= set(load_policies(POLICIES))


@pytest.mark.parametrize(
    ("material", "condition", "bin_key", "has_tip"),
    [
        # Edge cases listed in proposal 1.3
        (Material.PLASTIC, Condition.ITEM_INSIDE, "general_waste", True),  # banana peel inside
        (Material.BEVERAGE_CARTON, Condition.LEFTOVER_LIQUID, "general_waste", True),
        (Material.PAPER, Condition.FOOD_RESIDUE, "general_waste", False),  # stained paper
        (Material.STYROFOAM, Condition.CLEAN, "general_waste", False),
        (Material.E_WASTE, Condition.CLEAN, "e_waste_point", False),
        (Material.TEXTILE, Condition.CLEAN, "donation", False),
        (Material.METAL, Condition.CLEAN, "blue_bin", False),  # clean aluminium can
    ],
)
def test_hdb_edge_cases(
    hdb: SitePolicy, material: Material, condition: Condition, bin_key: str, has_tip: bool
) -> None:
    rule = hdb.first_match(material, condition)
    assert rule.then == bin_key
    assert (rule.tip is not None) == has_tip


def test_only_clean_recyclables_reach_the_blue_bin(hdb: SitePolicy) -> None:
    for m in Material:
        for c in Condition:
            if hdb.bin(hdb.first_match(m, c).then).channel is Channel.RECYCLE:
                assert c is Condition.CLEAN, f"{m}/{c} must not go to the blue bin"


def _raw() -> dict[str, object]:
    with (POLICIES / "hdb_default.yaml").open() as fh:
        data: dict[str, object] = yaml.safe_load(fh)
    return data


def test_uncovered_pair_is_rejected() -> None:
    data = _raw()
    rules = data["rules"]
    assert isinstance(rules, list)
    data["rules"] = [r for r in rules if r["id"] != "styrofoam"]
    with pytest.raises(ValidationError, match="no rule covers"):
        SitePolicy.model_validate(data)


def test_unknown_bin_is_rejected() -> None:
    data = _raw()
    rules = data["rules"]
    assert isinstance(rules, list)
    rules[0]["then"] = "purple_bin"
    with pytest.raises(ValidationError, match="unknown bin"):
        SitePolicy.model_validate(data)
