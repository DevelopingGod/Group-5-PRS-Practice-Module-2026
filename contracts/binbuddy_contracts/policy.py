"""Site policy contract (WS3 owns the engine; YAML files live in policies/).

Semantics: rules are checked top to bottom and the FIRST matching rule wins.
Every (material, condition) pair must be matched by some rule; validation fails otherwise.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml
from pydantic import Field, model_validator

from binbuddy_contracts.api import SLUG_PATTERN, Strict
from binbuddy_contracts.enums import Channel, Condition, Material


class BinSpec(Strict):
    key: str = Field(pattern=SLUG_PATTERN)
    label: str = Field(min_length=1)
    channel: Channel
    colour: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")

    @model_validator(mode="after")
    def _not_abstain(self) -> BinSpec:
        if self.channel is Channel.ABSTAIN:
            raise ValueError("a physical bin cannot have the 'abstain' channel")
        return self


class RuleWhen(Strict):
    """Omitted list = matches any value."""

    materials: list[Material] | None = None
    conditions: list[Condition] | None = None

    def matches(self, material: Material, condition: Condition) -> bool:
        return (self.materials is None or material in self.materials) and (
            self.conditions is None or condition in self.conditions
        )


class PolicyRule(Strict):
    id: str = Field(pattern=SLUG_PATTERN)
    when: RuleWhen
    then: str = Field(pattern=SLUG_PATTERN, description="Bin key")
    reason: str = Field(min_length=1)
    tip: str | None = None
    source_url: str | None = None


class SitePolicy(Strict):
    site: str = Field(pattern=SLUG_PATTERN)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str
    bins: list[BinSpec] = Field(min_length=1)
    rules: list[PolicyRule] = Field(min_length=1)
    uncertain_fallback: str = Field(
        pattern=SLUG_PATTERN, description="Bin used when still uncertain after all questions"
    )
    abstain_source_url: str = Field(description="Link shown when the system abstains")

    @model_validator(mode="after")
    def _integrity(self) -> SitePolicy:
        keys = [b.key for b in self.bins]
        if len(keys) != len(set(keys)):
            raise ValueError("bin keys must be unique")
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("rule ids must be unique")
        for r in self.rules:
            if r.then not in keys:
                raise ValueError(f"rule '{r.id}' points to unknown bin '{r.then}'")
        if self.uncertain_fallback not in keys:
            raise ValueError(f"uncertain_fallback '{self.uncertain_fallback}' is not a bin")
        if self.bin(self.uncertain_fallback).channel is not Channel.GENERAL_WASTE:
            raise ValueError("uncertain_fallback must be a general_waste bin (proposal 1.2)")
        uncovered = [
            f"{m.value}/{c.value}"
            for m, c in product(Material, Condition)
            if not any(r.when.matches(m, c) for r in self.rules)
        ]
        if uncovered:
            raise ValueError(f"no rule covers: {', '.join(uncovered)}")
        return self

    def bin(self, key: str) -> BinSpec:
        for b in self.bins:
            if b.key == key:
                return b
        raise KeyError(key)

    def first_match(self, material: Material, condition: Condition) -> PolicyRule:
        """Reference semantics. WS3's engine must return the same rule for every pair."""
        for rule in self.rules:
            if rule.when.matches(material, condition):
                return rule
        raise LookupError(f"no rule for {material}/{condition}")  # unreachable after validation


def load_policy(path: Path) -> SitePolicy:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return SitePolicy.model_validate(data)


def load_policies(directory: Path) -> dict[str, SitePolicy]:
    policies: dict[str, SitePolicy] = {}
    for path in sorted(directory.glob("*.yaml")):
        policy = load_policy(path)
        if path.stem != policy.site:
            raise ValueError(f"{path.name}: file name must equal site '{policy.site}'")
        policies[policy.site] = policy
    return policies
