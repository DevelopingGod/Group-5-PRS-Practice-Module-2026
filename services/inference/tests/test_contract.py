import json
from pathlib import Path

from app.main import app
from binbuddy_contracts.openapi import contract_differences

GENERATED = Path(__file__).resolve().parents[3] / "contracts" / "generated" / "openapi.json"


def test_service_matches_contract_openapi() -> None:
    contract = json.loads(GENERATED.read_text(encoding="utf-8"))
    assert contract_differences(contract, app.openapi()) == []
