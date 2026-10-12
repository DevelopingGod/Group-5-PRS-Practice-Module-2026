"""Bin Buddy shared contracts.

Single source of truth for every data shape that crosses a workstream boundary:
the model API (WS1/WS3 produce, WS4 consumes), site policies (WS3), the dataset
manifest (everyone), QR codes on bins (WS4) and embeddings (WS1 -> WS2/WS4).

Changing anything in this package is a *contract change*: bump CONTRACTS_VERSION,
add an entry to contracts/CHANGELOG.md, regenerate artefacts, and get approval
from both the producing and the consuming workstream (see CONTRIBUTING.md).
"""

from binbuddy_contracts.constants import CONTRACTS_VERSION

__all__ = ["CONTRACTS_VERSION"]
