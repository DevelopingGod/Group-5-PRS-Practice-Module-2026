"""Where the photos are. Photos live on Google Drive, never in git.

In Colab:
    from google.colab import drive; drive.mount("/content/drive")
    # the team folder must be added to "My Drive" as a shortcut named BinBuddy_Data
Locally: set BINBUDDY_DATA_ROOT to a synced copy of the Drive folder.
"""

from __future__ import annotations

import os
from pathlib import Path

COLAB_DEFAULT = Path("/content/drive/MyDrive/BinBuddy_Data")


def data_root() -> Path:
    env = os.getenv("BINBUDDY_DATA_ROOT")
    root = Path(env) if env else COLAB_DEFAULT
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data folder not found at {root}. Mount Google Drive in Colab, or set "
            "BINBUDDY_DATA_ROOT to a local copy of the BinBuddy_Data folder."
        )
    return root


def photo_path(drive_path: str, root: Path | None = None) -> Path:
    """Resolve a drive_path from data/labels.csv to a file on disk."""
    return (root or data_root()) / drive_path
