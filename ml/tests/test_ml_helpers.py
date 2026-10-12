from pathlib import Path

import pytest

from binbuddy_ml.labels import load_labels
from binbuddy_ml.paths import data_root, photo_path


def test_manifest_loads() -> None:
    assert isinstance(load_labels(), list)


def test_missing_data_root_explains_itself(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BINBUDDY_DATA_ROOT", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="Google Drive"):
        data_root()


def test_photo_path(tmp_path: Path) -> None:
    p = photo_path("local/ws1-0001/ws1-0001-a.jpg", root=tmp_path)
    assert p == tmp_path / "local/ws1-0001/ws1-0001-a.jpg"
