import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


def image_b64(seed: int = 0, dark: bool = False, flat: bool = False) -> str:
    rng = np.random.default_rng(seed)
    if flat:
        arr = np.full((128, 128, 3), 128, dtype=np.uint8)
    else:
        arr = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    if dark:
        arr = (arr * 0.05).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
