"""Real-model path of app/vision.py, tested with a tiny hand-built ONNX model.

The fake network: feature map = 32x32 average-pool of the RGB channels (3 x 7 x 7), pooled =
global average, logits = W @ pooled. W maps red -> metal and blue -> plastic, so the expected
output can be computed independently in NumPy.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from app import main, vision
from app.settings import Settings, load_settings
from binbuddy_contracts.api import PredictResponse, VisionAnalysis
from binbuddy_contracts.constants import EMBEDDING_DIM
from binbuddy_contracts.enums import Material

CLASSES = [m.value for m in Material]
METAL, PLASTIC = CLASSES.index("metal"), CLASSES.index("plastic")
W = np.zeros((len(CLASSES), 3), dtype=np.float32)
W[METAL, 0] = 8.0  # red channel
W[PLASTIC, 2] = 8.0  # blue channel


def _write_model(folder: Path, **config_overrides: object) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    nodes = [
        helper.make_node(
            "AveragePool", ["image"], ["feature_map"], kernel_shape=[32, 32], strides=[32, 32]
        ),
        helper.make_node("GlobalAveragePool", ["feature_map"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["pooled"]),
        helper.make_node("Gemm", ["pooled", "W", "b"], ["logits"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, 224, 224])],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", len(CLASSES)]),
            helper.make_tensor_value_info("pooled", TensorProto.FLOAT, ["N", 3]),
            helper.make_tensor_value_info("feature_map", TensorProto.FLOAT, ["N", 3, 7, 7]),
        ],
        initializer=[
            numpy_helper.from_array(W, "W"),
            numpy_helper.from_array(np.zeros(len(CLASSES), np.float32), "b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, folder / "model.onnx")
    rng = np.random.default_rng(0)
    np.savez(
        folder / "head.npz",
        classifier_weight=W,
        classifier_bias=np.zeros(len(CLASSES), np.float32),
        pca_components=rng.normal(size=(EMBEDDING_DIM, 3)).astype(np.float32),
        pca_mean=np.zeros(3, np.float32),
    )
    config = {
        "classes": CLASSES,
        "input_size": 224,
        "resize": 256,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "temperature": 2.0,
        "embedding_dim": EMBEDDING_DIM,
        **config_overrides,
    }
    (folder / "config.json").write_text(json.dumps(config))
    return folder


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_model(tmp_path_factory.mktemp("model"))


@pytest.fixture
def real_settings(model_dir: Path) -> Settings:
    return replace(load_settings(), model_dir=model_dir, model_revision="v-test")


def _photo(colour: tuple[int, int, int], seed: int = 0) -> Image.Image:
    """Textured photo (passes the blur check) with a coloured block in the top-left quarter."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(60, 120, size=(300, 400, 3)).astype(np.uint8)
    arr[:150, :200] = colour
    return Image.fromarray(arr)


def test_predicts_material_from_the_onnx_model(real_settings: Settings) -> None:
    img = _photo((255, 0, 0))
    res = vision.analyse(img, real_settings)
    VisionAnalysis.model_validate(res.model_dump())  # contract-valid
    m = vision.get_model(real_settings)
    x, _ = m.preprocess(img.convert("RGB"))
    pooled = x[0].astype(np.float64).mean(axis=(1, 2))  # what the ONNX graph computes
    z = (W.astype(np.float64) @ pooled) / 2.0  # temperature 2.0
    expected = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    got = np.array([s.probability for s in res.material_scores])
    np.testing.assert_allclose(got, expected, atol=1e-5)
    assert max(res.material_scores, key=lambda s: s.probability).label is Material.METAL
    assert res.confidence == pytest.approx(float(expected.max()), abs=1e-5)


def test_condition_is_left_to_the_questions(real_settings: Settings) -> None:
    res = vision.analyse(_photo((0, 0, 255)), real_settings)
    assert {s.probability for s in res.condition_scores} == {0.25}
    assert max(res.material_scores, key=lambda s: s.probability).label is Material.PLASTIC


def test_embedding_is_unit_length_and_contract_sized(real_settings: Settings) -> None:
    e = np.array(vision.analyse(_photo((255, 0, 0)), real_settings).embedding)
    assert e.shape == (EMBEDDING_DIM,)
    assert np.linalg.norm(e) == pytest.approx(1.0, abs=1e-5)


def test_heatmap_points_at_the_coloured_block(real_settings: Settings) -> None:
    res = vision.analyse(_photo((255, 0, 0)), real_settings)
    assert res.heatmap_png_base64 is not None
    heat = Image.open(io.BytesIO(base64.b64decode(res.heatmap_png_base64)))
    assert heat.size == (224, 224)
    m = vision.get_model(real_settings)
    x, _ = m.preprocess(_photo((255, 0, 0)))
    fmap = np.stack([x[0, c].reshape(7, 32, 7, 32).mean(axis=(1, 3)) for c in range(3)])
    cam = vision.class_activation_map(fmap, W[METAL])
    assert np.unravel_index(cam.argmax(), cam.shape)[1] < 3  # left side, where the red block is


def test_head_gradient_through_layernorm_matches_finite_differences() -> None:
    """ConvNeXt-style head: y = w . (gamma * (p - mean) / s + beta). Check d y / d p numerically."""
    rng = np.random.default_rng(1)
    k, eps = 16, 1e-6
    p, w, gamma, beta = (rng.normal(size=k) for _ in range(4))

    def score(q: np.ndarray) -> float:
        x = (q - q.mean()) / np.sqrt(q.var() + eps)
        return float(w @ (gamma * x + beta))

    h = 1e-5
    numeric = np.array([(score(p + h * e) - score(p - h * e)) / (2 * h) for e in np.eye(k)])
    f32 = np.float32
    got = vision.head_gradient(p.astype(f32), w.astype(f32), gamma.astype(f32), eps)
    np.testing.assert_allclose(got, numeric, rtol=1e-3, atol=1e-4)
    # Linear head: the gradient is the class weight vector itself.
    np.testing.assert_array_equal(
        vision.head_gradient(p.astype(f32), w.astype(f32), None, eps), w.astype(f32)
    )


def test_preprocess_matches_resize_then_centre_crop(real_settings: Settings) -> None:
    m = vision.get_model(real_settings)
    x, crop = m.preprocess(Image.new("RGB", (400, 300)))  # landscape: height -> 256
    assert x.shape == (1, 3, 224, 224) and crop.size == (224, 224)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"embedding_dim": 128}, "embedding_dim"),
        ({"classes": list(reversed(CLASSES))}, "class order"),
    ],
)
def test_mismatched_export_is_refused(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    folder = _write_model(tmp_path / "bad", **override)
    with pytest.raises(vision.ModelUnavailableError, match=message):
        vision.VisionModel.load(folder)


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_api_serves_the_real_model(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    s = replace(load_settings(), model_dir=model_dir, model_revision="v-api")
    monkeypatch.setattr(main, "settings", lambda: s)
    client = TestClient(main.app)
    assert client.get("/health").json()["model_loaded"] is False  # loaded lazily
    body = client.post("/predict", json={"image_base64": _b64(_photo((255, 0, 0), seed=3))}).json()
    res = PredictResponse.model_validate(body)
    assert res.analysis is not None and res.analysis.heatmap_png_base64
    assert client.get("/health").json()["model_loaded"] is True


def test_missing_model_is_a_503(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    s = replace(load_settings(), model_dir=tmp_path / "nothing-here", model_revision="v-x")
    monkeypatch.setattr(main, "settings", lambda: s)
    r = TestClient(main.app).post("/predict", json={"image_base64": _b64(_photo((0, 0, 255)))})
    assert r.status_code == 503
    assert r.json()["code"] == "model_unavailable"


def test_conservative_fusion_averages_only_when_bin_groups_agree() -> None:
    c = np.array([0.6, 0.3, 0, 0, 0, 0, 0, 0, 0, 0.1], np.float32)  # top: paper (blue bin)
    agree = np.array([0.2, 0.7, 0, 0, 0, 0, 0, 0, 0, 0.1], np.float32)  # top: plastic (blue bin)
    disagree = np.array([0.1, 0.1, 0, 0, 0, 0.7, 0, 0, 0, 0.1], np.float32)  # top: styrofoam
    np.testing.assert_allclose(vision.conservative_fusion(c, agree), (c + agree) / 2)
    np.testing.assert_array_equal(vision.conservative_fusion(c, disagree), c)
