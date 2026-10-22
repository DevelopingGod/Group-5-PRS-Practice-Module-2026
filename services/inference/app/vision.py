"""Vision model (WS1): material probabilities, 256-d embedding and Grad-CAM heatmap.

Runs the export written by ml/notebooks/04_final_model_export.ipynb with ONNX Runtime on CPU
(no PyTorch in the API):

    model.onnx   image -> (logits, pooled features, last feature map)
    head.npz     classifier weights, the optional LayerNorm before them (both for the heatmap)
                 and the PCA projection (for the embedding)
    config.json  class order, preprocessing, temperature, embedding size

The files come from MODEL_DIR (local disk, for the free offline setup) or from the Hugging Face
Hub repo MODEL_REPO at the pinned tag MODEL_REVISION. MODEL_REVISION=stub keeps the
deterministic fake model, so tests and the other workstreams never need the real weights.

The model predicts **material only**. Cleanliness comes from the question bank (Q2/Q3), so the
condition scores are an uninformative uniform prior, not a prediction.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from app.quality import average_hash
from app.settings import Settings
from binbuddy_contracts import examples
from binbuddy_contracts.api import ConditionScore, MaterialScore, VisionAnalysis
from binbuddy_contracts.constants import EMBEDDING_DIM
from binbuddy_contracts.enums import Condition, Material

Array = npt.NDArray[np.float32]
STUB_REVISION = "stub"
MODEL_FILES = ("config.json", "model.onnx", "head.npz")
FUSION_FILE = "swin.onnx"  # optional second model (WS2 conservative fusion)
BLUE_IDS = frozenset(range(5))  # paper, plastic, glass, metal, beverage_carton
HEATMAP_ALPHA = 0.45  # weight of the heatmap colours over the photo


class ModelUnavailableError(RuntimeError):
    """The real model could not be downloaded or loaded."""


@dataclass(frozen=True)
class VisionModel:
    session: Any  # onnxruntime.InferenceSession (untyped library)
    input_size: int
    resize: int
    mean: Array
    std: Array
    temperature: float
    classifier_weight: Array  # classes x K
    pca_components: Array  # EMBEDDING_DIM x K
    pca_mean: Array  # K
    norm_weight: Array | None  # K; set when the head is pool -> LayerNorm -> linear (ConvNeXt)
    norm_eps: float
    swin_session: Any = None  # set when swin.onnx is in the model folder
    swin_temperature: float = 1.0

    @classmethod
    def load(cls, folder: Path) -> VisionModel:
        import onnxruntime as ort

        missing = [f for f in MODEL_FILES if not (folder / f).exists()]
        if missing:
            raise ModelUnavailableError(f"model files missing in {folder}: {missing}")
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if cfg["classes"] != [m.value for m in Material]:
            raise ModelUnavailableError("model class order does not match the contract")
        if cfg["embedding_dim"] != EMBEDDING_DIM:
            raise ModelUnavailableError(
                f"model embedding_dim {cfg['embedding_dim']} != contract {EMBEDDING_DIM}"
            )
        head = np.load(folder / "head.npz")
        session = ort.InferenceSession(
            str(folder / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        return cls(
            session=session,
            input_size=int(cfg["input_size"]),
            resize=int(cfg["resize"]),
            mean=np.asarray(cfg["mean"], dtype=np.float32),
            std=np.asarray(cfg["std"], dtype=np.float32),
            temperature=float(cfg["temperature"]),
            classifier_weight=head["classifier_weight"].astype(np.float32),
            pca_components=head["pca_components"].astype(np.float32),
            pca_mean=head["pca_mean"].astype(np.float32),
            norm_weight=head["norm_weight"].astype(np.float32) if "norm_weight" in head else None,
            norm_eps=float(cfg.get("head_norm_eps") or 1e-6),
            swin_session=(
                ort.InferenceSession(str(folder / FUSION_FILE), providers=["CPUExecutionProvider"])
                if (folder / FUSION_FILE).exists()
                else None
            ),
            swin_temperature=float(cfg.get("swin_temperature", 1.0)),
        )

    def preprocess(self, img: Image.Image) -> tuple[Array, Image.Image]:
        """Same as the notebooks' torchvision Resize(resize) + CenterCrop(input_size)."""
        w, h = img.size
        if w <= h:
            size = (self.resize, int(self.resize * h / w))
        else:
            size = (int(self.resize * w / h), self.resize)
        resized = img.resize(size, Image.Resampling.BILINEAR)
        s = self.input_size
        left = round((size[0] - s) / 2.0)
        top = round((size[1] - s) / 2.0)
        crop = resized.crop((left, top, left + s, top + s))
        x = np.asarray(crop, dtype=np.float32) / 255.0
        x = (x - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32), crop

    def analyse(self, img: Image.Image) -> VisionAnalysis:
        x, crop = self.preprocess(img)
        logits, pooled, fmap = self.session.run(None, {"image": x})
        probs = softmax(np.asarray(logits[0], dtype=np.float32) / self.temperature)
        if self.swin_session is not None:  # same preprocessing and normalisation as ConvNeXt
            (swin_logits,) = self.swin_session.run(None, {"image": x})
            swin = softmax(np.asarray(swin_logits[0], dtype=np.float32) / self.swin_temperature)
            probs = conservative_fusion(probs, swin)
        top = int(probs.argmax())
        feature_map = np.asarray(fmap[0], dtype=np.float32)
        grad = head_gradient(
            feature_map.mean(axis=(1, 2)),
            self.classifier_weight[top],
            self.norm_weight,
            self.norm_eps,
        )
        heatmap = heatmap_png_base64(feature_map, grad, crop)
        return VisionAnalysis(
            material_scores=[
                MaterialScore(label=m, probability=float(p))
                for m, p in zip(Material, probs, strict=True)
            ],
            condition_scores=uniform_condition_scores(),
            confidence=float(probs[top]),
            is_ood=False,  # WS2 adds out-of-distribution detection on the embedding
            embedding=self.embed(np.asarray(pooled[0], dtype=np.float32)),
            phash=average_hash(img),
            heatmap_png_base64=heatmap,
        )

    def embed(self, pooled: Array) -> list[float]:
        e = self.pca_components @ (pooled - self.pca_mean)
        return [float(v) for v in e / np.linalg.norm(e)]


def conservative_fusion(convnext: Array, swin: Array) -> Array:
    """WS2's selected ensemble: average the two models when they agree on blue-bin vs other
    material, otherwise keep ConvNeXt (the safer, better-calibrated model). On the 395 team test
    photos: macro-F1 0.830 vs 0.799 for ConvNeXt alone, with the same costly errors (15 of 108).
    """
    if (int(convnext.argmax()) in BLUE_IDS) != (int(swin.argmax()) in BLUE_IDS):
        return convnext
    out: Array = (0.5 * (convnext + swin)).astype(np.float32)
    return out


def softmax(z: Array) -> Array:
    e = np.exp(z - z.max())
    out: Array = (e / e.sum()).astype(np.float32)
    return out


def uniform_condition_scores() -> list[ConditionScore]:
    p = 1.0 / len(Condition)
    return [ConditionScore(label=c, probability=p) for c in Condition]


def head_gradient(pooled: Array, weight: Array, norm_weight: Array | None, eps: float) -> Array:
    """d(class score) / d(pooled features): the per-channel Grad-CAM weights, up to 1/Z.

    The head is global average pooling -> [LayerNorm] -> linear. Every pixel of feature map k
    feeds the score only through the pooled value p_k, so Grad-CAM's channel weight is this
    gradient (notebook 03, section 5). Linear head: the gradient is just the class weights w.
    LayerNorm first (ConvNeXt), with x = (p - mean) / s and v = gamma * w:
        dy/dp = (v - mean(v) - x * mean(v * x)) / s
    """
    if norm_weight is None:
        return weight
    p = pooled.astype(np.float64)
    s = np.sqrt(p.var() + eps)
    x = (p - p.mean()) / s
    v = norm_weight.astype(np.float64) * weight
    grad: Array = ((v - v.mean() - x * (v * x).mean()) / s).astype(np.float32)
    return grad


def class_activation_map(fmap: Array, weight: Array) -> Array:
    """ReLU(sum_k w_k A_k), scaled to [0, 1]. With w = head_gradient(...) this is Grad-CAM."""
    cam = np.maximum(np.tensordot(weight, fmap, axes=1), 0.0)
    cam -= cam.min()
    peak = cam.max()
    return (cam / peak if peak > 0 else cam).astype(np.float32)


def _jet(v: Array) -> Array:
    """Blue (0) -> green -> red (1) colour map, without matplotlib or OpenCV."""
    channels = [np.clip(1.5 - np.abs(4.0 * v - c), 0.0, 1.0) for c in (3.0, 2.0, 1.0)]
    out: Array = np.stack(channels, axis=-1).astype(np.float32)
    return out


def heatmap_png_base64(fmap: Array, weight: Array, crop: Image.Image) -> str:
    cam = class_activation_map(fmap, weight)
    up = Image.fromarray(cam).resize(crop.size, Image.Resampling.BILINEAR)  # float32: mode "F"
    colours = _jet(np.clip(np.asarray(up, dtype=np.float32), 0.0, 1.0)) * 255.0
    photo = np.asarray(crop.convert("RGB"), dtype=np.float32)
    blended = (1 - HEATMAP_ALPHA) * photo + HEATMAP_ALPHA * colours
    buf = io.BytesIO()
    Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ loading and dispatch

_lock = threading.Lock()
_loaded: dict[tuple[str, str, str], VisionModel] = {}


def _key(s: Settings) -> tuple[str, str, str]:
    return (s.model_repo, s.model_revision, str(s.model_dir or ""))


def _model_folder(s: Settings) -> Path:
    if s.model_dir is not None:
        return s.model_dir
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                s.model_repo,
                revision=s.model_revision,
                allow_patterns=[*MODEL_FILES, FUSION_FILE],
                token=s.hf_token,
            )
        )
    except Exception as exc:  # network, auth or missing tag: all mean "no model"
        raise ModelUnavailableError(
            f"could not download {s.model_repo}@{s.model_revision}"
        ) from exc


def uses_stub(s: Settings) -> bool:
    return s.model_dir is None and s.model_revision == STUB_REVISION


def get_model(s: Settings) -> VisionModel:
    """Load once (lazily, on the first photo) and cache; /health stays fast."""
    key = _key(s)
    with _lock:
        if key not in _loaded:
            _loaded[key] = VisionModel.load(_model_folder(s))
        return _loaded[key]


def is_loaded(s: Settings) -> bool:
    return uses_stub(s) or _key(s) in _loaded


def analyse(img: Image.Image, s: Settings) -> VisionAnalysis:
    if uses_stub(s):
        return stub_analyse(img)
    return get_model(s).analyse(img)


def stub_analyse(img: Image.Image) -> VisionAnalysis:
    """Deterministic, contract-valid fake prediction (no model files needed)."""
    digest = hashlib.sha256(img.tobytes()).digest()
    materials, conditions = list(Material), list(Condition)
    material = materials[digest[0] % len(materials)]
    condition = conditions[digest[1] % len(conditions)]
    confidence = 0.5 + (digest[2] % 50) / 100.0  # 0.50-0.99
    return VisionAnalysis(
        material_scores=examples.material_scores(material, 0.7),
        condition_scores=examples.condition_scores(condition, 0.7),
        confidence=confidence,
        is_ood=digest[3] % 20 == 0,
        embedding=examples.unit_embedding(seed=digest[4]),
        phash=average_hash(img),
    )
