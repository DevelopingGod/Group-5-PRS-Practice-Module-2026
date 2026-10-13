"""The WS1 model comparison: which models, why, and with which hyperparameters.

Single source of truth for ml/notebooks/02_model_comparison.ipynb. Every model is trained and
evaluated under the same protocol (same data, splits, input size, augmentation, schedule and
metrics), so differences in the results come from the model, not from the setup.

Model ids are timm Hub ids (verified on the Hugging Face Hub, 13 Oct 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Mode(StrEnum):
    FINE_TUNE = "fine_tune"  # train a new head, then unfreeze and train the whole network
    LINEAR_PROBE = "linear_probe"  # frozen backbone -> embedding -> logistic regression
    ZERO_SHOT = "zero_shot"  # no training: compare image and text embeddings (CLIP)


@dataclass(frozen=True)
class Protocol:
    """Hyperparameters shared by every fine-tuned model (the controlled variables)."""

    input_size: int = 224
    batch_size: int = 64
    head_epochs: int = 2  # stage 1: backbone frozen, only the new head learns
    full_epochs: int = 12  # stage 2: whole network, smaller learning rate for the backbone
    head_lr: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_fraction: float = 0.1  # of stage-2 steps; then cosine decay to 0
    early_stop_patience: int = 3  # stage-2 epochs without a better val macro-F1
    class_weight_power: float = 0.5  # loss weight = (1 / class frequency) ** power
    seed: int = 42
    mixed_precision: bool = True


PROTOCOL = Protocol()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    timm_id: str
    family: str
    mode: Mode
    params_m: float  # backbone parameters, millions (Hub model card)
    backbone_lr: float  # stage-2 learning rate for pretrained weights
    what: str  # what the architecture does, in one or two sentences
    why: str  # why it is in this comparison
    batch_size: int = PROTOCOL.batch_size
    timm_kwargs: tuple[tuple[str, object], ...] = ()


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="mobilenetv3_l",
        timm_id="mobilenetv3_large_100.ra_in1k",
        family="Mobile CNN",
        mode=Mode.FINE_TUNE,
        params_m=5.5,
        backbone_lr=3e-4,
        what="Inverted-residual blocks with depthwise convolutions, squeeze-and-excitation "
        "and hard-swish; architecture found by neural architecture search for phone CPUs.",
        why="Named in the proposal. The speed floor: if it matches the larger models on our "
        "data, it is the best choice for a free, CPU-only, local deployment.",
    ),
    ModelSpec(
        key="efficientnet_b0",
        timm_id="efficientnet_b0.ra_in1k",
        family="Mobile CNN",
        mode=Mode.FINE_TUNE,
        params_m=5.3,
        backbone_lr=3e-4,
        what="MBConv blocks scaled jointly in depth, width and resolution (compound scaling); "
        "strong accuracy per FLOP.",
        why="Named in the proposal. Same size as MobileNetV3, so the pair isolates the effect "
        "of compound scaling versus latency-oriented search.",
    ),
    ModelSpec(
        key="resnet50",
        timm_id="resnet50.a1_in1k",
        family="Classic CNN",
        mode=Mode.FINE_TUNE,
        params_m=25.6,
        backbone_lr=1e-4,
        what="Residual (skip) connections that let very deep plain convolutional stacks "
        "train; bottleneck 1x1-3x3-1x1 blocks.",
        why="The standard transfer-learning reference point in the waste-classification "
        "literature; anchors our numbers against published work.",
    ),
    ModelSpec(
        key="convnext_t",
        timm_id="convnext_tiny.fb_in22k_ft_in1k",
        family="Modern CNN",
        mode=Mode.FINE_TUNE,
        params_m=28.6,
        backbone_lr=1e-4,
        what="A ResNet modernised with transformer design choices: 7x7 depthwise kernels, "
        "LayerNorm, GELU, fewer activations; pretrained on ImageNet-22k.",
        why="Tests whether larger pretraining data plus modern design beats classic CNNs at "
        "the same size as ResNet-50, while keeping Grad-CAM straightforward.",
    ),
    ModelSpec(
        key="swin_t",
        timm_id="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        family="Vision transformer (hierarchical)",
        mode=Mode.FINE_TUNE,
        params_m=28.5,
        backbone_lr=5e-5,
        what="Self-attention inside shifted local windows, with patch merging that builds a "
        "CNN-like feature pyramid; pretrained on ImageNet-22k.",
        why="Attention-based counterpart to ConvNeXt at equal size and pretraining, so the "
        "pair isolates convolution versus attention.",
    ),
    ModelSpec(
        key="deit_s",
        timm_id="deit_small_patch16_224.fb_in1k",
        family="Vision transformer (plain)",
        mode=Mode.FINE_TUNE,
        params_m=22.1,
        backbone_lr=5e-5,
        what="Plain ViT: the image is cut into 16x16 patches, each a token; global "
        "self-attention over all tokens; trained data-efficiently on ImageNet-1k only.",
        why="Shows how a transformer with weak spatial priors copes with a small dataset "
        "(expected to need more data than the CNNs).",
        batch_size=48,
    ),
    ModelSpec(
        key="dinov2_s_probe",
        timm_id="vit_small_patch14_dinov2.lvd142m",
        family="Self-supervised foundation model",
        mode=Mode.LINEAR_PROBE,
        params_m=22.1,
        backbone_lr=0.0,
        what="ViT trained without labels on 142M curated images by self-distillation; its "
        "embedding is used frozen, with a logistic-regression classifier on top.",
        why="'Vectorisation' approach: no fine-tuning, so it cannot overfit our few local "
        "photos; the same embedding can serve verification and out-of-distribution checks.",
        timm_kwargs=(("img_size", 224),),
    ),
    ModelSpec(
        key="clip_b16_probe",
        timm_id="vit_base_patch16_clip_224.openai",
        family="Language-supervised foundation model",
        mode=Mode.LINEAR_PROBE,
        params_m=86.0,
        backbone_lr=0.0,
        what="ViT trained on 400M image-caption pairs to match images with their text; "
        "frozen embedding plus logistic regression.",
        why="Tests whether language-aligned features separate materials better than "
        "ImageNet features; pairs with the zero-shot run below.",
    ),
    ModelSpec(
        key="clip_b16_zeroshot",
        timm_id="ViT-B-16/openai (open_clip)",
        family="Language-supervised foundation model",
        mode=Mode.ZERO_SHOT,
        params_m=86.0,
        backbone_lr=0.0,
        what="No training at all: each material is described in text prompts, and the "
        "image is assigned to the closest prompt embedding.",
        why="The only option that can recognise e-waste, which has no public training "
        "images; also a proxy for the chatbot-style baseline in the proposal.",
    ),
)

# Prompts for zero-shot CLIP; several templates per class are averaged (prompt ensembling).
ZERO_SHOT_PROMPTS: dict[str, tuple[str, ...]] = {
    "paper": ("paper", "a cardboard box", "a newspaper", "a paper bag"),
    "plastic": ("a plastic bottle", "a plastic container", "a plastic bag", "a plastic cup"),
    "glass": ("a glass bottle", "a glass jar"),
    "metal": ("an aluminium can", "a tin can", "a metal lid", "aluminium foil"),
    "beverage_carton": ("a milk carton", "a juice carton", "a drink carton"),
    "styrofoam": ("a styrofoam box", "a foam cup", "polystyrene foam packaging"),
    "e_waste": ("a phone charger", "a light bulb", "a battery", "a small electronic device"),
    "textile": ("old clothing", "a t-shirt", "a shoe", "fabric"),
    "food_waste": ("food waste", "a banana peel", "leftover food", "fruit peels"),
    "other_general": ("used tissues", "a snack wrapper", "general rubbish"),
}
PROMPT_TEMPLATES: tuple[str, ...] = (
    "a photo of {}.",
    "a photo of {} to throw away.",
    "a close-up photo of {}.",
)


def model(key: str) -> ModelSpec:
    for spec in MODELS:
        if spec.key == key:
            return spec
    raise KeyError(key)
