from __future__ import annotations

from dataclasses import dataclass, field

POC_NERF_SYNTHETIC_SCENES = (
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
)


@dataclass
class BackboneConfig:
    model_name: str = "da3-large"
    weights_path: str | None = None
    trainable: bool = False
    feature_index: int = -1
    use_camera_token: bool = False
    ref_view_strategy: str = "saddle_balanced"


@dataclass
class RendererConfig:
    num_heads: int = 8
    mlp_ratio: float = 4.0
    decoder_hidden_dim: int | None = None
    out_channels: int = 3
    use_query_ray_skip: bool = False
    use_support_rgb_skip: bool = False
    use_raw_rgb_stage1: bool = False
    skip_feature_dim: int = 64


@dataclass
class NerfSyntheticPOCConfig:
    image_size: int = 224
    support_views: int = 64
    train_query_views: int = 16
    eval_query_views: int = 16
    query_split: str = "val"
    scene_names: tuple[str, ...] = POC_NERF_SYNTHETIC_SCENES
    white_background: bool = True
    cache_images: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 1
    lr: float = 1e-4
    device: str = "cuda"


@dataclass
class DA3NVSConfig:
    patch_size: int = 14
    include_moment: bool = True
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    data: NerfSyntheticPOCConfig = field(default_factory=NerfSyntheticPOCConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
