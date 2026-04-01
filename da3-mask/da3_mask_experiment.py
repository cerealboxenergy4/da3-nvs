from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from da3_nvs.config import POC_NERF_SYNTHETIC_SCENES
from da3_nvs.data.nerf_synthetic import (
    BlenderFrame,
    BlenderScene,
    _camera_intrinsics,
    _load_rgb_image,
    _load_scene,
    _select_frames,
)
from da3_nvs.models import CrossAttentionNVSHead, DA3PatchBackbone
from da3_nvs.models.common import num_patches, patchify, unpatchify


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / "nerf_synthetic"


@dataclass
class MaskReconstructionBatch:
    support_images: torch.Tensor
    support_intrinsics: torch.Tensor
    support_c2w: torch.Tensor
    train_query_images: torch.Tensor
    train_query_intrinsics: torch.Tensor
    train_query_c2w: torch.Tensor
    eval_query_images: torch.Tensor
    eval_query_intrinsics: torch.Tensor
    eval_query_c2w: torch.Tensor


@dataclass
class MaskedPatchOutputs:
    pred_patch_rgb: torch.Tensor
    target_patch_rgb: torch.Tensor
    query_patch_mask: torch.Tensor
    masked_query_images: torch.Tensor
    pred_query_images: torch.Tensor
    reconstructed_query_images: torch.Tensor
    query_tokens: torch.Tensor


@dataclass(frozen=True)
class CO3DFrameAnnotation:
    category_name: str
    sequence_name: str
    frame_number: int
    image_path: Path
    mask_path: Path | None
    intrinsics: torch.Tensor
    c2w: torch.Tensor


class ToyPatchBackbone(nn.Module):
    def __init__(self, *, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        *,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del extrinsics, intrinsics
        batch_size, num_views, _, height, width = images.shape
        flat_images = images.reshape(batch_size * num_views, 3, height, width)
        tokens = self.norm(self.proj(patchify(flat_images, self.patch_size)))
        return tokens.reshape(batch_size, num_views, tokens.shape[1], tokens.shape[2])


class ToyMultiStageBackbone(ToyPatchBackbone):
    def forward(
        self,
        images: torch.Tensor,
        *,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        tokens = super().forward(images, extrinsics=extrinsics, intrinsics=intrinsics)
        return [
            tokens,
            tokens + 0.01,
            tokens + 0.02,
            tokens + 0.03,
        ]


class LightweightCNNDecoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        hidden_dim: int,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, max(hidden_dim // 2, 16), kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(max(hidden_dim // 2, 16), out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        image_size: tuple[int, int],
        patch_size: int,
    ) -> torch.Tensor:
        if tokens.dim() != 4:
            raise ValueError("tokens must have shape (B, V, P, D)")

        batch_size, num_views, num_patches, embed_dim = tokens.shape
        patch_height = image_size[0] // patch_size
        patch_width = image_size[1] // patch_size
        if patch_height * patch_width != num_patches:
            raise ValueError(
                "Token count does not match image patch grid: "
                f"{num_patches} vs {patch_height}x{patch_width}"
            )

        flat_tokens = self.token_norm(tokens.reshape(batch_size * num_views, num_patches, embed_dim))
        feature_map = flat_tokens.transpose(1, 2).reshape(batch_size * num_views, embed_dim, patch_height, patch_width)
        feature_map = self.proj(feature_map)
        feature_map = self.block1(feature_map)
        feature_map = F.interpolate(
            feature_map,
            size=(max(image_size[0] // 2, 1), max(image_size[1] // 2, 1)),
            mode="bilinear",
            align_corners=True,
        )
        feature_map = self.block2(feature_map)
        feature_map = F.interpolate(
            feature_map,
            size=image_size,
            mode="bilinear",
            align_corners=True,
        )
        pred_rgb = torch.sigmoid(self.head(feature_map))
        return pred_rgb.reshape(batch_size, num_views, pred_rgb.shape[1], image_size[0], image_size[1])


class MaskReconstructionDataset(Dataset[MaskReconstructionBatch]):
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        scene_names: tuple[str, ...] | None = POC_NERF_SYNTHETIC_SCENES,
        image_size: int = 224,
        support_views: int = 16,
        train_query_views: int = 4,
        eval_query_views: int = 4,
        eval_split: str = "val",
        size: int = 128,
        white_background: bool = True,
        cache_images: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else _default_root()
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")
        if eval_split not in {"val", "test"}:
            raise ValueError("eval_split must be one of {'val', 'test'}")

        scene_dirs = sorted(
            path for path in self.root.iterdir() if (path / "transforms_train.json").exists()
        )
        scene_map = {scene_dir.name: _load_scene(scene_dir) for scene_dir in scene_dirs}
        if scene_names is None:
            self.scenes = tuple(scene_map[name] for name in sorted(scene_map))
        else:
            missing = [name for name in scene_names if name not in scene_map]
            if missing:
                raise ValueError(f"Unknown scenes requested: {missing}")
            self.scenes = tuple(scene_map[name] for name in scene_names)
        if not self.scenes:
            raise ValueError(f"no Blender scenes found under {self.root}")

        self.image_size = image_size
        self.support_views = support_views
        self.train_query_views = train_query_views
        self.eval_query_views = eval_query_views
        self.eval_split = eval_split
        self.size = max(size, len(self.scenes))
        self.white_background = white_background
        self.cache_images = cache_images
        self._image_cache: dict[Path, torch.Tensor] = {}

    def __len__(self) -> int:
        return self.size

    def _load_image(self, image_path: Path) -> torch.Tensor:
        if self.cache_images and image_path in self._image_cache:
            return self._image_cache[image_path]

        image = _load_rgb_image(
            image_path,
            self.image_size,
            white_background=self.white_background,
        )
        if self.cache_images:
            self._image_cache[image_path] = image
        return image

    def _build_bundle(
        self,
        scene: BlenderScene,
        frame_pool: tuple[BlenderFrame, ...],
        view_count: int,
        *,
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_frames = _select_frames(frame_pool, view_count, offset=offset)
        intrinsics = _camera_intrinsics(
            scene.camera_angle_x,
            scene.original_size,
            self.image_size,
        )
        view_intrinsics = intrinsics.expand(len(selected_frames), -1, -1).clone()
        c2w = torch.stack([frame.c2w for frame in selected_frames], dim=0)
        images = torch.stack([self._load_image(frame.image_path) for frame in selected_frames], dim=0)
        return images, view_intrinsics, c2w

    def _eval_pool(self, scene: BlenderScene) -> tuple[BlenderFrame, ...]:
        return scene.val_frames if self.eval_split == "val" else scene.test_frames

    def __getitem__(self, index: int) -> MaskReconstructionBatch:
        scene = self.scenes[index % len(self.scenes)]
        train_pool = scene.train_frames
        eval_pool = self._eval_pool(scene)

        support_offset = index % len(train_pool)
        train_query_offset = (index * 7 + max(1, len(train_pool) // 3)) % len(train_pool)
        eval_query_offset = (index * 11 + 1) % len(eval_pool)

        support_images, support_intrinsics, support_c2w = self._build_bundle(
            scene,
            train_pool,
            self.support_views,
            offset=support_offset,
        )
        train_query_images, train_query_intrinsics, train_query_c2w = self._build_bundle(
            scene,
            train_pool,
            self.train_query_views,
            offset=train_query_offset,
        )
        eval_query_images, eval_query_intrinsics, eval_query_c2w = self._build_bundle(
            scene,
            eval_pool,
            self.eval_query_views,
            offset=eval_query_offset,
        )
        return MaskReconstructionBatch(
            support_images=support_images,
            support_intrinsics=support_intrinsics,
            support_c2w=support_c2w,
            train_query_images=train_query_images,
            train_query_intrinsics=train_query_intrinsics,
            train_query_c2w=train_query_c2w,
            eval_query_images=eval_query_images,
            eval_query_intrinsics=eval_query_intrinsics,
            eval_query_c2w=eval_query_c2w,
        )


def _co3d_intrinsics_from_viewpoint(
    viewpoint: dict,
    *,
    annotation_image_size: Sequence[int],
    output_image_size: tuple[int, int],
) -> torch.Tensor:
    annotation_height = float(annotation_image_size[0])
    annotation_width = float(annotation_image_size[1])
    output_height = float(output_image_size[0])
    output_width = float(output_image_size[1])
    focal_length = viewpoint["focal_length"]
    principal_point = viewpoint["principal_point"]
    intrinsics_format = viewpoint["intrinsics_format"]

    if intrinsics_format == "ndc_isotropic":
        scale = min(annotation_width, annotation_height) / 2.0
        fx = float(focal_length[0]) * scale
        fy = float(focal_length[1]) * scale
        px = (annotation_width / 2.0) - (float(principal_point[0]) * scale)
        py = (annotation_height / 2.0) - (float(principal_point[1]) * scale)
    elif intrinsics_format == "ndc_norm_image_bounds":
        fx = float(focal_length[0]) * (annotation_width / 2.0)
        fy = float(focal_length[1]) * (annotation_height / 2.0)
        px = (annotation_width / 2.0) - (float(principal_point[0]) * (annotation_width / 2.0))
        py = (annotation_height / 2.0) - (float(principal_point[1]) * (annotation_height / 2.0))
    else:
        raise ValueError(f"Unsupported CO3D intrinsics_format: {intrinsics_format}")

    scale_x = output_width / annotation_width
    scale_y = output_height / annotation_height
    return torch.tensor(
        [
            [fx * scale_x, 0.0, px * scale_x],
            [0.0, fy * scale_y, py * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _co3d_c2w_from_viewpoint(viewpoint: dict) -> torch.Tensor:
    rotation_w2c = torch.tensor(viewpoint["R"], dtype=torch.float32)
    translation_w2c = torch.tensor(viewpoint["T"], dtype=torch.float32)
    axis_flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float32))
    rotation_c2w = rotation_w2c.transpose(0, 1) @ axis_flip
    camera_center = -rotation_w2c.transpose(0, 1) @ translation_w2c

    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = rotation_c2w
    c2w[:3, 3] = camera_center
    return c2w


def _load_co3d_frame_annotations(
    category_root: Path,
    *,
    image_size: int,
) -> dict[tuple[str, int], CO3DFrameAnnotation]:
    dataset_root = category_root.parent
    category_name = category_root.name
    annotation_path = category_root / "frame_annotations.jgz"
    with gzip.open(annotation_path, "rt", encoding="utf-8") as handle:
        raw_annotations = json.load(handle)

    annotations: dict[tuple[str, int], CO3DFrameAnnotation] = {}
    for item in raw_annotations:
        image_path = dataset_root / item["image"]["path"]
        if not image_path.exists():
            continue

        mask_info = item.get("mask")
        mask_path = dataset_root / mask_info["path"] if isinstance(mask_info, dict) else None
        annotation = CO3DFrameAnnotation(
            category_name=category_name,
            sequence_name=item["sequence_name"],
            frame_number=int(item["frame_number"]),
            image_path=image_path,
            mask_path=mask_path if mask_path is not None and mask_path.exists() else None,
            intrinsics=_co3d_intrinsics_from_viewpoint(
                item["viewpoint"],
                annotation_image_size=item["image"]["size"],
                output_image_size=(image_size, image_size),
            ),
            c2w=_co3d_c2w_from_viewpoint(item["viewpoint"]),
        )
        annotations[(annotation.sequence_name, annotation.frame_number)] = annotation
    return annotations


def _load_co3d_set_lists(category_root: Path, set_list_name: str) -> dict[str, dict[str, tuple[CO3DFrameAnnotation, ...]]]:
    set_list_path = category_root / "set_lists" / set_list_name
    with open(set_list_path, "r", encoding="utf-8") as handle:
        split_entries = json.load(handle)

    return split_entries


def _iter_co3d_category_roots(root: Path) -> list[Path]:
    if (root / "frame_annotations.jgz").exists():
        return [root]

    category_roots = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "frame_annotations.jgz").exists()
    )
    if not category_roots:
        raise FileNotFoundError(
            "Expected a CO3D category root with frame_annotations.jgz or a CO3D root containing category folders, "
            f"got {root}"
        )
    return category_roots


def _co3d_scene_id(category_name: str, sequence_name: str) -> str:
    return f"{category_name}/{sequence_name}"


def _resolve_requested_co3d_scene_ids(
    requested_names: tuple[str, ...] | None,
    *,
    available_scene_ids: list[str],
) -> tuple[str, ...]:
    if requested_names is None:
        return tuple(available_scene_ids)

    available_set = set(available_scene_ids)
    by_category: dict[str, list[str]] = {}
    by_sequence: dict[str, list[str]] = {}
    for scene_id in available_scene_ids:
        category_name, sequence_name = scene_id.split("/", 1)
        by_category.setdefault(category_name, []).append(scene_id)
        by_sequence.setdefault(sequence_name, []).append(scene_id)

    resolved: list[str] = []
    missing: list[str] = []
    for name in requested_names:
        if name in available_set:
            resolved.append(name)
            continue
        if name in by_category:
            resolved.extend(by_category[name])
            continue
        if name in by_sequence and len(by_sequence[name]) == 1:
            resolved.extend(by_sequence[name])
            continue
        missing.append(name)

    if missing:
        raise ValueError(
            "Unknown or ambiguous CO3D scene requests. Use category names or fully-qualified "
            f"'category/sequence' names. Problematic entries: {missing}"
        )
    return tuple(dict.fromkeys(resolved))


class CO3DMaskReconstructionDataset(Dataset[MaskReconstructionBatch]):
    def __init__(
        self,
        root: str | Path,
        *,
        sequence_names: tuple[str, ...] | None = None,
        image_size: int = 224,
        support_views: int = 16,
        train_query_views: int = 4,
        eval_query_views: int = 4,
        set_list_name: str = "set_lists_manyview_dev_0.json",
        size: int = 128,
        white_background: bool = True,
        cache_images: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")
        self.image_size = image_size
        self.support_views = support_views
        self.train_query_views = train_query_views
        self.eval_query_views = eval_query_views
        self.set_list_name = set_list_name
        self.white_background = white_background
        self.cache_images = cache_images
        self._image_cache: dict[Path, torch.Tensor] = {}
        self.available_categories: tuple[str, ...] = ()
        self.skipped_categories: tuple[str, ...] = ()

        category_roots = _iter_co3d_category_roots(self.root)
        split_map: dict[str, dict[str, list[CO3DFrameAnnotation]]] = {
            "train": {},
            "test": {},
        }
        available_categories: list[str] = []
        skipped_categories: list[str] = []
        for category_root in category_roots:
            set_list_path = category_root / "set_lists" / self.set_list_name
            if not set_list_path.exists():
                skipped_categories.append(category_root.name)
                continue

            annotation_map = _load_co3d_frame_annotations(category_root, image_size=self.image_size)
            raw_split_entries = _load_co3d_set_lists(category_root, self.set_list_name)
            category_name = category_root.name
            available_categories.append(category_name)
            for split_name in ("train", "test"):
                for sequence_name, frame_number, _ in raw_split_entries.get(split_name, []):
                    key = (sequence_name, int(frame_number))
                    annotation = annotation_map.get(key)
                    if annotation is None:
                        continue
                    scene_id = _co3d_scene_id(category_name, sequence_name)
                    split_map[split_name].setdefault(scene_id, []).append(annotation)

        self.available_categories = tuple(sorted(available_categories))
        self.skipped_categories = tuple(sorted(skipped_categories))
        if not self.available_categories:
            raise FileNotFoundError(
                f"No CO3D categories under {self.root} contain set_lists/{self.set_list_name}"
            )

        available_scene_ids = sorted(
            scene_id
            for scene_id in split_map["train"].keys()
            if split_map["train"].get(scene_id) and split_map["test"].get(scene_id)
        )
        self.scenes = _resolve_requested_co3d_scene_ids(
            sequence_names,
            available_scene_ids=available_scene_ids,
        )
        if not self.scenes:
            raise ValueError(f"no usable CO3D scenes found under {self.root}")

        self.train_pools = {
            scene_id: tuple(split_map["train"][scene_id])
            for scene_id in self.scenes
        }
        self.eval_pools = {
            scene_id: tuple(split_map["test"][scene_id])
            for scene_id in self.scenes
        }
        self.size = max(size, len(self.scenes))

    def __len__(self) -> int:
        return self.size

    def _load_image(self, image_path: Path) -> torch.Tensor:
        if self.cache_images and image_path in self._image_cache:
            return self._image_cache[image_path]

        image = _load_rgb_image(
            image_path,
            self.image_size,
            white_background=self.white_background,
        )
        if self.cache_images:
            self._image_cache[image_path] = image
        return image

    def _build_bundle(
        self,
        frame_pool: tuple[CO3DFrameAnnotation, ...],
        view_count: int,
        *,
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_frames = _select_frames(frame_pool, view_count, offset=offset)
        images = torch.stack([self._load_image(frame.image_path) for frame in selected_frames], dim=0)
        intrinsics = torch.stack([frame.intrinsics.clone() for frame in selected_frames], dim=0)
        c2w = torch.stack([frame.c2w.clone() for frame in selected_frames], dim=0)
        return images, intrinsics, c2w

    def __getitem__(self, index: int) -> MaskReconstructionBatch:
        sequence_name = self.scenes[index % len(self.scenes)]
        train_pool = self.train_pools[sequence_name]
        eval_pool = self.eval_pools[sequence_name]

        support_offset = index % len(train_pool)
        train_query_offset = (index * 7 + max(1, len(train_pool) // 3)) % len(train_pool)
        eval_query_offset = (index * 11 + 1) % len(eval_pool)

        support_images, support_intrinsics, support_c2w = self._build_bundle(
            train_pool,
            self.support_views,
            offset=support_offset,
        )
        train_query_images, train_query_intrinsics, train_query_c2w = self._build_bundle(
            train_pool,
            self.train_query_views,
            offset=train_query_offset,
        )
        eval_query_images, eval_query_intrinsics, eval_query_c2w = self._build_bundle(
            eval_pool,
            self.eval_query_views,
            offset=eval_query_offset,
        )
        return MaskReconstructionBatch(
            support_images=support_images,
            support_intrinsics=support_intrinsics,
            support_c2w=support_c2w,
            train_query_images=train_query_images,
            train_query_intrinsics=train_query_intrinsics,
            train_query_c2w=train_query_c2w,
            eval_query_images=eval_query_images,
            eval_query_intrinsics=eval_query_intrinsics,
            eval_query_c2w=eval_query_c2w,
        )


def mask_batch_collate(samples: list[MaskReconstructionBatch]) -> MaskReconstructionBatch:
    return MaskReconstructionBatch(
        support_images=torch.stack([sample.support_images for sample in samples], dim=0),
        support_intrinsics=torch.stack([sample.support_intrinsics for sample in samples], dim=0),
        support_c2w=torch.stack([sample.support_c2w for sample in samples], dim=0),
        train_query_images=torch.stack([sample.train_query_images for sample in samples], dim=0),
        train_query_intrinsics=torch.stack(
            [sample.train_query_intrinsics for sample in samples],
            dim=0,
        ),
        train_query_c2w=torch.stack([sample.train_query_c2w for sample in samples], dim=0),
        eval_query_images=torch.stack([sample.eval_query_images for sample in samples], dim=0),
        eval_query_intrinsics=torch.stack(
            [sample.eval_query_intrinsics for sample in samples],
            dim=0,
        ),
        eval_query_c2w=torch.stack([sample.eval_query_c2w for sample in samples], dim=0),
    )


def move_mask_batch(batch: MaskReconstructionBatch, device: str) -> MaskReconstructionBatch:
    return MaskReconstructionBatch(
        support_images=batch.support_images.to(device),
        support_intrinsics=batch.support_intrinsics.to(device),
        support_c2w=batch.support_c2w.to(device),
        train_query_images=batch.train_query_images.to(device),
        train_query_intrinsics=batch.train_query_intrinsics.to(device),
        train_query_c2w=batch.train_query_c2w.to(device),
        eval_query_images=batch.eval_query_images.to(device),
        eval_query_intrinsics=batch.eval_query_intrinsics.to(device),
        eval_query_c2w=batch.eval_query_c2w.to(device),
    )


def sample_random_patch_mask(
    *,
    images: torch.Tensor,
    patch_size: int,
    mask_ratio: float,
    seed: int,
    background_threshold: float = 0.01,
) -> torch.Tensor:
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in the range (0, 1]")
    if not 0.0 <= background_threshold < 0.5:
        raise ValueError("background_threshold must be in the range [0, 0.5)")
    if images.dim() != 5:
        raise ValueError("images must have shape (B, V, C, H, W)")

    batch_size, num_views, channels, height, width = images.shape
    total_patches = num_patches((height, width), patch_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mask = torch.zeros(batch_size, num_views, total_patches, dtype=torch.bool)

    flat_images = images.reshape(batch_size * num_views, channels, height, width).detach().cpu()
    image_patches = patchify(flat_images, patch_size).reshape(
        batch_size,
        num_views,
        total_patches,
        channels,
        patch_size * patch_size,
    )
    white_background = (image_patches >= (1.0 - background_threshold)).all(dim=-1).all(dim=-1)
    black_background = (image_patches <= background_threshold).all(dim=-1).all(dim=-1)
    eligible = ~(white_background | black_background)

    for batch_idx in range(batch_size):
        for view_idx in range(num_views):
            eligible_indices = torch.nonzero(eligible[batch_idx, view_idx], as_tuple=False).flatten()
            if eligible_indices.numel() == 0:
                continue
            masked_patch_count = min(
                int(eligible_indices.numel()),
                max(1, int(round(float(eligible_indices.numel()) * float(mask_ratio)))),
            )
            perm = torch.randperm(int(eligible_indices.numel()), generator=generator)
            chosen = eligible_indices[perm[:masked_patch_count]]
            mask[batch_idx, view_idx, chosen] = True
    return mask


def apply_patch_mask(
    images: torch.Tensor,
    patch_mask: torch.Tensor,
    *,
    patch_size: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    if images.dim() != 5:
        raise ValueError("images must have shape (B, V, C, H, W)")

    batch_size, num_views, channels, height, width = images.shape
    flat_images = images.reshape(batch_size * num_views, channels, height, width)
    image_patches = patchify(flat_images, patch_size)
    flat_mask = patch_mask.reshape(batch_size * num_views, -1).to(image_patches.device)
    if image_patches.shape[1] != flat_mask.shape[1]:
        raise ValueError(
            "patch_mask patch count must match image patch count, "
            f"got mask={flat_mask.shape[1]} image={image_patches.shape[1]}"
        )

    masked_patches = image_patches.clone()
    masked_patches = masked_patches.masked_fill(flat_mask.unsqueeze(-1), fill_value)
    masked_images = unpatchify(
        masked_patches,
        patch_size=patch_size,
        image_size=(height, width),
        channels=channels,
    )
    return masked_images.reshape(batch_size, num_views, channels, height, width)


def patch_mask_to_image(
    patch_mask: torch.Tensor,
    *,
    patch_size: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    batch_size, num_views, total_patches = patch_mask.shape
    patch_pixels = patch_size * patch_size
    mask_patches = patch_mask.reshape(batch_size * num_views, total_patches, 1).float()
    mask_patches = mask_patches.expand(-1, -1, patch_pixels)
    mask_image = unpatchify(
        mask_patches,
        patch_size=patch_size,
        image_size=image_size,
        channels=1,
    )
    return mask_image.reshape(batch_size, num_views, 1, *image_size)


def compute_masked_patch_metrics(
    pred_patch_rgb: torch.Tensor,
    target_patch_rgb: torch.Tensor,
    patch_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    expanded_mask = patch_mask.unsqueeze(-1).expand_as(pred_patch_rgb)
    if not bool(expanded_mask.any()):
        raise ValueError("patch_mask must select at least one patch value")

    diff = pred_patch_rgb - target_patch_rgb
    masked_squared = diff.square()[expanded_mask]
    masked_abs = diff.abs()[expanded_mask]
    mse = masked_squared.mean()
    mae = masked_abs.mean()
    return {
        "mse": mse,
        "mae": mae,
        "masked_values": pred_patch_rgb.new_tensor(float(masked_squared.numel())),
    }


class DA3MaskedPatchModel(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 14,
        backbone: nn.Module | None = None,
        backbone_model_name: str = "da3-large",
        backbone_weights_path: str | None = None,
        backbone_trainable: bool = False,
        head_type: str = "mlp",
        head_hidden_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        decoder_hidden_dim: int | None = 128,
        mask_fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.head_type = head_type
        self.head_hidden_dim = head_hidden_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.decoder_hidden_dim = decoder_hidden_dim
        self.mask_fill_value = mask_fill_value
        if self.head_type not in {"mlp", "cnn", "dpt"}:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

        self.backbone = backbone or DA3PatchBackbone(
            model_name=backbone_model_name,
            weights_path=backbone_weights_path,
            trainable=backbone_trainable,
            return_all_features=self.head_type == "dpt",
        )
        backbone_patch_size = getattr(self.backbone, "patch_size", None)
        if backbone_patch_size is not None and backbone_patch_size != self.patch_size:
            raise ValueError(
                f"Backbone patch_size={backbone_patch_size} does not match model patch_size={self.patch_size}"
            )

        self.patch_head: nn.Module | None = None
        self.cnn_head: LightweightCNNDecoder | None = None
        self.dpt_head: CrossAttentionNVSHead | None = None

    @staticmethod
    def _normalize_backbone_outputs(
        backbone_outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        if isinstance(backbone_outputs, torch.Tensor):
            return [backbone_outputs]
        if isinstance(backbone_outputs, (list, tuple)):
            if not backbone_outputs:
                raise RuntimeError("Backbone returned an empty feature list")
            return list(backbone_outputs)
        raise TypeError(f"Unsupported backbone output type: {type(backbone_outputs)!r}")

    def _build_head(self, embed_dim: int, device: torch.device) -> None:
        if self.patch_head is not None:
            return

        patch_dim = 3 * self.patch_size * self.patch_size
        self.patch_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.head_hidden_dim, patch_dim),
        ).to(device)

    def _build_dpt_head(self, embed_dim: int, device: torch.device) -> None:
        if self.dpt_head is not None:
            return

        self.dpt_head = CrossAttentionNVSHead(
            embed_dim=embed_dim,
            patch_size=self.patch_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            decoder_hidden_dim=self.decoder_hidden_dim,
            out_channels=3,
        ).to(device)

    def _build_cnn_head(self, embed_dim: int, device: torch.device) -> None:
        if self.cnn_head is not None:
            return

        hidden_dim = self.decoder_hidden_dim or min(128, max(embed_dim // 2, 32))
        self.cnn_head = LightweightCNNDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_channels=3,
        ).to(device)

    def forward(
        self,
        *,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        query_patch_mask: torch.Tensor,
    ) -> MaskedPatchOutputs:
        if support_images.dim() != 5 or query_images.dim() != 5:
            raise ValueError("support_images and query_images must have shape (B, V, C, H, W)")
        if support_images.shape[0] != query_images.shape[0]:
            raise ValueError("support_images and query_images must share the batch dimension")
        if support_images.shape[-2:] != query_images.shape[-2:]:
            raise ValueError("support_images and query_images must share the image resolution")

        batch_size, query_views, channels, height, width = query_images.shape
        expected_patches = num_patches((height, width), self.patch_size)
        if query_patch_mask.shape != (batch_size, query_views, expected_patches):
            raise ValueError(
                "query_patch_mask must have shape "
                f"{(batch_size, query_views, expected_patches)}, got {tuple(query_patch_mask.shape)}"
            )

        device = query_images.device
        query_patch_mask = query_patch_mask.to(device)
        masked_query_images = apply_patch_mask(
            query_images,
            query_patch_mask,
            patch_size=self.patch_size,
            fill_value=self.mask_fill_value,
        )
        encoder_images = torch.cat([support_images, masked_query_images], dim=1)
        backbone_outputs = self.backbone(encoder_images)
        stage_tokens = self._normalize_backbone_outputs(backbone_outputs)
        support_views = support_images.shape[1]
        support_stage_tokens = [tokens[:, :support_views, :, :] for tokens in stage_tokens]
        query_stage_tokens = [tokens[:, support_views:, :, :] for tokens in stage_tokens]
        all_tokens = query_stage_tokens[-1]
        if all_tokens.dim() != 4:
            raise RuntimeError(f"Expected backbone tokens with shape (B, V, P, D), got {all_tokens.shape}")

        query_tokens = all_tokens
        if query_tokens.shape[:3] != (batch_size, query_views, expected_patches):
            raise RuntimeError(
                "Query token layout does not match expected masked query layout: "
                f"expected {(batch_size, query_views, expected_patches)}, got {tuple(query_tokens.shape[:3])}"
            )

        embed_dim = query_tokens.shape[-1]
        flat_query_images = query_images.reshape(batch_size * query_views, channels, height, width)
        target_patch_rgb = patchify(flat_query_images, self.patch_size).reshape(
            batch_size,
            query_views,
            expected_patches,
            -1,
        )
        if self.head_type == "mlp":
            self._build_head(embed_dim, device)
            assert self.patch_head is not None
            pred_patch_rgb = self.patch_head(query_tokens)
            rendered_tokens = query_tokens
            pred_query_images = unpatchify(
                pred_patch_rgb.reshape(batch_size * query_views, expected_patches, -1),
                patch_size=self.patch_size,
                image_size=(height, width),
                channels=channels,
            ).reshape(batch_size, query_views, channels, height, width)
        elif self.head_type == "cnn":
            self._build_cnn_head(embed_dim, device)
            assert self.cnn_head is not None
            rendered_tokens = query_tokens
            pred_query_images = self.cnn_head(
                query_tokens,
                image_size=(height, width),
                patch_size=self.patch_size,
            )
            pred_patch_rgb = patchify(
                pred_query_images.reshape(batch_size * query_views, channels, height, width),
                self.patch_size,
            ).reshape(batch_size, query_views, expected_patches, -1)
        else:
            self._build_dpt_head(embed_dim, device)
            assert self.dpt_head is not None
            support_stage_memory = [
                stage.reshape(batch_size, -1, embed_dim)
                for stage in support_stage_tokens
            ]
            pred_query_images, rendered_tokens = self.dpt_head(
                query_tokens,
                support_stage_memory,
                image_size=(height, width),
            )
            pred_patch_rgb = patchify(
                pred_query_images.reshape(batch_size * query_views, channels, height, width),
                self.patch_size,
            ).reshape(batch_size, query_views, expected_patches, -1)

        reconstructed_patch_rgb = torch.where(
            query_patch_mask.unsqueeze(-1),
            pred_patch_rgb,
            target_patch_rgb,
        )
        reconstructed_query_images = unpatchify(
            reconstructed_patch_rgb.reshape(batch_size * query_views, expected_patches, -1),
            patch_size=self.patch_size,
            image_size=(height, width),
            channels=channels,
        ).reshape(batch_size, query_views, channels, height, width)

        return MaskedPatchOutputs(
            pred_patch_rgb=pred_patch_rgb,
            target_patch_rgb=target_patch_rgb,
            query_patch_mask=query_patch_mask,
            masked_query_images=masked_query_images,
            pred_query_images=pred_query_images,
            reconstructed_query_images=reconstructed_query_images,
            query_tokens=rendered_tokens,
        )
