from __future__ import annotations

import gzip
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from da3_nvs.config import POC_NERF_SYNTHETIC_SCENES
from da3_nvs.data import raymap_from_cameras
from da3_nvs.data.nerf_synthetic import (
    BlenderFrame,
    BlenderScene,
    _camera_intrinsics,
    _load_rgb_image,
    _load_scene,
    _select_frames,
)
from da3_nvs.models import CrossAttentionNVSHead, DA3PatchBackbone, RayMapEncoder
from da3_nvs.models.common import num_patches, patchify, unpatchify


def debug_log(message: str) -> None:
    print(message, flush=True)


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


def _try_import_co3d_dataset_class():
    try:
        from co3d.dataset.co3d_dataset import Co3dDataset
    except ImportError:
        return None
    return Co3dDataset


class OfficialCo3dImageBackend:
    def __init__(
        self,
        *,
        category_root: Path,
        image_size: int,
        white_background: bool,
    ) -> None:
        co3d_dataset_cls = _try_import_co3d_dataset_class()
        if co3d_dataset_cls is None:
            raise RuntimeError("co3d.dataset.co3d_dataset.Co3dDataset is not importable")

        dataset_root = category_root.parent
        signature = inspect.signature(co3d_dataset_cls)
        candidate_kwargs = {
            "dataset_root": str(dataset_root),
            "frame_annotations_file": str(category_root / "frame_annotations.jgz"),
            "sequence_annotations_file": str(category_root / "sequence_annotations.jgz"),
            "subset_lists_file": None,
            "subsets": None,
            "subset": None,
            "path_manager": None,
            "pick_sequence": None,
            "image_height": image_size,
            "image_width": image_size,
            "box_crop": False,
            "box_crop_context": 0.0,
            "remove_empty_masks": False,
            "load_images": True,
            "load_masks": False,
            "load_depths": False,
            "load_depth_masks": False,
            "load_point_clouds": False,
            "mask_images": False,
        }
        init_kwargs = {
            name: candidate_kwargs[name]
            for name in signature.parameters.keys()
            if name != "self" and name in candidate_kwargs
        }
        self.dataset = co3d_dataset_cls(**init_kwargs)
        self.image_size = image_size
        self.white_background = white_background
        self.index_by_key = self._build_index()

    @staticmethod
    def _extract_attr(value, names: Sequence[str]):
        for name in names:
            if isinstance(value, dict) and name in value:
                return value[name]
            if hasattr(value, name):
                return getattr(value, name)
        return None

    def _build_index(self) -> dict[tuple[str, int], int]:
        frame_annots = getattr(self.dataset, "frame_annots", None)
        if frame_annots is None:
            frame_annots = getattr(self.dataset, "frame_annotations", None)
        if frame_annots is None:
            raise RuntimeError("Official Co3dDataset does not expose frame annotations")

        index_by_key: dict[tuple[str, int], int] = {}
        for dataset_index, annotation in enumerate(frame_annots):
            sequence_name = self._extract_attr(annotation, ("sequence_name",))
            frame_number = self._extract_attr(annotation, ("frame_number",))
            if sequence_name is None:
                frame_annotation = self._extract_attr(annotation, ("frame_annotation",))
                if frame_annotation is not None:
                    sequence_name = self._extract_attr(frame_annotation, ("sequence_name",))
                    frame_number = self._extract_attr(frame_annotation, ("frame_number",))
            if sequence_name is None or frame_number is None:
                continue
            index_by_key[(str(sequence_name), int(frame_number))] = dataset_index
        if not index_by_key:
            raise RuntimeError("Official Co3dDataset frame annotation index is empty")
        return index_by_key

    def load_image(self, sequence_name: str, frame_number: int) -> torch.Tensor:
        dataset_index = self.index_by_key.get((sequence_name, int(frame_number)))
        if dataset_index is None:
            raise KeyError(f"Official Co3dDataset is missing frame {(sequence_name, frame_number)}")

        frame_data = self.dataset[dataset_index]
        image = getattr(frame_data, "image_rgb", None)
        if image is None:
            image = getattr(frame_data, "image", None)
        if image is None:
            raise RuntimeError("Official Co3dDataset frame does not expose image_rgb/image")

        if image.dim() == 4:
            image = image[0]
        if image.dim() != 3:
            raise RuntimeError(f"Unexpected official CO3D image shape: {tuple(image.shape)}")
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        fg_probability = getattr(frame_data, "fg_probability", None)
        if self.white_background and fg_probability is not None:
            if fg_probability.dim() == 4:
                fg_probability = fg_probability[0]
            if fg_probability.dim() == 3 and fg_probability.shape[0] == 1:
                image = image * fg_probability + (1.0 - fg_probability)

        if tuple(image.shape[-2:]) != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return image.contiguous()


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


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightCNNDecoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        hidden_dim: int,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        bottleneck_dim = max(hidden_dim * 2, 32)
        refine_dim = max(hidden_dim // 2, 16)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.encoder = ConvBlock(hidden_dim, hidden_dim)
        self.downsample = nn.Sequential(
            nn.Conv2d(hidden_dim, bottleneck_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.bottleneck = ConvBlock(bottleneck_dim, bottleneck_dim)
        self.up_project = nn.Sequential(
            nn.Conv2d(bottleneck_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.decoder = ConvBlock(hidden_dim * 2, hidden_dim)
        self.refine = ConvBlock(hidden_dim, refine_dim)
        self.head = nn.Sequential(
            nn.Conv2d(refine_dim, refine_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(refine_dim, out_channels, kernel_size=1, stride=1, padding=0),
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
        skip = self.encoder(self.proj(feature_map))
        down = self.downsample(skip)
        bottleneck = self.bottleneck(down)
        up = F.interpolate(
            bottleneck,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        up = self.up_project(up)
        feature_map = self.decoder(torch.cat([skip, up], dim=1))
        feature_map = F.interpolate(
            feature_map,
            size=(max(image_size[0] // 2, 1), max(image_size[1] // 2, 1)),
            mode="bilinear",
            align_corners=True,
        )
        feature_map = self.refine(feature_map)
        feature_map = F.interpolate(feature_map, size=image_size, mode="bilinear", align_corners=True)
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


def _load_json_file(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _list_co3d_set_list_paths(category_root: Path) -> list[Path]:
    set_lists_dir = category_root / "set_lists"
    if not set_lists_dir.exists():
        return []
    return sorted(set_lists_dir.glob("*.json"))


def _split_co3d_set_list_paths(set_list_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    manyview_paths = [path for path in set_list_paths if "manyview" in path.name]
    fewview_paths = [path for path in set_list_paths if "fewview" in path.name]
    return manyview_paths, fewview_paths


def _entries_to_annotations_by_sequence(
    entries: list,
    annotation_map: dict[tuple[str, int], CO3DFrameAnnotation],
) -> dict[str, list[CO3DFrameAnnotation]]:
    grouped: dict[str, list[CO3DFrameAnnotation]] = {}
    for sequence_name, frame_number, _ in entries:
        annotation = annotation_map.get((sequence_name, int(frame_number)))
        if annotation is None:
            continue
        grouped.setdefault(sequence_name, []).append(annotation)
    return grouped


def _merge_unique_annotations(
    annotations: list[CO3DFrameAnnotation],
) -> tuple[CO3DFrameAnnotation, ...]:
    unique: dict[int, CO3DFrameAnnotation] = {}
    for annotation in annotations:
        unique[int(annotation.frame_number)] = annotation
    return tuple(unique[key] for key in sorted(unique))


def _random_split_annotations(
    annotations: tuple[CO3DFrameAnnotation, ...],
    *,
    eval_ratio: float,
    seed_key: str,
) -> tuple[tuple[CO3DFrameAnnotation, ...], tuple[CO3DFrameAnnotation, ...]]:
    if len(annotations) < 2:
        return annotations, annotations

    annotations_list = list(annotations)
    rng = random.Random(seed_key)
    rng.shuffle(annotations_list)
    eval_count = max(1, int(round(len(annotations_list) * eval_ratio)))
    eval_count = min(eval_count, len(annotations_list) - 1)
    eval_split = tuple(sorted(annotations_list[:eval_count], key=lambda item: item.frame_number))
    train_split = tuple(sorted(annotations_list[eval_count:], key=lambda item: item.frame_number))
    return train_split, eval_split


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
        fallback_eval_ratio: float = 0.1,
        fallback_split_seed: int = 0,
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
        self.fallback_eval_ratio = fallback_eval_ratio
        self.fallback_split_seed = fallback_split_seed
        self.white_background = white_background
        self.cache_images = cache_images
        self._image_cache: dict[Path, torch.Tensor] = {}
        self.official_image_backends: dict[str, OfficialCo3dImageBackend] = {}
        self.available_categories: tuple[str, ...] = ()
        self.skipped_categories: tuple[str, ...] = ()
        self.category_set_lists: dict[str, tuple[str, ...]] = {}
        self.category_debug_stats: dict[str, dict[str, int]] = {}

        category_roots = _iter_co3d_category_roots(self.root)
        debug_log(f"[co3d] scanning categories under {self.root} ({len(category_roots)} candidates)")
        scene_train_pools: dict[str, tuple[CO3DFrameAnnotation, ...]] = {}
        scene_eval_pools: dict[str, tuple[CO3DFrameAnnotation, ...]] = {}
        available_categories: list[str] = []
        skipped_categories: list[str] = []
        category_set_lists: dict[str, tuple[str, ...]] = {}
        category_debug_stats: dict[str, dict[str, int]] = {}
        for category_root in category_roots:
            set_list_paths = _list_co3d_set_list_paths(category_root)
            if not set_list_paths:
                skipped_categories.append(category_root.name)
                continue

            debug_log(f"[co3d] loading category {category_root.name}")
            annotation_map = _load_co3d_frame_annotations(category_root, image_size=self.image_size)
            category_name = category_root.name
            available_categories.append(category_name)
            category_set_lists[category_name] = tuple(path.name for path in set_list_paths)
            category_stats = {
                "annotations_with_existing_images": len(annotation_map),
                "manyview_files": 0,
                "fewview_files": 0,
                "manyview_sequences_total": 0,
                "manyview_sequences_usable": 0,
                "fewview_sequences_total": 0,
                "fewview_sequences_usable": 0,
            }
            try:
                self.official_image_backends[category_name] = OfficialCo3dImageBackend(
                    category_root=category_root,
                    image_size=self.image_size,
                    white_background=self.white_background,
                )
                debug_log(f"[co3d] category {category_name} using official Co3dDataset image backend")
            except Exception as error:
                debug_log(
                    f"[co3d] category {category_name} official Co3dDataset unavailable, "
                    f"falling back to direct image loading: {error}"
                )

            manyview_paths, fewview_paths = _split_co3d_set_list_paths(set_list_paths)
            category_stats["manyview_files"] = len(manyview_paths)
            category_stats["fewview_files"] = len(fewview_paths)
            manyview_scene_ids: set[str] = set()
            manyview_sequences_seen: set[str] = set()
            for path in manyview_paths:
                raw_split_entries = _load_json_file(path)
                if not isinstance(raw_split_entries, dict):
                    continue
                train_by_sequence = _entries_to_annotations_by_sequence(
                    list(raw_split_entries.get("train", [])),
                    annotation_map,
                )
                eval_entries = list(raw_split_entries.get("val", [])) + list(raw_split_entries.get("test", []))
                eval_by_sequence = _entries_to_annotations_by_sequence(
                    eval_entries,
                    annotation_map,
                )
                sequence_names = sorted(set(train_by_sequence) | set(eval_by_sequence))
                manyview_sequences_seen.update(sequence_names)
                for sequence_name in sequence_names:
                    scene_id = _co3d_scene_id(category_name, sequence_name)
                    train_annotations = _merge_unique_annotations(train_by_sequence.get(sequence_name, []))
                    eval_annotations = _merge_unique_annotations(eval_by_sequence.get(sequence_name, []))
                    if len(train_annotations) == 0 and len(eval_annotations) > 1:
                        train_annotations, eval_annotations = _random_split_annotations(
                            eval_annotations,
                            eval_ratio=self.fallback_eval_ratio,
                            seed_key=f"{scene_id}:{self.fallback_split_seed}:manyview-eval-only",
                        )
                    elif len(eval_annotations) == 0 and len(train_annotations) > 1:
                        train_annotations, eval_annotations = _random_split_annotations(
                            train_annotations,
                            eval_ratio=self.fallback_eval_ratio,
                            seed_key=f"{scene_id}:{self.fallback_split_seed}:manyview-train-only",
                        )
                    if len(train_annotations) == 0 or len(eval_annotations) == 0:
                        continue
                    scene_train_pools[scene_id] = train_annotations
                    scene_eval_pools[scene_id] = eval_annotations
                    manyview_scene_ids.add(scene_id)
            category_stats["manyview_sequences_total"] = len(manyview_sequences_seen)
            category_stats["manyview_sequences_usable"] = len(manyview_scene_ids)

            if fewview_paths:
                fewview_annotations_by_sequence: dict[str, list[CO3DFrameAnnotation]] = {}
                for path in fewview_paths:
                    raw_split_entries = _load_json_file(path)
                    if not isinstance(raw_split_entries, dict):
                        continue
                    for split_entries in raw_split_entries.values():
                        grouped = _entries_to_annotations_by_sequence(list(split_entries), annotation_map)
                        for sequence_name, annotations in grouped.items():
                            fewview_annotations_by_sequence.setdefault(sequence_name, []).extend(annotations)

                category_stats["fewview_sequences_total"] = len(fewview_annotations_by_sequence)
                for sequence_name, annotations in fewview_annotations_by_sequence.items():
                    scene_id = _co3d_scene_id(category_name, sequence_name)
                    if scene_id in manyview_scene_ids:
                        continue
                    merged_annotations = _merge_unique_annotations(annotations)
                    if len(merged_annotations) < 2:
                        continue
                    train_annotations, eval_annotations = _random_split_annotations(
                        merged_annotations,
                        eval_ratio=self.fallback_eval_ratio,
                        seed_key=f"{scene_id}:{self.fallback_split_seed}:fewview-fallback",
                    )
                    if len(train_annotations) == 0 or len(eval_annotations) == 0:
                        continue
                    scene_train_pools[scene_id] = train_annotations
                    scene_eval_pools[scene_id] = eval_annotations
                    category_stats["fewview_sequences_usable"] += 1
            category_debug_stats[category_name] = category_stats
            debug_log(
                "[co3d] category stats | "
                f"{category_name} annotations={category_stats['annotations_with_existing_images']} "
                f"manyview_files={category_stats['manyview_files']} "
                f"manyview_sequences={category_stats['manyview_sequences_total']} "
                f"manyview_usable={category_stats['manyview_sequences_usable']} "
                f"fewview_files={category_stats['fewview_files']} "
                f"fewview_sequences={category_stats['fewview_sequences_total']} "
                f"fewview_usable={category_stats['fewview_sequences_usable']}"
            )

        self.available_categories = tuple(sorted(available_categories))
        self.skipped_categories = tuple(sorted(skipped_categories))
        self.category_set_lists = dict(sorted(category_set_lists.items()))
        self.category_debug_stats = dict(sorted(category_debug_stats.items()))
        if not self.available_categories:
            raise FileNotFoundError(
                f"No CO3D categories under {self.root} contain usable set_lists json files"
            )
        debug_log(
            f"[co3d] categories ready | usable={len(self.available_categories)} "
            f"skipped_missing_set_list={len(self.skipped_categories)}"
        )

        available_scene_ids = sorted(
            scene_id
            for scene_id in scene_train_pools.keys()
            if scene_train_pools.get(scene_id) and scene_eval_pools.get(scene_id)
        )
        self.scenes = _resolve_requested_co3d_scene_ids(
            sequence_names,
            available_scene_ids=available_scene_ids,
        )
        if not self.scenes:
            debug_preview = json.dumps(self.category_debug_stats, indent=2)
            raise ValueError(
                f"no usable CO3D scenes found under {self.root}\n"
                f"category_debug_stats={debug_preview}"
            )

        self.train_pools = {
            scene_id: tuple(scene_train_pools[scene_id])
            for scene_id in self.scenes
        }
        self.eval_pools = {
            scene_id: tuple(scene_eval_pools[scene_id])
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

    def _load_co3d_image(self, annotation: CO3DFrameAnnotation) -> torch.Tensor:
        backend = self.official_image_backends.get(annotation.category_name)
        if backend is not None:
            try:
                return backend.load_image(annotation.sequence_name, annotation.frame_number)
            except Exception as error:
                debug_log(
                    f"[co3d] official image backend miss for "
                    f"{annotation.category_name}/{annotation.sequence_name}/{annotation.frame_number}: {error}"
                )
        return self._load_image(annotation.image_path)

    def _build_bundle(
        self,
        frame_pool: tuple[CO3DFrameAnnotation, ...],
        view_count: int,
        *,
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_frames = _select_frames(frame_pool, view_count, offset=offset)
        images = torch.stack([self._load_co3d_image(frame) for frame in selected_frames], dim=0)
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
        expanded_mask = torch.ones_like(pred_patch_rgb, dtype=torch.bool)

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
        include_moment: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.head_type = head_type
        self.head_hidden_dim = head_hidden_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.decoder_hidden_dim = decoder_hidden_dim
        self.mask_fill_value = mask_fill_value
        self.include_moment = include_moment
        self.ray_channels = 9 if include_moment else 6
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
        self.query_ray_encoder: RayMapEncoder | None = None

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

    def _build_query_ray_encoder(self, embed_dim: int, device: torch.device) -> None:
        if self.query_ray_encoder is not None:
            return

        self.query_ray_encoder = RayMapEncoder(
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            ray_channels=self.ray_channels,
        ).to(device)

    def _infer_backbone_embed_dim(self) -> int:
        embed_dim = getattr(self.backbone, "embed_dim", None)
        if isinstance(embed_dim, int):
            return embed_dim
        proj = getattr(self.backbone, "proj", None)
        if isinstance(proj, nn.Linear):
            return proj.out_features
        if isinstance(self.backbone, DA3PatchBackbone):
            return self.backbone.get_embed_dim()
        raise RuntimeError("Could not infer backbone embedding dimension for query ray embeddings")

    def _build_masked_query_ray_bias(
        self,
        *,
        query_intrinsics: torch.Tensor | None,
        query_c2w: torch.Tensor | None,
        query_patch_mask: torch.Tensor,
        image_size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        if query_intrinsics is None or query_c2w is None:
            return None

        batch_size, query_views = query_intrinsics.shape[:2]
        raymaps = raymap_from_cameras(
            query_intrinsics.reshape(batch_size * query_views, 3, 3),
            query_c2w.reshape(batch_size * query_views, *query_c2w.shape[-2:]),
            image_size[0],
            image_size[1],
            include_moment=self.include_moment,
        )
        embed_dim = self._infer_backbone_embed_dim()
        self._build_query_ray_encoder(embed_dim, device)
        assert self.query_ray_encoder is not None
        ray_tokens = self.query_ray_encoder(raymaps.to(device)).reshape(
            batch_size,
            query_views,
            -1,
            embed_dim,
        )
        if ray_tokens.shape[:3] != query_patch_mask.shape:
            raise RuntimeError(
                "Query ray token layout does not match masked query patch layout: "
                f"{tuple(ray_tokens.shape[:3])} vs {tuple(query_patch_mask.shape)}"
            )
        return ray_tokens * query_patch_mask.unsqueeze(-1).to(dtype=ray_tokens.dtype)

    def forward(
        self,
        *,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        query_patch_mask: torch.Tensor,
        query_intrinsics: torch.Tensor | None = None,
        query_c2w: torch.Tensor | None = None,
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
        if query_intrinsics is not None:
            query_intrinsics = query_intrinsics.to(device)
        if query_c2w is not None:
            query_c2w = query_c2w.to(device)
        masked_query_images = apply_patch_mask(
            query_images,
            query_patch_mask,
            patch_size=self.patch_size,
            fill_value=self.mask_fill_value,
        )
        encoder_images = torch.cat([support_images, masked_query_images], dim=1)
        query_ray_bias = self._build_masked_query_ray_bias(
            query_intrinsics=query_intrinsics,
            query_c2w=query_c2w,
            query_patch_mask=query_patch_mask,
            image_size=(height, width),
            device=device,
        )
        backbone_kwargs: dict[str, torch.Tensor] = {}
        if query_ray_bias is not None and isinstance(self.backbone, DA3PatchBackbone):
            support_ray_bias = torch.zeros(
                batch_size,
                support_images.shape[1],
                expected_patches,
                query_ray_bias.shape[-1],
                device=device,
                dtype=query_ray_bias.dtype,
            )
            backbone_kwargs["patch_token_bias"] = torch.cat([support_ray_bias, query_ray_bias], dim=1)
        backbone_outputs = self.backbone(encoder_images, **backbone_kwargs)
        stage_tokens = self._normalize_backbone_outputs(backbone_outputs)
        support_views = support_images.shape[1]
        support_stage_tokens = [tokens[:, :support_views, :, :] for tokens in stage_tokens]
        query_stage_tokens = [tokens[:, support_views:, :, :] for tokens in stage_tokens]
        if query_ray_bias is not None and not isinstance(self.backbone, DA3PatchBackbone):
            query_stage_tokens = [stage + query_ray_bias for stage in query_stage_tokens]
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
