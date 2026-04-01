from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from da3_nvs.config import POC_NERF_SYNTHETIC_SCENES
from da3_nvs.data.types import SceneBatch


def _default_root() -> Path:
    return Path(__file__).resolve().parents[4] / "datasets" / "nerf_synthetic"


@dataclass(frozen=True)
class BlenderFrame:
    image_path: Path
    c2w: torch.Tensor


@dataclass(frozen=True)
class BlenderScene:
    name: str
    camera_angle_x: float
    original_size: tuple[int, int]
    train_frames: tuple[BlenderFrame, ...]
    val_frames: tuple[BlenderFrame, ...]
    test_frames: tuple[BlenderFrame, ...]


def _resolve_frame_path(scene_dir: Path, file_path: str) -> Path:
    path = scene_dir / file_path
    if path.suffix == "":
        path = path.with_suffix(".png")
    return path


def _load_split(scene_dir: Path, split: str) -> tuple[float, tuple[BlenderFrame, ...]]:
    metadata = json.loads((scene_dir / f"transforms_{split}.json").read_text())
    frames = []
    for frame in metadata["frames"]:
        frames.append(
            BlenderFrame(
                image_path=_resolve_frame_path(scene_dir, frame["file_path"]),
                c2w=torch.tensor(frame["transform_matrix"], dtype=torch.float32),
            )
        )
    return float(metadata["camera_angle_x"]), tuple(frames)


def _load_scene(scene_dir: Path) -> BlenderScene:
    camera_angle_x, train_frames = _load_split(scene_dir, "train")
    _, val_frames = _load_split(scene_dir, "val")
    _, test_frames = _load_split(scene_dir, "test")
    first_image = Image.open(train_frames[0].image_path)
    width, height = first_image.size
    first_image.close()
    return BlenderScene(
        name=scene_dir.name,
        camera_angle_x=camera_angle_x,
        original_size=(height, width),
        train_frames=train_frames,
        val_frames=val_frames,
        test_frames=test_frames,
    )


def _resample_mode() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _pil_to_rgba_tensor(image: Image.Image) -> torch.Tensor:
    rgba = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    rgba = rgba.view(image.height, image.width, 4).float() / 255.0
    return rgba


def _load_rgb_image(
    image_path: Path,
    image_size: int,
    *,
    white_background: bool,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGBA")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), resample=_resample_mode())

    rgba = _pil_to_rgba_tensor(image)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3:4]
    if white_background:
        rgb = rgb * alpha + (1.0 - alpha)
    else:
        rgb = rgb * alpha
    return rgb.permute(2, 0, 1).contiguous()


def _camera_intrinsics(
    camera_angle_x: float,
    original_size: tuple[int, int],
    image_size: int,
) -> torch.Tensor:
    original_height, original_width = original_size
    focal = 0.5 * float(original_width) / math.tan(0.5 * camera_angle_x)
    scale_x = float(image_size) / float(original_width)
    scale_y = float(image_size) / float(original_height)
    return torch.tensor(
        [
            [focal * scale_x, 0.0, image_size / 2.0],
            [0.0, focal * scale_y, image_size / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _select_frames(
    frames: tuple[BlenderFrame, ...],
    count: int,
    *,
    offset: int,
) -> list[BlenderFrame]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not frames:
        raise ValueError("frame list is empty")

    total = len(frames)
    return [frames[int((offset + (step * total / count)) % total)] for step in range(count)]


class NerfSyntheticSceneDataset(Dataset[SceneBatch]):
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        scene_names: tuple[str, ...] = POC_NERF_SYNTHETIC_SCENES,
        train_query_scene_names: tuple[str, ...] | None = None,
        eval_scene_names: tuple[str, ...] | None = None,
        image_size: int = 224,
        support_views: int = 64,
        query_views: int | None = None,
        train_query_views: int | None = None,
        eval_query_views: int | None = None,
        query_split: str = "val",
        size: int = 128,
        white_background: bool = True,
        cache_images: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else _default_root()
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")
        if query_split not in {"val", "test"}:
            raise ValueError("query_split must be one of {'val', 'test'}")

        scene_dirs = sorted(
            path for path in self.root.iterdir() if (path / "transforms_train.json").exists()
        )
        scene_map = {scene_dir.name: _load_scene(scene_dir) for scene_dir in scene_dirs}

        def resolve_scene_group(names: tuple[str, ...] | None) -> tuple[BlenderScene, ...]:
            if names is None:
                return tuple(scene_map[name] for name in sorted(scene_map))
            missing = [name for name in names if name not in scene_map]
            if missing:
                raise ValueError(f"Unknown scenes requested: {missing}")
            return tuple(scene_map[name] for name in names)

        self.support_scenes = resolve_scene_group(scene_names)
        if not self.support_scenes:
            raise ValueError(f"no Blender scenes found under {self.root}")
        self.train_query_scene_names = train_query_scene_names
        self.eval_scene_names = eval_scene_names
        self.train_query_scenes = (
            self.support_scenes
            if train_query_scene_names is None
            else resolve_scene_group(train_query_scene_names)
        )
        self.eval_scenes = (
            self.support_scenes
            if eval_scene_names is None
            else resolve_scene_group(eval_scene_names)
        )

        self.image_size = image_size
        self.support_views = support_views
        resolved_query_views = query_views or 16
        self.train_query_views = train_query_views or resolved_query_views
        self.eval_query_views = eval_query_views or resolved_query_views
        self.query_split = query_split
        self.size = max(size, len(self.support_scenes), len(self.train_query_scenes), len(self.eval_scenes))
        self.white_background = white_background
        self.cache_images = cache_images
        self._image_cache: dict[Path, torch.Tensor] = {}

    def __len__(self) -> int:
        return self.size

    def _query_frames_for_scene(self, scene: BlenderScene) -> tuple[BlenderFrame, ...]:
        return scene.val_frames if self.query_split == "val" else scene.test_frames

    def _split_query_frame_pool(
        self,
        scene: BlenderScene,
    ) -> tuple[tuple[BlenderFrame, ...], tuple[BlenderFrame, ...]]:
        query_frames = self._query_frames_for_scene(scene)
        train_pool = query_frames[::2]
        eval_pool = query_frames[1::2]
        if not train_pool:
            train_pool = query_frames
        if not eval_pool:
            eval_pool = query_frames
        return train_pool, eval_pool

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

    def _build_query_bundle(
        self,
        scene: BlenderScene,
        frame_pool: tuple[BlenderFrame, ...],
        query_views: int,
        *,
        query_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_query_frames = _select_frames(
            frame_pool,
            query_views,
            offset=query_offset,
        )
        intrinsics = _camera_intrinsics(
            scene.camera_angle_x,
            scene.original_size,
            self.image_size,
        )
        query_intrinsics = intrinsics.expand(len(selected_query_frames), -1, -1).clone()
        query_c2w = torch.stack([frame.c2w for frame in selected_query_frames], dim=0)
        target_rgb = torch.stack(
            [self._load_image(frame.image_path) for frame in selected_query_frames],
            dim=0,
        )
        return query_intrinsics, query_c2w, target_rgb

    def __getitem__(self, index: int) -> SceneBatch:
        support_scene = self.support_scenes[index % len(self.support_scenes)]
        train_query_scene = (
            support_scene
            if self.train_query_scene_names is None
            else self.train_query_scenes[(index * 3) % len(self.train_query_scenes)]
        )
        eval_scene = (
            support_scene
            if self.eval_scene_names is None
            else self.eval_scenes[(index * 5) % len(self.eval_scenes)]
        )
        support_offset = index % len(support_scene.train_frames)
        train_query_pool, _ = self._split_query_frame_pool(train_query_scene)
        _, eval_query_pool = self._split_query_frame_pool(eval_scene)
        train_query_offset = (index * 7) % len(train_query_pool)
        eval_query_offset = (index * 11) % len(eval_query_pool)

        support_frames = _select_frames(
            support_scene.train_frames,
            self.support_views,
            offset=support_offset,
        )

        intrinsics = _camera_intrinsics(
            support_scene.camera_angle_x,
            support_scene.original_size,
            self.image_size,
        )
        support_intrinsics = intrinsics.expand(len(support_frames), -1, -1).clone()
        support_c2w = torch.stack([frame.c2w for frame in support_frames], dim=0)

        support_images = torch.stack(
            [self._load_image(frame.image_path) for frame in support_frames],
            dim=0,
        )
        train_query_intrinsics, train_query_c2w, train_target_rgb = self._build_query_bundle(
            train_query_scene,
            train_query_pool,
            self.train_query_views,
            query_offset=train_query_offset,
        )
        eval_query_intrinsics, eval_query_c2w, eval_target_rgb = self._build_query_bundle(
            eval_scene,
            eval_query_pool,
            self.eval_query_views,
            query_offset=eval_query_offset,
        )
        return SceneBatch(
            support_images=support_images,
            support_intrinsics=support_intrinsics,
            support_c2w=support_c2w,
            train_query_intrinsics=train_query_intrinsics,
            train_query_c2w=train_query_c2w,
            train_target_rgb=train_target_rgb,
            eval_query_intrinsics=eval_query_intrinsics,
            eval_query_c2w=eval_query_c2w,
            eval_target_rgb=eval_target_rgb,
        )


def scene_batch_collate(samples: list[SceneBatch]) -> SceneBatch:
    return SceneBatch(
        support_images=torch.stack([sample.support_images for sample in samples], dim=0),
        support_intrinsics=torch.stack([sample.support_intrinsics for sample in samples], dim=0),
        support_c2w=torch.stack([sample.support_c2w for sample in samples], dim=0),
        train_query_intrinsics=torch.stack(
            [sample.train_query_intrinsics for sample in samples],
            dim=0,
        ),
        train_query_c2w=torch.stack([sample.train_query_c2w for sample in samples], dim=0),
        train_target_rgb=torch.stack([sample.train_target_rgb for sample in samples], dim=0),
        eval_query_intrinsics=torch.stack(
            [sample.eval_query_intrinsics for sample in samples],
            dim=0,
        ),
        eval_query_c2w=torch.stack([sample.eval_query_c2w for sample in samples], dim=0),
        eval_target_rgb=torch.stack([sample.eval_target_rgb for sample in samples], dim=0),
    )
