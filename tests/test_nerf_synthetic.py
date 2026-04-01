from __future__ import annotations

import json

import torch
from PIL import Image

from da3_nvs.data import NerfSyntheticSceneDataset, scene_batch_collate


def _write_scene(root, scene_name: str, *, frame_count: int = 8) -> None:
    scene_dir = root / scene_name
    scene_dir.mkdir(parents=True)

    def write_split(split: str) -> None:
        frames = []
        for index in range(frame_count):
            image_name = f"{split}_{index:03d}.png"
            image_path = scene_dir / image_name
            Image.new(
                "RGBA",
                (32, 32),
                color=(index * 10 % 255, index * 20 % 255, index * 30 % 255, 255),
            ).save(image_path)
            transform = torch.eye(4, dtype=torch.float32)
            transform[0, 3] = 0.1 * index
            frames.append(
                {
                    "file_path": image_name,
                    "transform_matrix": transform.tolist(),
                }
            )
        metadata = {
            "camera_angle_x": 0.6911112070083618,
            "frames": frames,
        }
        (scene_dir / f"transforms_{split}.json").write_text(json.dumps(metadata))

    write_split("train")
    write_split("val")
    write_split("test")


def test_nerf_synthetic_dataset_emits_poc_batch_shapes(tmp_path) -> None:
    _write_scene(tmp_path, "chair")
    dataset = NerfSyntheticSceneDataset(
        root=tmp_path,
        scene_names=("chair",),
        image_size=32,
        support_views=4,
        train_query_views=2,
        eval_query_views=3,
        size=1,
    )

    sample = dataset[0]
    batch = scene_batch_collate([sample, sample])

    assert sample.support_images.shape == (4, 3, 32, 32)
    assert sample.support_intrinsics.shape == (4, 3, 3)
    assert sample.support_c2w.shape == (4, 4, 4)
    assert sample.train_query_intrinsics.shape == (2, 3, 3)
    assert sample.train_query_c2w.shape == (2, 4, 4)
    assert sample.train_target_rgb.shape == (2, 3, 32, 32)
    assert sample.eval_query_intrinsics.shape == (3, 3, 3)
    assert sample.eval_query_c2w.shape == (3, 4, 4)
    assert sample.eval_target_rgb.shape == (3, 3, 32, 32)
    assert batch.support_images.shape == (2, 4, 3, 32, 32)
