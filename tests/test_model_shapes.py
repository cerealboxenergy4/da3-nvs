from __future__ import annotations

import torch
from torch import nn

from da3_nvs.data import SceneBatch, default_intrinsics, orbit_camera_pose
from da3_nvs.models import DA3NVSModel
from da3_nvs.models.common import patchify
from da3_nvs.train import Trainer


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


def build_camera_batch(
    *,
    batch_size: int,
    num_views: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = default_intrinsics(height, width).expand(batch_size, num_views, -1, -1).clone()
    poses = []
    for batch_idx in range(batch_size):
        batch_poses = []
        for view_idx in range(num_views):
            angle = 0.4 * view_idx + 0.1 * batch_idx
            batch_poses.append(orbit_camera_pose(angle))
        poses.append(torch.stack(batch_poses, dim=0))
    return intrinsics, torch.stack(poses, dim=0)


def test_da3_nvs_forward_matches_query_resolution() -> None:
    batch_size = 2
    support_views = 3
    query_views = 2
    patch_size = 8
    height = 32
    width = 32

    model = DA3NVSModel(
        patch_size=patch_size,
        backbone=ToyPatchBackbone(patch_size=patch_size, embed_dim=64),
        num_heads=4,
    )

    support_images = torch.rand(batch_size, support_views, 3, height, width)
    support_intrinsics, support_c2w = build_camera_batch(
        batch_size=batch_size,
        num_views=support_views,
        height=height,
        width=width,
    )
    query_intrinsics, query_c2w = build_camera_batch(
        batch_size=batch_size,
        num_views=query_views,
        height=height,
        width=width,
    )

    outputs = model(
        support_images=support_images,
        support_intrinsics=support_intrinsics,
        support_c2w=support_c2w,
        query_intrinsics=query_intrinsics,
        query_c2w=query_c2w,
    )

    num_patches = (height // patch_size) * (width // patch_size)
    assert outputs.pred_rgb.shape == (batch_size, query_views, 3, height, width)
    assert outputs.query_tokens.shape == (batch_size, query_views, num_patches, 64)
    assert outputs.scene_tokens.shape == (batch_size, support_views * num_patches, 64)


def test_da3_nvs_supports_6d_query_rays() -> None:
    model = DA3NVSModel(
        patch_size=8,
        include_moment=False,
        backbone=ToyPatchBackbone(patch_size=8, embed_dim=48),
        num_heads=4,
    )

    support_images = torch.rand(1, 2, 3, 32, 32)
    support_intrinsics, support_c2w = build_camera_batch(
        batch_size=1,
        num_views=2,
        height=32,
        width=32,
    )
    query_intrinsics, query_c2w = build_camera_batch(
        batch_size=1,
        num_views=1,
        height=32,
        width=32,
    )

    outputs = model(
        support_images=support_images,
        support_intrinsics=support_intrinsics,
        support_c2w=support_c2w,
        query_intrinsics=query_intrinsics,
        query_c2w=query_c2w,
    )

    assert outputs.support_raymaps.shape[2] == 6
    assert outputs.query_raymaps.shape[2] == 6
    assert outputs.pred_rgb.shape == (1, 1, 3, 32, 32)


def test_da3_nvs_can_reuse_encoded_support_scene() -> None:
    model = DA3NVSModel(
        patch_size=8,
        backbone=ToyPatchBackbone(patch_size=8, embed_dim=64),
        num_heads=4,
    )
    support_images = torch.rand(1, 4, 3, 32, 32)
    support_intrinsics, support_c2w = build_camera_batch(
        batch_size=1,
        num_views=4,
        height=32,
        width=32,
    )
    query_intrinsics, query_c2w = build_camera_batch(
        batch_size=1,
        num_views=3,
        height=32,
        width=32,
    )

    scene = model.encode_support(
        support_images=support_images,
        support_intrinsics=support_intrinsics,
        support_c2w=support_c2w,
    )
    outputs = model.render_queries(
        scene,
        query_intrinsics=query_intrinsics,
        query_c2w=query_c2w,
        query_image_size=(32, 32),
    )

    assert scene.scene_tokens.shape == (1, 64, 64)
    assert outputs.pred_rgb.shape == (1, 3, 3, 32, 32)


def test_da3_nvs_accepts_multistage_support_backbone() -> None:
    model = DA3NVSModel(
        patch_size=8,
        backbone=ToyMultiStageBackbone(patch_size=8, embed_dim=64),
        num_heads=4,
    )
    support_images = torch.rand(1, 4, 3, 32, 32)
    support_intrinsics, support_c2w = build_camera_batch(
        batch_size=1,
        num_views=4,
        height=32,
        width=32,
    )
    query_intrinsics, query_c2w = build_camera_batch(
        batch_size=1,
        num_views=2,
        height=32,
        width=32,
    )

    outputs = model(
        support_images=support_images,
        support_intrinsics=support_intrinsics,
        support_c2w=support_c2w,
        query_intrinsics=query_intrinsics,
        query_c2w=query_c2w,
    )

    assert outputs.pred_rgb.shape == (1, 2, 3, 32, 32)
    assert outputs.epipolar_mask is not None


def test_da3_nvs_can_disable_epipolar_masking() -> None:
    model = DA3NVSModel(
        patch_size=8,
        backbone=ToyMultiStageBackbone(patch_size=8, embed_dim=64),
        num_heads=4,
        use_epipolar_masking=False,
    )
    support_images = torch.rand(1, 4, 3, 32, 32)
    support_intrinsics, support_c2w = build_camera_batch(
        batch_size=1,
        num_views=4,
        height=32,
        width=32,
    )
    query_intrinsics, query_c2w = build_camera_batch(
        batch_size=1,
        num_views=2,
        height=32,
        width=32,
    )

    outputs = model(
        support_images=support_images,
        support_intrinsics=support_intrinsics,
        support_c2w=support_c2w,
        query_intrinsics=query_intrinsics,
        query_c2w=query_c2w,
    )

    assert outputs.pred_rgb.shape == (1, 2, 3, 32, 32)
    assert outputs.epipolar_mask is None


def test_da3_nvs_supports_optional_query_and_support_skip_branches() -> None:
    model = DA3NVSModel(
        patch_size=8,
        backbone=ToyMultiStageBackbone(patch_size=8, embed_dim=64),
        num_heads=4,
        use_query_ray_skip=True,
        use_support_rgb_skip=True,
        skip_feature_dim=16,
    )
    support_images = torch.rand(1, 4, 3, 32, 32)
    support_intrinsics, support_c2w = build_camera_batch(
        batch_size=1,
        num_views=4,
        height=32,
        width=32,
    )
    query_intrinsics, query_c2w = build_camera_batch(
        batch_size=1,
        num_views=2,
        height=32,
        width=32,
    )

    outputs = model(
        support_images=support_images,
        support_intrinsics=support_intrinsics,
        support_c2w=support_c2w,
        query_intrinsics=query_intrinsics,
        query_c2w=query_c2w,
    )

    assert outputs.pred_rgb.shape == (1, 2, 3, 32, 32)


def test_da3_nvs_supports_raw_rgb_stage1_decoder_path() -> None:
    model = DA3NVSModel(
        patch_size=8,
        backbone=ToyMultiStageBackbone(patch_size=8, embed_dim=64),
        num_heads=4,
        use_raw_rgb_stage1=True,
    )
    support_images = torch.rand(1, 4, 3, 32, 32)
    support_intrinsics, support_c2w = build_camera_batch(
        batch_size=1,
        num_views=4,
        height=32,
        width=32,
    )
    query_intrinsics, query_c2w = build_camera_batch(
        batch_size=1,
        num_views=2,
        height=32,
        width=32,
    )

    scene = model.encode_support(
        support_images=support_images,
        support_intrinsics=support_intrinsics,
        support_c2w=support_c2w,
    )
    assert scene.support_rgb_patch_tokens is not None
    assert len(scene.scene_decoder_stage_tokens) == 4

    outputs = model.render_queries(
        scene,
        query_intrinsics=query_intrinsics,
        query_c2w=query_c2w,
        query_image_size=(32, 32),
    )

    assert outputs.pred_rgb.shape == (1, 2, 3, 32, 32)


def test_trainer_train_step_runs_for_rgb_loss() -> None:
    model = DA3NVSModel(
        patch_size=8,
        backbone=ToyPatchBackbone(patch_size=8, embed_dim=48),
        num_heads=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer)

    support_intrinsics, support_c2w = build_camera_batch(
        batch_size=1,
        num_views=4,
        height=32,
        width=32,
    )
    train_query_intrinsics, train_query_c2w = build_camera_batch(
        batch_size=1,
        num_views=2,
        height=32,
        width=32,
    )
    eval_query_intrinsics, eval_query_c2w = build_camera_batch(
        batch_size=1,
        num_views=2,
        height=32,
        width=32,
    )

    batch = SceneBatch(
        support_images=torch.rand(1, 4, 3, 32, 32),
        support_intrinsics=support_intrinsics,
        support_c2w=support_c2w,
        train_query_intrinsics=train_query_intrinsics,
        train_query_c2w=train_query_c2w,
        train_target_rgb=torch.rand(1, 2, 3, 32, 32),
        eval_query_intrinsics=eval_query_intrinsics,
        eval_query_c2w=eval_query_c2w,
        eval_target_rgb=torch.rand(1, 2, 3, 32, 32),
    )

    metrics = trainer.train_step(batch)

    assert set(metrics) == {"loss", "train_rgb_recon_loss", "train_epipolar_keep_ratio"}


def test_optimizer_builder_accepts_adamw_and_sgd() -> None:
    from scripts.mock_train import build_optimizer

    model = DA3NVSModel(
        patch_size=8,
        backbone=ToyPatchBackbone(patch_size=8, embed_dim=48),
        num_heads=4,
    )
    args = type(
        "Args",
        (),
        {
            "optimizer": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-2,
            "momentum": 0.9,
        },
    )()
    optimizer = build_optimizer(args, model)
    assert optimizer.__class__.__name__ == "AdamW"

    args.optimizer = "sgd"
    optimizer = build_optimizer(args, model)
    assert optimizer.__class__.__name__ == "SGD"


def test_trainer_reports_tv_loss_when_enabled() -> None:
    model = DA3NVSModel(
        patch_size=8,
        backbone=ToyPatchBackbone(patch_size=8, embed_dim=48),
        num_heads=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, tv_weight=0.1)

    support_intrinsics, support_c2w = build_camera_batch(
        batch_size=1,
        num_views=4,
        height=32,
        width=32,
    )
    train_query_intrinsics, train_query_c2w = build_camera_batch(
        batch_size=1,
        num_views=2,
        height=32,
        width=32,
    )
    eval_query_intrinsics, eval_query_c2w = build_camera_batch(
        batch_size=1,
        num_views=2,
        height=32,
        width=32,
    )

    batch = SceneBatch(
        support_images=torch.rand(1, 4, 3, 32, 32),
        support_intrinsics=support_intrinsics,
        support_c2w=support_c2w,
        train_query_intrinsics=train_query_intrinsics,
        train_query_c2w=train_query_c2w,
        train_target_rgb=torch.rand(1, 2, 3, 32, 32),
        eval_query_intrinsics=eval_query_intrinsics,
        eval_query_c2w=eval_query_c2w,
        eval_target_rgb=torch.rand(1, 2, 3, 32, 32),
    )

    metrics = trainer.train_step(batch)
    assert "train_tv_loss" in metrics
