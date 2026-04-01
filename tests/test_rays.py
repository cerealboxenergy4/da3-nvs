from __future__ import annotations

import torch

from da3_nvs.data import default_intrinsics, raymap_from_cameras
from da3_nvs.models import RayMapEncoder


def test_raymap_from_cameras_matches_expected_center_ray() -> None:
    intrinsics = default_intrinsics(7, 7)
    c2w = torch.eye(4)

    raymap = raymap_from_cameras(intrinsics, c2w, 7, 7)

    assert raymap.shape == (9, 7, 7)
    center_dir = raymap[3:6, 3, 3]
    assert torch.allclose(center_dir, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    assert torch.allclose(raymap[6:9, 3, 3], torch.zeros(3), atol=1e-5)


def test_raymap_can_drop_moment_channels() -> None:
    intrinsics = default_intrinsics(8, 8).expand(2, -1, -1)
    c2w = torch.eye(4).expand(2, -1, -1)

    raymap = raymap_from_cameras(intrinsics, c2w, 8, 8, include_moment=False)

    assert raymap.shape == (2, 6, 8, 8)


def test_ray_encoder_respects_patch_grid() -> None:
    encoder = RayMapEncoder(patch_size=4, embed_dim=32, ray_channels=9)
    raymaps = torch.randn(2, 9, 8, 12)

    tokens = encoder(raymaps)

    assert tokens.shape == (2, 6, 32)

