from __future__ import annotations

import torch

from da3_nvs.data import default_intrinsics
from da3_nvs.models.epipolar import compute_epipolar_attention_mask


def test_epipolar_mask_shape_matches_flattened_query_and_support_tokens() -> None:
    query_intrinsics = default_intrinsics(16, 16).view(1, 1, 3, 3)
    support_intrinsics = default_intrinsics(16, 16).view(1, 1, 3, 3).expand(1, 2, 3, 3).clone()

    query_c2w = torch.eye(4).view(1, 1, 4, 4).clone()
    support_c2w = torch.eye(4).view(1, 1, 4, 4).expand(1, 2, 4, 4).clone()
    support_c2w[:, 1, 0, 3] = 0.25

    mask = compute_epipolar_attention_mask(
        query_intrinsics,
        query_c2w,
        support_intrinsics,
        support_c2w,
        query_image_size=(16, 16),
        support_image_size=(16, 16),
        patch_size=8,
        patch_band_width=1.0,
    )

    assert mask.shape == (1, 4, 8)
    assert mask.dtype == torch.bool
    assert mask.any(dim=-1).all()


def test_epipolar_mask_falls_back_to_full_attention_for_degenerate_geometry() -> None:
    intrinsics = default_intrinsics(16, 16).view(1, 1, 3, 3)
    c2w = torch.eye(4).view(1, 1, 4, 4)

    mask = compute_epipolar_attention_mask(
        intrinsics,
        c2w,
        intrinsics,
        c2w,
        query_image_size=(16, 16),
        support_image_size=(16, 16),
        patch_size=8,
        patch_band_width=1.0,
    )

    assert mask.all()
