from __future__ import annotations

import torch


def _as_homogeneous_pose(c2w: torch.Tensor) -> torch.Tensor:
    if c2w.shape[-2:] == (4, 4):
        return c2w
    if c2w.shape[-2:] != (3, 4):
        raise ValueError("camera poses must end with shape (3, 4) or (4, 4)")

    out = torch.eye(4, device=c2w.device, dtype=c2w.dtype).expand(*c2w.shape[:-2], -1, -1).clone()
    out[..., :3, :4] = c2w
    return out


def patch_centers_homogeneous(
    image_size: tuple[int, int],
    patch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    height, width = image_size
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"image_size must be divisible by patch_size={patch_size}, got {image_size}"
        )

    patch_rows = height // patch_size
    patch_cols = width // patch_size
    ys, xs = torch.meshgrid(
        torch.arange(patch_rows, device=device, dtype=dtype),
        torch.arange(patch_cols, device=device, dtype=dtype),
        indexing="ij",
    )
    centers_x = (xs + 0.5) * patch_size
    centers_y = (ys + 0.5) * patch_size
    ones = torch.ones_like(centers_x)
    return torch.stack([centers_x, centers_y, ones], dim=-1).reshape(-1, 3)


def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(v[..., 0])
    vx, vy, vz = v.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([zeros, -vz, vy], dim=-1),
            torch.stack([vz, zeros, -vx], dim=-1),
            torch.stack([-vy, vx, zeros], dim=-1),
        ],
        dim=-2,
    )


def fundamental_matrices(
    query_intrinsics: torch.Tensor,
    query_c2w: torch.Tensor,
    support_intrinsics: torch.Tensor,
    support_c2w: torch.Tensor,
) -> torch.Tensor:
    if query_intrinsics.shape[-2:] != (3, 3) or support_intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must end with shape (3, 3)")

    query_c2w = _as_homogeneous_pose(query_c2w)
    support_c2w = _as_homogeneous_pose(support_c2w)

    query_w2c = torch.linalg.inv(query_c2w)
    support_w2c = torch.linalg.inv(support_c2w)

    query_r = query_w2c[..., :3, :3]
    support_r = support_w2c[..., :3, :3]
    query_t = query_w2c[..., :3, 3]
    support_t = support_w2c[..., :3, 3]

    r_rel = support_r[:, None, :, :, :] @ query_r[:, :, None, :, :].transpose(-1, -2)
    t_rel = support_t[:, None, :, :, None] - (r_rel @ query_t[:, :, None, :, None])
    t_rel = t_rel.squeeze(-1)

    essential = _skew_symmetric(t_rel) @ r_rel
    support_k_inv_t = torch.linalg.inv(support_intrinsics).transpose(-1, -2)[:, None, :, :, :]
    query_k_inv = torch.linalg.inv(query_intrinsics)[:, :, None, :, :]
    return support_k_inv_t @ essential @ query_k_inv


def compute_epipolar_attention_mask(
    query_intrinsics: torch.Tensor,
    query_c2w: torch.Tensor,
    support_intrinsics: torch.Tensor,
    support_c2w: torch.Tensor,
    *,
    query_image_size: tuple[int, int],
    support_image_size: tuple[int, int],
    patch_size: int,
    patch_band_width: float = 1.0,
) -> torch.Tensor:
    batch_size, query_views = query_intrinsics.shape[:2]
    support_views = support_intrinsics.shape[:2][1]
    device = query_intrinsics.device
    dtype = query_intrinsics.dtype

    query_points = patch_centers_homogeneous(
        query_image_size,
        patch_size,
        device=device,
        dtype=dtype,
    )
    support_points = patch_centers_homogeneous(
        support_image_size,
        patch_size,
        device=device,
        dtype=dtype,
    )
    num_query_patches = query_points.shape[0]
    num_support_patches = support_points.shape[0]
    band_width_px = patch_band_width * float(patch_size)

    fundamentals = fundamental_matrices(
        query_intrinsics,
        query_c2w,
        support_intrinsics,
        support_c2w,
    )
    mask = torch.zeros(
        batch_size,
        query_views * num_query_patches,
        support_views * num_support_patches,
        device=device,
        dtype=torch.bool,
    )

    support_points_t = support_points.transpose(0, 1)
    for batch_idx in range(batch_size):
        for query_view_idx in range(query_views):
            query_row_start = query_view_idx * num_query_patches
            query_row_end = query_row_start + num_query_patches
            for support_view_idx in range(support_views):
                support_col_start = support_view_idx * num_support_patches
                support_col_end = support_col_start + num_support_patches
                f_mat = fundamentals[batch_idx, query_view_idx, support_view_idx]
                lines = (f_mat @ query_points.transpose(0, 1)).transpose(0, 1)
                line_norm = torch.linalg.vector_norm(lines[:, :2], dim=-1)
                distances = torch.abs(lines @ support_points_t) / line_norm.clamp_min(1e-6).unsqueeze(-1)
                allowed = distances <= band_width_px
                degenerate = line_norm <= 1e-6
                if degenerate.any():
                    allowed[degenerate] = True

                mask[
                    batch_idx,
                    query_row_start:query_row_end,
                    support_col_start:support_col_end,
                ] = allowed

        valid_rows = mask[batch_idx].any(dim=-1)
        if not valid_rows.all():
            mask[batch_idx, ~valid_rows, :] = True

    return mask
