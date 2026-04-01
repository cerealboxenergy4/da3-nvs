from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def default_intrinsics(
    height: int,
    width: int,
    focal: float | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if focal is None:
        focal = 0.5 * float(max(height, width))

    return torch.tensor(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype or torch.float32,
    )


def orbit_camera_pose(
    angle_radians: float,
    radius: float = 2.2,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    dtype = dtype or torch.float32
    origin = torch.tensor(
        [
            radius * math.cos(angle_radians),
            0.25 * radius,
            radius * math.sin(angle_radians),
        ],
        device=device,
        dtype=dtype,
    )

    forward = F.normalize(-origin, dim=0)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    right = F.normalize(torch.cross(world_up, forward, dim=0), dim=0)
    up = F.normalize(torch.cross(forward, right, dim=0), dim=0)

    c2w = torch.eye(4, device=device, dtype=dtype)
    c2w[:3, :3] = torch.stack([right, up, forward], dim=-1)
    c2w[:3, 3] = origin
    return c2w


def _as_homogeneous_pose(c2w: torch.Tensor) -> torch.Tensor:
    if c2w.shape[-2:] == (4, 4):
        return c2w
    if c2w.shape[-2:] != (3, 4):
        raise ValueError("camera poses must end with shape (3, 4) or (4, 4)")

    out = torch.eye(4, device=c2w.device, dtype=c2w.dtype).expand(*c2w.shape[:-2], -1, -1).clone()
    out[..., :3, :4] = c2w
    return out


def raymap_from_cameras(
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    height: int,
    width: int,
    *,
    include_moment: bool = True,
) -> torch.Tensor:
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must end with shape (3, 3)")

    c2w = _as_homogeneous_pose(c2w)
    if intrinsics.shape[:-2] != c2w.shape[:-2]:
        raise ValueError("intrinsics and c2w batch dimensions must match")

    batch_shape = intrinsics.shape[:-2]
    flat_batch = int(math.prod(batch_shape)) if batch_shape else 1
    intrinsics_flat = intrinsics.reshape(flat_batch, 3, 3)
    c2w_flat = c2w.reshape(flat_batch, 4, 4)

    device = intrinsics.device
    dtype = intrinsics.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(-1, 3)

    inv_k = torch.linalg.inv(intrinsics_flat)
    cam_dirs = torch.einsum("bij,pj->bpi", inv_k, pixels)
    cam_dirs = F.normalize(cam_dirs, dim=-1)

    rotation = c2w_flat[:, :3, :3]
    world_dirs = torch.einsum("bij,bpj->bpi", rotation, cam_dirs)
    world_dirs = F.normalize(world_dirs, dim=-1)

    origins = c2w_flat[:, :3, 3][:, None, :].expand_as(world_dirs)
    parts = [origins, world_dirs]
    if include_moment:
        parts.append(torch.cross(origins, world_dirs, dim=-1))

    raymap = torch.cat(parts, dim=-1)
    channels = raymap.shape[-1]
    raymap = raymap.reshape(*batch_shape, height, width, channels)
    permute_order = list(range(len(batch_shape))) + [
        len(batch_shape) + 2,
        len(batch_shape),
        len(batch_shape) + 1,
    ]
    return raymap.permute(permute_order)

