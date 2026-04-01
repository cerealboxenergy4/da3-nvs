from __future__ import annotations

import torch
import torch.nn.functional as F


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.dim() != 4:
        raise ValueError("images must have shape (B, C, H, W)")

    _, _, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")

    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2)


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    image_size: tuple[int, int],
    channels: int,
) -> torch.Tensor:
    if patches.dim() != 3:
        raise ValueError("patches must have shape (B, P, C * patch_size^2)")

    height, width = image_size
    grid_height = height // patch_size
    grid_width = width // patch_size
    expected_patches = grid_height * grid_width

    if patches.shape[1] != expected_patches:
        raise ValueError("number of patches does not match image_size and patch_size")
    if patches.shape[2] != channels * patch_size * patch_size:
        raise ValueError("patch channel dimension is inconsistent with decoder settings")

    return F.fold(
        patches.transpose(1, 2),
        output_size=image_size,
        kernel_size=patch_size,
        stride=patch_size,
    )


def num_patches(image_size: tuple[int, int], patch_size: int) -> int:
    height, width = image_size
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")
    return (height // patch_size) * (width // patch_size)

