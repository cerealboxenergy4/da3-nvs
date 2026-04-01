from __future__ import annotations

import torch
from torch import nn

from da3_nvs.models.common import patchify


class RGBPatchEncoder(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 14,
        embed_dim: int = 1024,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        hidden_dim = hidden_dim or embed_dim
        patch_dim = 3 * patch_size * patch_size
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError("images must have shape (B, 3, H, W)")
        if images.shape[1] != 3:
            raise ValueError(f"Expected RGB images with 3 channels, got {images.shape[1]}")

        patches = patchify(images, self.patch_size)
        return self.norm(self.proj(patches))
