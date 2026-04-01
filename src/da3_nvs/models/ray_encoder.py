from __future__ import annotations

import torch
from torch import nn

from da3_nvs.models.common import patchify


class RayMapEncoder(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 14,
        embed_dim: int = 1024,
        ray_channels: int = 9,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.ray_channels = ray_channels

        hidden_dim = hidden_dim or embed_dim
        patch_dim = ray_channels * patch_size * patch_size
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, raymaps: torch.Tensor) -> torch.Tensor:
        if raymaps.dim() != 4:
            raise ValueError("raymaps must have shape (B, C, H, W)")
        if raymaps.shape[1] != self.ray_channels:
            raise ValueError(
                f"Expected raymaps with {self.ray_channels} channels, got {raymaps.shape[1]}"
            )

        patches = patchify(raymaps, self.patch_size)
        return self.norm(self.proj(patches))

