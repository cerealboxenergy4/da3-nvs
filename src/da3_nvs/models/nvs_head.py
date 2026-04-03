from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _resolve_num_heads(embed_dim: int, preferred_num_heads: int) -> int:
    candidates = [
        preferred_num_heads,
        min(preferred_num_heads, embed_dim),
        8,
        4,
        2,
        1,
    ]
    for num_heads in candidates:
        if num_heads > 0 and embed_dim % num_heads == 0:
            return num_heads
    raise ValueError(f"Could not find a valid attention head count for embed_dim={embed_dim}")


def custom_interpolate(
    x: torch.Tensor,
    *,
    size: tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    int_max = 1610612736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]
    if total > int_max:
        chunks = torch.chunk(x, chunks=(total // int_max) + 1, dim=0)
        outs = [
            F.interpolate(chunk, size=size, mode=mode, align_corners=align_corners)
            for chunk in chunks
        ]
        return torch.cat(outs, dim=0).contiguous()
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, *, activation: nn.Module, groups: int = 1) -> None:
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        activation: nn.Module,
        align_corners: bool = True,
        has_residual: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.has_residual = has_residual
        self.res_conf_unit1 = (
            ResidualConvUnit(features, activation=activation, groups=groups)
            if has_residual
            else None
        )
        self.res_conf_unit2 = ResidualConvUnit(features, activation=activation, groups=groups)
        self.out_conv = nn.Conv2d(features, features, 1, 1, 0, bias=True, groups=groups)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.res_conf_unit1 is not None:
            y = self.skip_add.add(y, self.res_conf_unit1(xs[1]))

        y = self.res_conf_unit2(y)
        y = custom_interpolate(
            y,
            size=size if size is not None else (y.shape[-2] * 2, y.shape[-1] * 2),
            mode="bilinear",
            align_corners=self.align_corners,
        )
        y = self.out_conv(y)
        return y


def _make_fusion_block(
    features: int,
    *,
    has_residual: bool = True,
    inplace: bool = False,
) -> nn.Module:
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=inplace),
        align_corners=True,
        has_residual=has_residual,
    )


def _make_scratch(in_shape: Sequence[int], out_shape: int) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, 3, 1, 1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, 3, 1, 1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, 3, 1, 1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, 3, 1, 1, bias=False)
    return scratch


class DPTStyleRGBDecoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        patch_size: int,
        features: int | None = None,
        out_channels: int = 3,
        stage_channels: Sequence[int] | None = None,
        query_skip_channels: int = 0,
        support_skip_channels: int = 0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.features = features or min(256, max(embed_dim // 2, 32))
        self.stage_channels = tuple(stage_channels or (
            self.features,
            self.features * 2,
            self.features * 4,
            self.features * 4,
        ))
        self.token_norm = nn.LayerNorm(embed_dim)

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(embed_dim, stage_channel, kernel_size=1, stride=1, padding=0)
                for stage_channel in self.stage_channels
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    self.stage_channels[0],
                    self.stage_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    self.stage_channels[1],
                    self.stage_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    self.stage_channels[3],
                    self.stage_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.scratch = _make_scratch(list(self.stage_channels), self.features)
        self.scratch.refinenet1 = _make_fusion_block(self.features)
        self.scratch.refinenet2 = _make_fusion_block(self.features)
        self.scratch.refinenet3 = _make_fusion_block(self.features)
        self.scratch.refinenet4 = _make_fusion_block(self.features, has_residual=False)

        mid_features = max(self.features // 2, 32)
        head_features = max(self.features // 4, 32)
        self.output_conv1 = nn.Conv2d(self.features, mid_features, kernel_size=3, stride=1, padding=1)
        total_skip_channels = query_skip_channels + support_skip_channels
        self.skip_fuse = None
        if total_skip_channels > 0:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(mid_features + total_skip_channels, mid_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_features, mid_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(mid_features, head_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def _reshape_tokens(
        self,
        tokens: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        if tokens.dim() != 4:
            raise ValueError("stage tokens must have shape (B, V, P, D)")

        batch_size, num_views, num_patches, embed_dim = tokens.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {embed_dim}")

        height, width = image_size
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size={self.patch_size}, got {image_size}"
            )
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        expected_patches = patch_height * patch_width
        if num_patches != expected_patches:
            raise ValueError(
                f"Token count {num_patches} does not match patch grid {patch_height}x{patch_width}"
            )

        flattened = tokens.reshape(batch_size * num_views, num_patches, embed_dim)
        flattened = self.token_norm(flattened)
        return flattened.transpose(1, 2).reshape(batch_size * num_views, embed_dim, patch_height, patch_width)

    def _fuse(self, feats: list[torch.Tensor]) -> torch.Tensor:
        layer1, layer2, layer3, layer4 = feats
        layer1_rn = self.scratch.layer1_rn(layer1)
        layer2_rn = self.scratch.layer2_rn(layer2)
        layer3_rn = self.scratch.layer3_rn(layer3)
        layer4_rn = self.scratch.layer4_rn(layer4)

        out = self.scratch.refinenet4(layer4_rn, size=layer3_rn.shape[-2:])
        out = self.scratch.refinenet3(out, layer3_rn, size=layer2_rn.shape[-2:])
        out = self.scratch.refinenet2(out, layer2_rn, size=layer1_rn.shape[-2:])
        out = self.scratch.refinenet1(out, layer1_rn)
        return out

    def forward(
        self,
        stage_tokens: list[torch.Tensor],
        *,
        image_size: tuple[int, int],
        query_ray_skip: torch.Tensor | None = None,
        support_rgb_skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(stage_tokens) != 4:
            raise ValueError(f"DPT-style decoder expects 4 stage tensors, got {len(stage_tokens)}")

        resized_feats = []
        for stage_idx, tokens in enumerate(stage_tokens):
            feature_map = self._reshape_tokens(tokens, image_size=image_size)
            feature_map = self.projects[stage_idx](feature_map)
            feature_map = self.resize_layers[stage_idx](feature_map)
            resized_feats.append(feature_map)

        fused = self._fuse(resized_feats)
        fused = self.output_conv1(fused)
        batch_size, num_views = stage_tokens[0].shape[:2]
        half_size = (max(image_size[0] // 2, 1), max(image_size[1] // 2, 1))
        fused = custom_interpolate(fused, size=half_size, mode="bilinear", align_corners=True)

        skip_parts = [fused]
        expected_batch_views = batch_size * num_views
        if query_ray_skip is not None:
            if query_ray_skip.dim() != 5:
                raise ValueError("query_ray_skip must have shape (B, V, C, H, W)")
            query_ray_skip = query_ray_skip.reshape(expected_batch_views, *query_ray_skip.shape[2:])
            if query_ray_skip.shape[-2:] != half_size:
                query_ray_skip = custom_interpolate(
                    query_ray_skip,
                    size=half_size,
                    mode="bilinear",
                    align_corners=True,
                )
            skip_parts.append(query_ray_skip)
        if support_rgb_skip is not None:
            if support_rgb_skip.dim() != 5:
                raise ValueError("support_rgb_skip must have shape (B, V, C, H, W)")
            support_rgb_skip = support_rgb_skip.reshape(expected_batch_views, *support_rgb_skip.shape[2:])
            if support_rgb_skip.shape[-2:] != half_size:
                support_rgb_skip = custom_interpolate(
                    support_rgb_skip,
                    size=half_size,
                    mode="bilinear",
                    align_corners=True,
                )
            skip_parts.append(support_rgb_skip)
        if self.skip_fuse is not None and len(skip_parts) > 1:
            fused = self.skip_fuse(torch.cat(skip_parts, dim=1))

        fused = custom_interpolate(fused, size=image_size, mode="bilinear", align_corners=True)
        pred_rgb = torch.sigmoid(self.output_conv2(fused))

        return pred_rgb.reshape(batch_size, num_views, -1, image_size[0], image_size[1])


class HalfResCNNEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or max(out_channels, 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TokenUNetRGBDecoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        patch_size: int,
        hidden_dim: int | None = None,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim or min(256, max(embed_dim // 2, 32))
        bottleneck_dim = max(self.hidden_dim * 2, 32)
        refine_dim = max(self.hidden_dim // 2, 16)

        self.token_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(embed_dim, self.hidden_dim, kernel_size=1, stride=1, padding=0)
        self.encoder = ConvBlock(self.hidden_dim, self.hidden_dim)
        self.downsample = nn.Sequential(
            nn.Conv2d(self.hidden_dim, bottleneck_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.bottleneck = ConvBlock(bottleneck_dim, bottleneck_dim)
        self.up_project = nn.Sequential(
            nn.Conv2d(bottleneck_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.decoder = ConvBlock(self.hidden_dim * 2, self.hidden_dim)
        self.refine = ConvBlock(self.hidden_dim, refine_dim)
        self.head = nn.Sequential(
            nn.Conv2d(refine_dim, refine_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(refine_dim, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        if tokens.dim() != 4:
            raise ValueError("tokens must have shape (B, V, P, D)")

        batch_size, num_views, num_patches, embed_dim = tokens.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {embed_dim}")

        patch_height = image_size[0] // self.patch_size
        patch_width = image_size[1] // self.patch_size
        if patch_height * patch_width != num_patches:
            raise ValueError(
                "Token count does not match image patch grid: "
                f"{num_patches} vs {patch_height}x{patch_width}"
            )

        flat_tokens = self.token_norm(tokens.reshape(batch_size * num_views, num_patches, embed_dim))
        feature_map = flat_tokens.transpose(1, 2).reshape(
            batch_size * num_views,
            embed_dim,
            patch_height,
            patch_width,
        )
        skip = self.encoder(self.proj(feature_map))
        down = self.downsample(skip)
        bottleneck = self.bottleneck(down)
        up = custom_interpolate(
            bottleneck,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        up = self.up_project(up)
        feature_map = self.decoder(torch.cat([skip, up], dim=1))
        feature_map = custom_interpolate(
            feature_map,
            size=(max(image_size[0] // 2, 1), max(image_size[1] // 2, 1)),
            mode="bilinear",
            align_corners=True,
        )
        feature_map = self.refine(feature_map)
        feature_map = custom_interpolate(
            feature_map,
            size=image_size,
            mode="bilinear",
            align_corners=True,
        )
        pred_rgb = torch.sigmoid(self.head(feature_map))
        return pred_rgb.reshape(batch_size, num_views, pred_rgb.shape[1], image_size[0], image_size[1])


class CrossAttentionNVSHead(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        patch_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        decoder_hidden_dim: int | None = None,
        out_channels: int = 3,
        query_skip_channels: int = 0,
        support_skip_channels: int = 0,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        resolved_heads = _resolve_num_heads(embed_dim, num_heads)
        self.num_heads = resolved_heads
        self.head_dim = embed_dim // resolved_heads
        self.scale = self.head_dim ** -0.5

        self.query_norm = nn.LayerNorm(embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.decoder = DPTStyleRGBDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            features=decoder_hidden_dim,
            out_channels=out_channels,
            query_skip_channels=query_skip_channels,
            support_skip_channels=support_skip_channels,
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor | Sequence[torch.Tensor],
        *,
        image_size: tuple[int, int],
        epipolar_mask: torch.Tensor | None = None,
        query_ray_skip: torch.Tensor | None = None,
        support_rgb_skip: torch.Tensor | None = None,
        support_image_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_tokens.dim() != 4:
            raise ValueError("query_tokens must have shape (B, V, P, D)")

        batch_size, num_views, num_patches, embed_dim = query_tokens.shape
        flat_queries = query_tokens.reshape(batch_size, num_views * num_patches, embed_dim)
        if isinstance(memory_tokens, torch.Tensor):
            memory_stages = [memory_tokens]
        else:
            memory_stages = list(memory_tokens)
        if not memory_stages:
            raise ValueError("memory_tokens must contain at least one stage")
        for stage_idx, stage_memory in enumerate(memory_stages):
            if stage_memory.dim() != 3:
                raise ValueError(
                    f"memory_tokens stage {stage_idx} must have shape (B, T, D), got {stage_memory.shape}"
                )

        normalized_query = self.query_norm(flat_queries)
        q = self.q_proj(normalized_query).reshape(
            batch_size,
            flat_queries.shape[1],
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        def attend_to_memory(stage_memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            normalized_memory = self.memory_norm(stage_memory)
            if epipolar_mask is not None and epipolar_mask.shape != (
                batch_size,
                flat_queries.shape[1],
                normalized_memory.shape[1],
            ):
                raise ValueError(
                    "epipolar_mask must have shape "
                    f"{(batch_size, flat_queries.shape[1], normalized_memory.shape[1])}, "
                    f"got {tuple(epipolar_mask.shape)}"
                )

            k = self.k_proj(normalized_memory).reshape(
                batch_size,
                normalized_memory.shape[1],
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)
            v = self.v_proj(normalized_memory).reshape(
                batch_size,
                normalized_memory.shape[1],
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if epipolar_mask is not None:
                attn_logits = attn_logits.masked_fill(
                    ~epipolar_mask[:, None, :, :],
                    torch.finfo(attn_logits.dtype).min,
                )
            attn = torch.softmax(attn_logits, dim=-1)
            attended = torch.matmul(attn, v).transpose(1, 2).reshape(
                batch_size,
                flat_queries.shape[1],
                embed_dim,
            )
            return self.out_proj(attended), attn.mean(dim=1)

        stage_fused_tokens: list[torch.Tensor] = []
        final_attn_weights: torch.Tensor | None = None
        for stage_memory in memory_stages:
            attended, stage_attn_weights = attend_to_memory(stage_memory)
            fused = flat_queries + attended
            stage_fused_tokens.append(fused.reshape(batch_size, num_views, num_patches, embed_dim))
            final_attn_weights = stage_attn_weights

        final_stage_flat = stage_fused_tokens[-1].reshape(batch_size, num_views * num_patches, embed_dim)
        rendered = final_stage_flat + self.ffn(self.ffn_norm(final_stage_flat))
        rendered_tokens = rendered.reshape(batch_size, num_views, num_patches, embed_dim)

        if len(stage_fused_tokens) >= 4:
            decoder_stages = stage_fused_tokens[:3] + [rendered_tokens]
        else:
            while len(stage_fused_tokens) < 3:
                stage_fused_tokens.append(stage_fused_tokens[-1])
            decoder_stages = stage_fused_tokens[:3] + [rendered_tokens]

        aggregated_support_rgb_skip = None
        if support_rgb_skip is not None:
            if support_image_size is None:
                raise ValueError("support_image_size is required when support_rgb_skip is provided")
            if final_attn_weights is None:
                raise RuntimeError("Expected final attention weights when support_rgb_skip is enabled")
            if support_rgb_skip.dim() != 5:
                raise ValueError("support_rgb_skip must have shape (B, V, C, H, W)")

            support_views = support_rgb_skip.shape[1]
            support_patch_h = support_image_size[0] // self.decoder.patch_size
            support_patch_w = support_image_size[1] // self.decoder.patch_size
            pooled_support_skip = F.adaptive_avg_pool2d(
                support_rgb_skip.reshape(batch_size * support_views, *support_rgb_skip.shape[2:]),
                (support_patch_h, support_patch_w),
            )
            pooled_support_skip = pooled_support_skip.reshape(
                batch_size,
                support_views,
                pooled_support_skip.shape[1],
                support_patch_h,
                support_patch_w,
            )
            pooled_support_skip = pooled_support_skip.permute(0, 1, 3, 4, 2).reshape(
                batch_size,
                support_views * support_patch_h * support_patch_w,
                pooled_support_skip.shape[2],
            )
            if pooled_support_skip.shape[1] != final_attn_weights.shape[-1]:
                raise RuntimeError(
                    "support_rgb_skip patch grid does not match attention memory tokens: "
                    f"{pooled_support_skip.shape[1]} vs {final_attn_weights.shape[-1]}"
                )
            query_patch_h = image_size[0] // self.decoder.patch_size
            query_patch_w = image_size[1] // self.decoder.patch_size
            aggregated_support_rgb_skip = torch.matmul(final_attn_weights, pooled_support_skip)
            aggregated_support_rgb_skip = aggregated_support_rgb_skip.reshape(
                batch_size,
                num_views,
                query_patch_h,
                query_patch_w,
                pooled_support_skip.shape[-1],
            ).permute(0, 1, 4, 2, 3)

        pred_rgb = self.decoder(
            decoder_stages,
            image_size=image_size,
            query_ray_skip=query_ray_skip,
            support_rgb_skip=aggregated_support_rgb_skip,
        )
        return pred_rgb, rendered_tokens


class HybridCrossAttentionCNNNVSHead(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        patch_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        decoder_hidden_dim: int | None = None,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        resolved_heads = _resolve_num_heads(embed_dim, num_heads)
        self.num_heads = resolved_heads
        self.head_dim = embed_dim // resolved_heads
        self.scale = self.head_dim ** -0.5

        self.query_norm = nn.LayerNorm(embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.decoder = TokenUNetRGBDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            hidden_dim=decoder_hidden_dim,
            out_channels=out_channels,
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor | Sequence[torch.Tensor],
        *,
        image_size: tuple[int, int],
        epipolar_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_tokens.dim() != 4:
            raise ValueError("query_tokens must have shape (B, V, P, D)")

        batch_size, num_views, num_patches, embed_dim = query_tokens.shape
        flat_queries = query_tokens.reshape(batch_size, num_views * num_patches, embed_dim)
        if isinstance(memory_tokens, torch.Tensor):
            memory_stages = [memory_tokens]
        else:
            memory_stages = list(memory_tokens)
        if not memory_stages:
            raise ValueError("memory_tokens must contain at least one stage")
        for stage_idx, stage_memory in enumerate(memory_stages):
            if stage_memory.dim() != 3:
                raise ValueError(
                    f"memory_tokens stage {stage_idx} must have shape (B, T, D), got {stage_memory.shape}"
                )

        fused_queries = flat_queries
        for stage_idx, stage_memory in enumerate(memory_stages):
            normalized_query = self.query_norm(fused_queries)
            normalized_memory = self.memory_norm(stage_memory)
            if epipolar_mask is not None and epipolar_mask.shape != (
                batch_size,
                fused_queries.shape[1],
                normalized_memory.shape[1],
            ):
                raise ValueError(
                    "epipolar_mask must have shape "
                    f"{(batch_size, fused_queries.shape[1], normalized_memory.shape[1])}, "
                    f"got {tuple(epipolar_mask.shape)}"
                )

            q = self.q_proj(normalized_query).reshape(
                batch_size,
                fused_queries.shape[1],
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)
            k = self.k_proj(normalized_memory).reshape(
                batch_size,
                normalized_memory.shape[1],
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)
            v = self.v_proj(normalized_memory).reshape(
                batch_size,
                normalized_memory.shape[1],
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if epipolar_mask is not None:
                attn_logits = attn_logits.masked_fill(
                    ~epipolar_mask[:, None, :, :],
                    torch.finfo(attn_logits.dtype).min,
                )
            attn = torch.softmax(attn_logits, dim=-1)
            attended = torch.matmul(attn, v).transpose(1, 2).reshape(
                batch_size,
                fused_queries.shape[1],
                embed_dim,
            )
            fused_queries = fused_queries + self.out_proj(attended)

        rendered = fused_queries + self.ffn(self.ffn_norm(fused_queries))
        rendered_tokens = rendered.reshape(batch_size, num_views, num_patches, embed_dim)
        pred_rgb = self.decoder(rendered_tokens, image_size=image_size)
        return pred_rgb, rendered_tokens
