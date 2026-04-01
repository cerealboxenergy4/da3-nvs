from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from da3_nvs.data import raymap_from_cameras
from da3_nvs.models.da3_backbone import DA3PatchBackbone
from da3_nvs.models.epipolar import compute_epipolar_attention_mask
from da3_nvs.models.nvs_head import CrossAttentionNVSHead, HalfResCNNEncoder
from da3_nvs.models.rgb_patch_encoder import RGBPatchEncoder
from da3_nvs.models.ray_encoder import RayMapEncoder


@dataclass
class DA3NVSOutputs:
    pred_rgb: torch.Tensor
    scene_tokens: torch.Tensor
    support_tokens: torch.Tensor
    query_tokens: torch.Tensor
    rendered_tokens: torch.Tensor
    support_raymaps: torch.Tensor
    query_raymaps: torch.Tensor
    epipolar_mask: torch.Tensor | None = None


@dataclass
class DA3SceneEncoding:
    scene_tokens: torch.Tensor
    scene_stage_tokens: list[torch.Tensor]
    scene_decoder_stage_tokens: list[torch.Tensor]
    support_tokens: torch.Tensor
    support_stage_tokens: list[torch.Tensor]
    support_raymaps: torch.Tensor
    support_image_size: tuple[int, int]
    support_intrinsics: torch.Tensor
    support_c2w: torch.Tensor
    support_rgb_patch_tokens: torch.Tensor | None = None
    support_rgb_skip: torch.Tensor | None = None


class DA3NVSModel(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 14,
        include_moment: bool = True,
        backbone: nn.Module | None = None,
        backbone_model_name: str = "da3-large",
        backbone_weights_path: str | None = None,
        backbone_trainable: bool = False,
        use_camera_token: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        decoder_hidden_dim: int | None = None,
        out_channels: int = 3,
        epipolar_patch_band_width: float = 1.0,
        use_epipolar_masking: bool = True,
        use_query_ray_skip: bool = False,
        use_support_rgb_skip: bool = False,
        use_raw_rgb_stage1: bool = False,
        skip_feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.include_moment = include_moment
        self.ray_channels = 9 if include_moment else 6
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.decoder_hidden_dim = decoder_hidden_dim
        self.out_channels = out_channels
        self.epipolar_patch_band_width = epipolar_patch_band_width
        self.use_epipolar_masking = use_epipolar_masking
        self.use_query_ray_skip = use_query_ray_skip
        self.use_support_rgb_skip = use_support_rgb_skip
        self.use_raw_rgb_stage1 = use_raw_rgb_stage1
        self.skip_feature_dim = skip_feature_dim

        self.backbone = backbone or DA3PatchBackbone(
            model_name=backbone_model_name,
            weights_path=backbone_weights_path,
            trainable=backbone_trainable,
            use_camera_token=use_camera_token,
            ref_view_strategy=ref_view_strategy,
        )
        backbone_patch_size = getattr(self.backbone, "patch_size", None)
        if backbone_patch_size is not None and backbone_patch_size != self.patch_size:
            raise ValueError(
                f"Backbone patch_size={backbone_patch_size} does not match model patch_size={self.patch_size}"
            )
        self.ray_encoder: RayMapEncoder | None = None
        self.rgb_patch_encoder: RGBPatchEncoder | None = None
        self.nvs_head: CrossAttentionNVSHead | None = None
        self.query_ray_skip_encoder: HalfResCNNEncoder | None = None
        self.support_rgb_skip_encoder: HalfResCNNEncoder | None = None

    @staticmethod
    def _normalize_backbone_outputs(
        backbone_outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        if isinstance(backbone_outputs, torch.Tensor):
            return [backbone_outputs]
        if isinstance(backbone_outputs, (list, tuple)):
            if not backbone_outputs:
                raise RuntimeError("Backbone returned an empty feature list")
            return list(backbone_outputs)
        raise TypeError(f"Unsupported backbone output type: {type(backbone_outputs)!r}")

    def _build_heads(self, embed_dim: int, device: torch.device) -> None:
        if self.ray_encoder is None:
            self.ray_encoder = RayMapEncoder(
                patch_size=self.patch_size,
                embed_dim=embed_dim,
                ray_channels=self.ray_channels,
            ).to(device)

        if self.use_raw_rgb_stage1 and self.rgb_patch_encoder is None:
            self.rgb_patch_encoder = RGBPatchEncoder(
                patch_size=self.patch_size,
                embed_dim=embed_dim,
            ).to(device)

        if self.use_query_ray_skip and self.query_ray_skip_encoder is None:
            self.query_ray_skip_encoder = HalfResCNNEncoder(
                in_channels=self.ray_channels,
                out_channels=self.skip_feature_dim,
            ).to(device)

        if self.use_support_rgb_skip and self.support_rgb_skip_encoder is None:
            self.support_rgb_skip_encoder = HalfResCNNEncoder(
                in_channels=3,
                out_channels=self.skip_feature_dim,
            ).to(device)

        if self.nvs_head is None:
            self.nvs_head = CrossAttentionNVSHead(
                embed_dim=embed_dim,
                patch_size=self.patch_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                decoder_hidden_dim=self.decoder_hidden_dim,
                out_channels=self.out_channels,
                query_skip_channels=self.skip_feature_dim if self.use_query_ray_skip else 0,
                support_skip_channels=self.skip_feature_dim if self.use_support_rgb_skip else 0,
            ).to(device)

    def _encode_support_rays(
        self,
        support_intrinsics: torch.Tensor,
        support_c2w: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, num_views = support_intrinsics.shape[:2]
        raymaps = raymap_from_cameras(
            support_intrinsics.reshape(batch_size * num_views, 3, 3),
            support_c2w.reshape(batch_size * num_views, *support_c2w.shape[-2:]),
            image_size[0],
            image_size[1],
            include_moment=self.include_moment,
        )
        return raymaps.reshape(batch_size, num_views, self.ray_channels, image_size[0], image_size[1])

    def _encode_query_rays(
        self,
        query_intrinsics: torch.Tensor,
        query_c2w: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, num_views = query_intrinsics.shape[:2]
        raymaps = raymap_from_cameras(
            query_intrinsics.reshape(batch_size * num_views, 3, 3),
            query_c2w.reshape(batch_size * num_views, *query_c2w.shape[-2:]),
            image_size[0],
            image_size[1],
            include_moment=self.include_moment,
        )
        return raymaps.reshape(batch_size, num_views, self.ray_channels, image_size[0], image_size[1])

    def encode_support(
        self,
        *,
        support_images: torch.Tensor,
        support_intrinsics: torch.Tensor,
        support_c2w: torch.Tensor,
        support_backbone_extrinsics: torch.Tensor | None = None,
    ) -> DA3SceneEncoding:
        if support_images.dim() != 5:
            raise ValueError("support_images must have shape (B, V, 3, H, W)")

        device = support_images.device
        support_intrinsics = support_intrinsics.to(device)
        support_c2w = support_c2w.to(device)
        if support_backbone_extrinsics is not None:
            support_backbone_extrinsics = support_backbone_extrinsics.to(device)

        support_image_size = (support_images.shape[-2], support_images.shape[-1])

        backbone_outputs = self.backbone(
            support_images,
            extrinsics=support_backbone_extrinsics,
            intrinsics=support_intrinsics,
        )
        support_stage_tokens = self._normalize_backbone_outputs(backbone_outputs)
        for stage_idx, stage_tokens in enumerate(support_stage_tokens):
            if stage_tokens.dim() != 4:
                raise RuntimeError(
                    f"Backbone stage {stage_idx} must return tokens with shape (B, V, P, D), "
                    f"got {stage_tokens.shape}"
                )

        support_tokens = support_stage_tokens[-1]
        embed_dim = support_tokens.shape[-1]
        self._build_heads(embed_dim, support_images.device)
        assert self.ray_encoder is not None
        assert self.nvs_head is not None

        support_raymaps = self._encode_support_rays(
            support_intrinsics,
            support_c2w,
            support_image_size,
        )

        batch_size, support_views = support_raymaps.shape[:2]
        support_ray_tokens = self.ray_encoder(
            support_raymaps.reshape(batch_size * support_views, self.ray_channels, *support_image_size)
        ).reshape(batch_size, support_views, -1, embed_dim)
        if support_tokens.shape[:3] != support_ray_tokens.shape[:3]:
            raise RuntimeError(
                "Backbone tokens and support ray tokens must share the same (batch, views, patches) layout. "
                f"Got backbone={support_tokens.shape[:3]} and rays={support_ray_tokens.shape[:3]}"
            )
        lifted_support_stage_tokens: list[torch.Tensor] = []
        for stage_idx, stage_tokens in enumerate(support_stage_tokens):
            if stage_tokens.shape[-1] != embed_dim:
                raise RuntimeError(
                    f"Backbone stage {stage_idx} has embed_dim={stage_tokens.shape[-1]}, but expected {embed_dim}. "
                    "Current ray lifting assumes all DA3 support stages share the same embedding width."
                )
            if stage_tokens.shape[:3] != support_ray_tokens.shape[:3]:
                raise RuntimeError(
                    f"Backbone stage {stage_idx} layout {stage_tokens.shape[:3]} does not match support ray token "
                    f"layout {support_ray_tokens.shape[:3]}"
                )
            lifted_support_stage_tokens.append(stage_tokens + support_ray_tokens)

        lifted_support_tokens = lifted_support_stage_tokens[-1]
        scene_stage_tokens = [
            stage_tokens.reshape(batch_size, -1, embed_dim)
            for stage_tokens in lifted_support_stage_tokens
        ]
        scene_tokens = scene_stage_tokens[-1]
        support_rgb_patch_tokens = None
        if self.rgb_patch_encoder is not None:
            support_rgb_patch_tokens = self.rgb_patch_encoder(
                support_images.reshape(batch_size * support_views, 3, *support_image_size)
            ).reshape(batch_size, support_views, -1, embed_dim)
        if self.use_raw_rgb_stage1:
            if support_rgb_patch_tokens is None:
                raise RuntimeError("RGB patch encoder should be initialized when use_raw_rgb_stage1=True")
            scene_decoder_stage_tokens = [
                support_rgb_patch_tokens.reshape(batch_size, -1, embed_dim),
                *scene_stage_tokens[-3:],
            ]
        else:
            scene_decoder_stage_tokens = scene_stage_tokens
        support_rgb_skip = None
        if self.support_rgb_skip_encoder is not None:
            support_rgb_skip = self.support_rgb_skip_encoder(
                support_images.reshape(batch_size * support_views, 3, *support_image_size)
            ).reshape(
                batch_size,
                support_views,
                self.skip_feature_dim,
                support_image_size[0] // 2,
                support_image_size[1] // 2,
            )

        return DA3SceneEncoding(
            scene_tokens=scene_tokens,
            scene_stage_tokens=scene_stage_tokens,
            scene_decoder_stage_tokens=scene_decoder_stage_tokens,
            support_tokens=lifted_support_tokens,
            support_stage_tokens=lifted_support_stage_tokens,
            support_rgb_patch_tokens=support_rgb_patch_tokens,
            support_raymaps=support_raymaps,
            support_image_size=support_image_size,
            support_intrinsics=support_intrinsics,
            support_c2w=support_c2w,
            support_rgb_skip=support_rgb_skip,
        )

    def render_queries(
        self,
        scene_encoding: DA3SceneEncoding,
        *,
        query_intrinsics: torch.Tensor,
        query_c2w: torch.Tensor,
        query_image_size: tuple[int, int] | None = None,
    ) -> DA3NVSOutputs:
        device = scene_encoding.scene_tokens.device
        query_intrinsics = query_intrinsics.to(device)
        query_c2w = query_c2w.to(device)
        query_image_size = query_image_size or scene_encoding.support_image_size
        assert self.ray_encoder is not None
        assert self.nvs_head is not None

        query_raymaps = self._encode_query_rays(
            query_intrinsics,
            query_c2w,
            query_image_size,
        )
        epipolar_mask = None
        if self.use_epipolar_masking:
            epipolar_mask = compute_epipolar_attention_mask(
                query_intrinsics,
                query_c2w,
                scene_encoding.support_intrinsics,
                scene_encoding.support_c2w,
                query_image_size=query_image_size,
                support_image_size=scene_encoding.support_image_size,
                patch_size=self.patch_size,
                patch_band_width=self.epipolar_patch_band_width,
            )
        embed_dim = scene_encoding.scene_tokens.shape[-1]

        batch_size, query_views = query_raymaps.shape[:2]
        query_tokens = self.ray_encoder(
            query_raymaps.reshape(batch_size * query_views, self.ray_channels, *query_image_size)
        ).reshape(batch_size, query_views, -1, embed_dim)
        query_ray_skip = None
        if self.query_ray_skip_encoder is not None:
            query_ray_skip = self.query_ray_skip_encoder(
                query_raymaps.reshape(batch_size * query_views, self.ray_channels, *query_image_size)
            ).reshape(
                batch_size,
                query_views,
                self.skip_feature_dim,
                query_image_size[0] // 2,
                query_image_size[1] // 2,
            )

        pred_rgb, rendered_tokens = self.nvs_head(
            query_tokens,
            scene_encoding.scene_decoder_stage_tokens,
            image_size=query_image_size,
            epipolar_mask=epipolar_mask,
            query_ray_skip=query_ray_skip,
            support_rgb_skip=scene_encoding.support_rgb_skip,
            support_image_size=scene_encoding.support_image_size,
        )

        return DA3NVSOutputs(
            pred_rgb=pred_rgb,
            scene_tokens=scene_encoding.scene_tokens,
            support_tokens=scene_encoding.support_tokens,
            query_tokens=query_tokens,
            rendered_tokens=rendered_tokens,
            support_raymaps=scene_encoding.support_raymaps,
            query_raymaps=query_raymaps,
            epipolar_mask=epipolar_mask,
        )

    def forward(
        self,
        *,
        support_images: torch.Tensor,
        support_intrinsics: torch.Tensor,
        support_c2w: torch.Tensor,
        query_intrinsics: torch.Tensor,
        query_c2w: torch.Tensor,
        query_image_size: tuple[int, int] | None = None,
        support_backbone_extrinsics: torch.Tensor | None = None,
    ) -> DA3NVSOutputs:
        scene_encoding = self.encode_support(
            support_images=support_images,
            support_intrinsics=support_intrinsics,
            support_c2w=support_c2w,
            support_backbone_extrinsics=support_backbone_extrinsics,
        )
        return self.render_queries(
            scene_encoding,
            query_intrinsics=query_intrinsics,
            query_c2w=query_c2w,
            query_image_size=query_image_size,
        )
