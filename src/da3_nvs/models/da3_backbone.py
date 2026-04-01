from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


def _add_local_da3_repo_to_path() -> None:
    candidate = Path(__file__).resolve().parents[4] / "Depth-Anything-3" / "src"
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


class DA3PatchBackbone(nn.Module):
    PATCH_SIZE = 14

    def __init__(
        self,
        *,
        model_name: str = "da3-large",
        weights_path: str | None = None,
        trainable: bool = False,
        feature_index: int = -1,
        return_all_features: bool = True,
        use_camera_token: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> None:
        super().__init__()
        self.patch_size = self.PATCH_SIZE
        self.model_name = model_name
        self.weights_path = weights_path
        self.trainable = trainable
        self.feature_index = feature_index
        self.return_all_features = return_all_features
        self.use_camera_token = use_camera_token
        self.ref_view_strategy = ref_view_strategy

        self.model: nn.Module | None = None
        self.feature_net: nn.Module | None = None

    def get_stage_indices(self) -> list[int] | None:
        self._ensure_loaded()
        assert self.feature_net is not None
        out_layers = getattr(self.feature_net.backbone, "out_layers", None)
        if out_layers is None:
            return None
        return list(out_layers)

    def _build_model(self) -> nn.Module:
        _add_local_da3_repo_to_path()
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as error:
            raise RuntimeError(
                "Failed to import depth_anything_3. Install the package or place "
                "Depth-Anything-3/src next to this repository."
            ) from error

        if "/" in self.model_name:
            return DepthAnything3.from_pretrained(self.model_name)
        return DepthAnything3(model_name=self.model_name)

    def train(self, mode: bool = True) -> "DA3PatchBackbone":
        super().train(mode)
        if not self.trainable:
            if self.model is not None:
                self.model.eval()
            if self.feature_net is not None:
                self.feature_net.eval()
        return self

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.feature_net is not None:
            return

        model = self._build_model()
        if self.weights_path is not None:
            self._load_weights(model, self.weights_path)

        if not self.trainable:
            model.requires_grad_(False)
            model.eval()

        self.model = model
        inner_model = model.model
        self.feature_net = inner_model.anyview if hasattr(inner_model, "anyview") else inner_model

    def _load_weights(self, model: nn.Module, weights_path: str) -> None:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"DA3 weights not found: {weights_path}")

        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as error:
                raise RuntimeError(
                    "safetensors is required to load .safetensors DA3 weights."
                ) from error
            state_dict = load_file(str(path))
        else:
            state_dict = torch.load(str(path), map_location="cpu")

        model.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        images: torch.Tensor,
        *,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if images.dim() != 5:
            raise ValueError("images must have shape (B, V, 3, H, W)")
        if images.shape[-2] % self.PATCH_SIZE != 0 or images.shape[-1] % self.PATCH_SIZE != 0:
            raise ValueError(
                f"DA3 expects image sizes divisible by patch size {self.PATCH_SIZE}, "
                f"got {(images.shape[-2], images.shape[-1])}"
            )

        self._ensure_loaded()
        assert self.model is not None
        assert self.feature_net is not None

        if not self.trainable:
            self.model.eval()
            self.feature_net.eval()

        if next(self.model.parameters()).device != images.device:
            self.model = self.model.to(images.device)

        cam_token = None
        if self.use_camera_token:
            if extrinsics is None or intrinsics is None:
                raise ValueError("extrinsics and intrinsics are required when use_camera_token=True")
            with torch.autocast(device_type=images.device.type, enabled=False):
                cam_token = self.feature_net.cam_enc(extrinsics, intrinsics, images.shape[-2:])

        grad_context = torch.enable_grad() if self.trainable else torch.no_grad()
        with grad_context:
            feats, _ = self.feature_net.backbone(
                images,
                cam_token=cam_token,
                export_feat_layers=[],
                ref_view_strategy=self.ref_view_strategy,
            )

        if self.return_all_features:
            stage_tokens = [feat[0] for feat in feats]
            for stage_idx, tokens in enumerate(stage_tokens):
                if tokens.dim() != 4:
                    raise RuntimeError(
                        f"Expected DA3 patch tokens with 4 dimensions at stage {stage_idx}, got {tokens.shape}"
                    )
            return stage_tokens

        tokens = feats[self.feature_index][0]
        if tokens.dim() != 4:
            raise RuntimeError(f"Expected DA3 patch tokens with 4 dimensions, got {tokens.shape}")
        return tokens
