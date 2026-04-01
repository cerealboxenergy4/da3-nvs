from __future__ import annotations

import torch
import torch.nn.functional as F

from da3_nvs.data import SceneBatch


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: str = "cpu",
        lpips_weight: float = 0.0,
        lpips_net: str = "alex",
        tv_weight: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.lpips_net = lpips_net
        self.lpips_loss_fn: torch.nn.Module | None = None
        self.lpips_weight = 0.0
        self.tv_weight = tv_weight

        if self.tv_weight < 0.0:
            raise ValueError("tv_weight must be non-negative")

        self.set_lpips_weight(lpips_weight)

    def set_lpips_weight(self, lpips_weight: float) -> None:
        if lpips_weight < 0.0:
            raise ValueError("lpips_weight must be non-negative")
        if lpips_weight > 0.0 and self.lpips_loss_fn is None:
            try:
                import lpips
            except ImportError as error:
                raise RuntimeError(
                    "LPIPS loss requested but the `lpips` package is not installed. "
                    "Install it first or run with --lpips-weight 0."
                ) from error

            self.lpips_loss_fn = lpips.LPIPS(net=self.lpips_net).to(self.device)
            self.lpips_loss_fn.eval()
            self.lpips_loss_fn.requires_grad_(False)
        self.lpips_weight = lpips_weight

    def _ensure_optimizer_tracks_all_params(self) -> None:
        tracked_param_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group["params"]
        }
        missing_params = [
            param
            for param in self.model.parameters()
            if param.requires_grad and id(param) not in tracked_param_ids
        ]
        if missing_params:
            self.optimizer.add_param_group({"params": missing_params})

    def _move_batch(self, batch: SceneBatch) -> SceneBatch:
        return SceneBatch(
            support_images=batch.support_images.to(self.device),
            support_intrinsics=batch.support_intrinsics.to(self.device),
            support_c2w=batch.support_c2w.to(self.device),
            train_query_intrinsics=batch.train_query_intrinsics.to(self.device),
            train_query_c2w=batch.train_query_c2w.to(self.device),
            train_target_rgb=batch.train_target_rgb.to(self.device),
            eval_query_intrinsics=batch.eval_query_intrinsics.to(self.device),
            eval_query_c2w=batch.eval_query_c2w.to(self.device),
            eval_target_rgb=batch.eval_target_rgb.to(self.device),
        )

    @staticmethod
    def _epipolar_keep_ratio(outputs) -> float:
        mask = getattr(outputs, "epipolar_mask", None)
        if mask is None:
            return 1.0
        return float(mask.float().mean().item())

    def _compute_lpips_loss(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        if self.lpips_loss_fn is None:
            return pred_rgb.new_zeros(())

        batch_size, num_views = pred_rgb.shape[:2]
        pred_flat = pred_rgb.reshape(batch_size * num_views, *pred_rgb.shape[2:]).float()
        target_flat = target_rgb.reshape(batch_size * num_views, *target_rgb.shape[2:]).float()
        pred_flat = pred_flat * 2.0 - 1.0
        target_flat = target_flat * 2.0 - 1.0
        return self.lpips_loss_fn(pred_flat, target_flat).mean()

    @staticmethod
    def _compute_tv_loss(pred_rgb: torch.Tensor) -> torch.Tensor:
        tv_h = torch.abs(pred_rgb[..., 1:, :] - pred_rgb[..., :-1, :]).mean()
        tv_w = torch.abs(pred_rgb[..., :, 1:] - pred_rgb[..., :, :-1]).mean()
        return tv_h + tv_w

    def train_step(self, batch: SceneBatch) -> dict[str, float]:
        batch = self._move_batch(batch)
        self.model.train()

        scene_encoding = self.model.encode_support(
            support_images=batch.support_images,
            support_intrinsics=batch.support_intrinsics,
            support_c2w=batch.support_c2w,
        )
        outputs = self.model.render_queries(
            scene_encoding,
            query_intrinsics=batch.train_query_intrinsics,
            query_c2w=batch.train_query_c2w,
            query_image_size=batch.train_target_rgb.shape[-2:],
        )
        self._ensure_optimizer_tracks_all_params()
        recon_loss = F.mse_loss(outputs.pred_rgb, batch.train_target_rgb)
        lpips_loss = self._compute_lpips_loss(outputs.pred_rgb, batch.train_target_rgb)
        tv_loss = self._compute_tv_loss(outputs.pred_rgb)
        total_loss = recon_loss + (self.lpips_weight * lpips_loss) + (self.tv_weight * tv_loss)

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        metrics = {
            "loss": float(total_loss.detach().item()),
            "train_rgb_recon_loss": float(recon_loss.detach().item()),
            "train_epipolar_keep_ratio": self._epipolar_keep_ratio(outputs),
        }
        if self.lpips_weight > 0.0:
            metrics["train_lpips_loss"] = float(lpips_loss.detach().item())
        if self.tv_weight > 0.0:
            metrics["train_tv_loss"] = float(tv_loss.detach().item())
        return metrics

    @torch.no_grad()
    def evaluate_unseen_metrics(self, batch: SceneBatch) -> dict[str, float]:
        batch = self._move_batch(batch)
        self.model.eval()

        scene_encoding = self.model.encode_support(
            support_images=batch.support_images,
            support_intrinsics=batch.support_intrinsics,
            support_c2w=batch.support_c2w,
        )
        outputs = self.model.render_queries(
            scene_encoding,
            query_intrinsics=batch.eval_query_intrinsics,
            query_c2w=batch.eval_query_c2w,
            query_image_size=batch.eval_target_rgb.shape[-2:],
        )
        unseen_recon = F.mse_loss(outputs.pred_rgb, batch.eval_target_rgb)
        metrics = {
            "unseen_recon_loss": float(unseen_recon.item()),
            "eval_epipolar_keep_ratio": self._epipolar_keep_ratio(outputs),
        }
        if self.lpips_weight > 0.0:
            eval_lpips_loss = self._compute_lpips_loss(outputs.pred_rgb, batch.eval_target_rgb)
            metrics["eval_lpips_loss"] = float(eval_lpips_loss.item())
        if self.tv_weight > 0.0:
            eval_tv_loss = self._compute_tv_loss(outputs.pred_rgb)
            metrics["eval_tv_loss"] = float(eval_tv_loss.item())
        return metrics
