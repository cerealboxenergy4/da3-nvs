from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import tqdm

from da3_nvs.data import NerfSyntheticSceneDataset, scene_batch_collate
from da3_nvs.models import DA3NVSModel
from da3_nvs.models.common import patchify
from da3_nvs.train import Trainer
from da3_nvs.data.types import SceneBatch


class ToyPatchBackbone(nn.Module):
    def __init__(self, *, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        *,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del extrinsics, intrinsics
        batch_size, num_views, _, height, width = images.shape
        flat_images = images.reshape(batch_size * num_views, 3, height, width)
        tokens = self.norm(self.proj(patchify(flat_images, self.patch_size)))
        return tokens.reshape(batch_size, num_views, tokens.shape[1], tokens.shape[2])


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal da3-nvs mock training loop.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/hunn/projects/datasets/nerf_synthetic"),
        help="Path to the nerf_synthetic dataset root.",
    )
    parser.add_argument(
        "--backbone",
        choices=("toy", "da3"),
        default="toy",
        help="Use a cheap toy backbone for pipeline smoke tests, or the real frozen DA3 backbone.",
    )
    parser.add_argument(
        "--da3-model-name",
        default="da3-large",
        help="DA3 model name when --backbone=da3.",
    )
    parser.add_argument(
        "--da3-weights-path",
        default=None,
        help="Optional local DA3 weights path.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--support-views", type=int, default=64)
    parser.add_argument("--train-query-views", type=int, default=16)
    parser.add_argument("--eval-query-views", type=int, default=16)
    parser.add_argument(
        "--epipolar-masking",
        type=int,
        choices=(0, 1),
        default=1,
        help="Enable epipolar attention masking (1) or disable it (0).",
    )
    parser.add_argument(
        "--use-query-ray-skip",
        action="store_true",
        help="Fuse a shallow 1/2-resolution CNN feature from the query ray map into the highest-resolution DPT branch.",
    )
    parser.add_argument(
        "--use-support-rgb-skip",
        action="store_true",
        help="Fuse an attention-aggregated 1/2-resolution support RGB CNN feature into the highest-resolution DPT branch.",
    )
    parser.add_argument(
        "--use-raw-rgb-stage1",
        action="store_true",
        help="Replace the first DPT stage with query-aligned tokens obtained by attending to projected support RGB patches.",
    )
    parser.add_argument(
        "--skip-feature-dim",
        type=int,
        default=64,
        help="Channel dimension for the optional query-ray/support-RGB skip branches.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=("adamw", "adam", "sgd"),
        help="Optimizer to use for trainable query-side parameters.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.0,
        help="Optional total variation loss weight.",
    )
    parser.add_argument(
        "--lpips-weight",
        type=float,
        default=0.0,
        help="Optional LPIPS loss weight. Set to 0.1 to add the requested perceptual term.",
    )
    parser.add_argument(
        "--start-lpips",
        type=int,
        default=1,
        help="1-based step index from which LPIPS loss becomes active.",
    )
    parser.add_argument(
        "--lpips-net",
        default="vgg",
        choices=("alex", "vgg", "squeeze"),
        help="Backbone for LPIPS when --lpips-weight > 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for query sampling.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--support-refresh-period",
        type=int,
        default=1,
        help="Re-sample support views every K steps while still changing query views each step.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        default=None,
        help="Restrict to one or more nerf_synthetic scenes. Repeat the flag to add multiple scenes.",
    )
    parser.add_argument(
        "--toy-embed-dim",
        type=int,
        default=192,
        help="Embedding dimension for the toy backbone.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run unseen eval every N steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save loss plots and eval visualizations. Defaults to outputs/mock_train/<timestamp>.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a saved checkpoint.pt. --steps is interpreted as the final total step count.",
    )
    parser.add_argument(
        "--da3-log-level",
        default="WARN",
        choices=("ERROR", "WARN", "INFO", "DEBUG"),
        help="Depth Anything 3 internal log level.",
    )
    return parser


def build_model(args: argparse.Namespace) -> DA3NVSModel:
    if args.backbone == "toy":
        backbone = ToyPatchBackbone(patch_size=14, embed_dim=args.toy_embed_dim)
        return DA3NVSModel(
            patch_size=14,
            backbone=backbone,
            num_heads=4,
            decoder_hidden_dim=128,
            use_epipolar_masking=bool(args.epipolar_masking),
            use_query_ray_skip=args.use_query_ray_skip,
            use_support_rgb_skip=args.use_support_rgb_skip,
            use_raw_rgb_stage1=args.use_raw_rgb_stage1,
            skip_feature_dim=args.skip_feature_dim,
        )

    return DA3NVSModel(
        patch_size=14,
        backbone_model_name=args.da3_model_name,
        backbone_weights_path=args.da3_weights_path,
        backbone_trainable=False,
        num_heads=8,
        decoder_hidden_dim=128, # 256
        use_epipolar_masking=bool(args.epipolar_masking),
        use_query_ray_skip=args.use_query_ray_skip,
        use_support_rgb_skip=args.use_support_rgb_skip,
        use_raw_rgb_stage1=args.use_raw_rgb_stage1,
        skip_feature_dim=args.skip_feature_dim,
    )


def describe_support_stages(model: DA3NVSModel) -> str:
    stage_getter = getattr(model.backbone, "get_stage_indices", None)
    if callable(stage_getter):
        stage_indices = stage_getter()
        if stage_indices is not None:
            return str(stage_indices)
    return "unavailable"


def describe_decoder_stages(args: argparse.Namespace, model: DA3NVSModel) -> str:
    stage_getter = getattr(model.backbone, "get_stage_indices", None)
    stage_indices = stage_getter() if callable(stage_getter) else None
    if stage_indices is None:
        return "unavailable"

    if args.use_raw_rgb_stage1:
        if len(stage_indices) >= 4:
            decoder_stages = ["rgb", *stage_indices[-3:]]
        else:
            decoder_stages = ["rgb", *stage_indices]
    else:
        decoder_stages = stage_indices
    return str(decoder_stages)


def build_optimizer(args: argparse.Namespace, model: DA3NVSModel) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adam":
        return torch.optim.Adam(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def move_batch(batch: SceneBatch, device: str) -> SceneBatch:
    return SceneBatch(
        support_images=batch.support_images.to(device),
        support_intrinsics=batch.support_intrinsics.to(device),
        support_c2w=batch.support_c2w.to(device),
        train_query_intrinsics=batch.train_query_intrinsics.to(device),
        train_query_c2w=batch.train_query_c2w.to(device),
        train_target_rgb=batch.train_target_rgb.to(device),
        eval_query_intrinsics=batch.eval_query_intrinsics.to(device),
        eval_query_c2w=batch.eval_query_c2w.to(device),
        eval_target_rgb=batch.eval_target_rgb.to(device),
    )


def replace_support(current_batch: SceneBatch, support_batch: SceneBatch) -> SceneBatch:
    return SceneBatch(
        support_images=support_batch.support_images,
        support_intrinsics=support_batch.support_intrinsics,
        support_c2w=support_batch.support_c2w,
        train_query_intrinsics=current_batch.train_query_intrinsics,
        train_query_c2w=current_batch.train_query_c2w,
        train_target_rgb=current_batch.train_target_rgb,
        eval_query_intrinsics=current_batch.eval_query_intrinsics,
        eval_query_c2w=current_batch.eval_query_c2w,
        eval_target_rgb=current_batch.eval_target_rgb,
    )


@torch.no_grad()
def materialize_trainable_modules(
    model: DA3NVSModel,
    batch: SceneBatch,
    *,
    device: str,
) -> None:
    batch = move_batch(batch, device)
    model = model.to(device)
    model.eval()
    model(
        support_images=batch.support_images,
        support_intrinsics=batch.support_intrinsics,
        support_c2w=batch.support_c2w,
        query_intrinsics=batch.train_query_intrinsics,
        query_c2w=batch.train_query_c2w,
        query_image_size=batch.train_target_rgb.shape[-2:],
    )


def build_step_batch(
    dataset: NerfSyntheticSceneDataset,
    *,
    step_idx: int,
    batch_size: int,
    support_refresh_period: int,
    base_seed: int,
) -> SceneBatch:
    num_scenes = len(dataset.support_scenes)
    group_idx = step_idx // support_refresh_period
    support_rng = random.Random(base_seed + (group_idx * 10_003))

    if num_scenes == 1:
        scene_slots = [0] * batch_size
    elif batch_size <= num_scenes:
        scene_slots = support_rng.sample(range(num_scenes), k=batch_size)
    else:
        scene_slots = [support_rng.randrange(num_scenes) for _ in range(batch_size)]

    support_samples = []
    current_samples = []
    for batch_idx in range(batch_size):
        scene_slot = scene_slots[batch_idx]
        anchor_offset = group_idx * support_refresh_period

        support_index = scene_slot + (anchor_offset * num_scenes)
        query_rng = random.Random(base_seed + (step_idx * 1_000_003) + (batch_idx * 97) + 17)
        current_cycle = query_rng.randrange(max(dataset.size, 1))
        current_index = scene_slot + (current_cycle * num_scenes)

        support_samples.append(dataset[support_index])
        current_samples.append(dataset[current_index])

    support_batch = scene_batch_collate(support_samples)
    current_batch = scene_batch_collate(current_samples)
    return replace_support(current_batch, support_batch)


@torch.no_grad()
def predict_query_batch(
    model: DA3NVSModel,
    batch: SceneBatch,
    *,
    device: str,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = move_batch(batch, device)
    model.eval()
    scene_encoding = model.encode_support(
        support_images=batch.support_images,
        support_intrinsics=batch.support_intrinsics,
        support_c2w=batch.support_c2w,
    )

    if split == "eval":
        query_intrinsics = batch.eval_query_intrinsics
        query_c2w = batch.eval_query_c2w
        target_rgb = batch.eval_target_rgb
    elif split == "train":
        query_intrinsics = batch.train_query_intrinsics
        query_c2w = batch.train_query_c2w
        target_rgb = batch.train_target_rgb
    else:
        raise ValueError(f"Unsupported split: {split}")

    outputs = model.render_queries(
        scene_encoding,
        query_intrinsics=query_intrinsics,
        query_c2w=query_c2w,
        query_image_size=target_rgb.shape[-2:],
    )
    return outputs.pred_rgb.detach().cpu(), target_rgb.detach().cpu()


def resolve_output_dir(requested_output_dir: Path | None) -> Path:
    if requested_output_dir is not None:
        return requested_output_dir
    raise RuntimeError("resolve_output_dir now expects explicit scene/model metadata")


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


def infer_scene_label(args: argparse.Namespace) -> str:
    if args.scenes is None:
        return "all-scenes"
    if len(args.scenes) == 1:
        return slugify(args.scenes[0])
    return "multi-scenes"


def infer_model_alias(args: argparse.Namespace) -> str:
    if args.backbone == "toy":
        return "toy"

    name = args.da3_model_name.lower()
    size_alias_map = (
        ("giant", "DA3-giant"),
        ("large", "DA3-large"),
        ("base", "DA3-base"),
        ("small", "DA3-small"),
    )
    for needle, alias in size_alias_map:
        if needle in name:
            return alias
    return "DA3-unknown"


def build_default_output_dir(args: argparse.Namespace) -> Path:
    base_dir = Path("/home/hunn/projects/da3-nvs/outputs/mock_train")
    stem = f"{infer_scene_label(args)}_{infer_model_alias(args)}_{args.steps}"
    candidate = base_dir / stem
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        next_candidate = base_dir / f"{stem}_run{suffix}"
        if not next_candidate.exists():
            return next_candidate
        suffix += 1


def save_loss_plot(
    train_history: list[dict[str, float]],
    eval_history: list[dict[str, float]],
    output_path: Path,
    *,
    support_refresh_period: int,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [item["step"] for item in train_history],
        [item["train_rgb_recon_loss"] for item in train_history],
        marker="o",
        linewidth=2,
        label="train_rgb_recon_loss",
    )
    if eval_history:
        axis.plot(
            [item["step"] for item in eval_history],
            [item["unseen_recon_loss"] for item in eval_history],
            marker="s",
            linewidth=2,
            label="unseen_recon_loss",
        )
    if support_refresh_period > 0:
        max_step = int(max([item["step"] for item in train_history], default=0))
        refresh_steps = range(support_refresh_period + 1, max_step + 1, support_refresh_period)
        for refresh_idx, refresh_step in enumerate(refresh_steps):
            axis.axvline(
                refresh_step,
                linestyle=":",
                linewidth=1.2,
                color="gray",
                alpha=0.8,
                label="support refresh" if refresh_idx == 0 else None,
            )
    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")
    axis.set_title("mock_train loss history")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def select_view_indices(num_views: int, max_views: int) -> list[int]:
    if num_views <= max_views:
        return list(range(num_views))
    return torch.linspace(0, num_views - 1, steps=max_views).round().long().tolist()


def save_combined_visualizations(
    train_pred_rgb: torch.Tensor,
    train_gt_rgb: torch.Tensor,
    eval_pred_rgb: torch.Tensor,
    eval_gt_rgb: torch.Tensor,
    output_dir: Path,
    *,
    support_views: int,
    train_query_views: int,
    support_refresh_period: int,
    max_views: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_diff_rgb = (train_pred_rgb - train_gt_rgb).abs().clamp(0.0, 1.0)
    eval_diff_rgb = (eval_pred_rgb - eval_gt_rgb).abs().clamp(0.0, 1.0)

    batch_size = min(train_pred_rgb.shape[0], eval_pred_rgb.shape[0])
    for batch_idx in range(batch_size):
        train_indices = select_view_indices(train_pred_rgb.shape[1], max_views)
        eval_indices = select_view_indices(eval_pred_rgb.shape[1], max_views)
        num_rows = max(len(train_indices), len(eval_indices))

        figure, axes = plt.subplots(num_rows, 6, figsize=(18, max(num_rows, 1) * 3))
        if num_rows == 1:
            axes = axes[None, :]

        for row_idx in range(num_rows):
            if row_idx < len(train_indices):
                view_idx = train_indices[row_idx]
                train_images = (
                    train_gt_rgb[batch_idx, view_idx],
                    train_pred_rgb[batch_idx, view_idx],
                    train_diff_rgb[batch_idx, view_idx],
                )
                train_titles = (
                    f"Train GT {view_idx}",
                    f"Train Pred {view_idx}",
                    f"Train Diff {view_idx}",
                )
                for col_idx, (image, title) in enumerate(zip(train_images, train_titles)):
                    axes[row_idx, col_idx].imshow(image.permute(1, 2, 0).numpy())
                    axes[row_idx, col_idx].set_title(title)
                    axes[row_idx, col_idx].axis("off")
            else:
                for col_idx in range(3):
                    axes[row_idx, col_idx].axis("off")

            if row_idx < len(eval_indices):
                view_idx = eval_indices[row_idx]
                eval_images = (
                    eval_gt_rgb[batch_idx, view_idx],
                    eval_pred_rgb[batch_idx, view_idx],
                    eval_diff_rgb[batch_idx, view_idx],
                )
                eval_titles = (
                    f"Eval GT {view_idx}",
                    f"Eval Pred {view_idx}",
                    f"Eval Diff {view_idx}",
                )
                for offset, (image, title) in enumerate(zip(eval_images, eval_titles), start=3):
                    axes[row_idx, offset].imshow(image.permute(1, 2, 0).numpy())
                    axes[row_idx, offset].set_title(title)
                    axes[row_idx, offset].axis("off")
            else:
                for col_idx in range(3, 6):
                    axes[row_idx, col_idx].axis("off")

        figure.suptitle(
            " | ".join(
                [
                    f"support views: {support_views}",
                    f"train query views: {train_query_views}",
                    f"support refresh period: {support_refresh_period}",
                ]
            ),
            fontsize=14,
            y=0.995,
        )
        figure.tight_layout()
        figure.savefig(output_dir / f"combined_batch_{batch_idx:02d}.png", dpi=160, bbox_inches="tight")
        plt.close(figure)


def count_trainable_parameters(model: DA3NVSModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_trainable_state_dict(model: DA3NVSModel) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if not name.startswith("backbone.")
    }


def save_checkpoint(
    model: DA3NVSModel,
    optimizer: torch.optim.Optimizer,
    output_path: Path,
    *,
    args: argparse.Namespace,
    train_history: list[dict[str, float]],
    eval_history: list[dict[str, float]],
    elapsed_seconds: float,
    trainable_params: int,
) -> None:
    checkpoint = {
        "checkpoint_type": "query-side-only",
        "args": vars(args),
        "trainable_params": trainable_params,
        "elapsed_seconds": elapsed_seconds,
        "train_history": train_history,
        "eval_history": eval_history,
        "model_state_dict": build_trainable_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, output_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: DA3NVSModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[list[dict[str, float]], list[dict[str, float]], float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint.get("model_state_dict", {})
    current_state = model.state_dict()
    unexpected = [name for name in checkpoint_state.keys() if name not in current_state]
    if unexpected:
        raise RuntimeError(
            "Checkpoint contains unexpected query-side keys: "
            f"{unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )

    current_state.update(checkpoint_state)
    model.load_state_dict(current_state, strict=False)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    train_history = list(checkpoint.get("train_history", []))
    eval_history = list(checkpoint.get("eval_history", []))
    elapsed_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
    return train_history, eval_history, elapsed_seconds


def main() -> None:
    args = build_argparser().parse_args()
    if args.support_refresh_period <= 0:
        raise ValueError("--support-refresh-period must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.start_lpips <= 0:
        raise ValueError("--start-lpips must be positive")
    if args.resume is not None and not args.resume.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
    os.environ["DA3_LOG_LEVEL"] = args.da3_log_level
    if args.resume is not None and args.output_dir is None:
        output_dir = args.resume.resolve().parent
    else:
        output_dir = args.output_dir if args.output_dir is not None else build_default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    start_time = time.perf_counter()

    dataset = NerfSyntheticSceneDataset(
        root=args.root,
        scene_names=tuple(args.scenes) if args.scenes else None,
        image_size=args.image_size,
        support_views=args.support_views,
        train_query_views=args.train_query_views,
        eval_query_views=args.eval_query_views,
        size=max(args.steps * args.batch_size, 8),
    )

    model = build_model(args)
    first_batch = build_step_batch(
        dataset,
        step_idx=0,
        batch_size=args.batch_size,
        support_refresh_period=args.support_refresh_period,
        base_seed=args.seed,
    )
    materialize_trainable_modules(model, first_batch, device=args.device)
    optimizer = build_optimizer(args, model)
    trainer = Trainer(
        model,
        optimizer,
        device=args.device,
        lpips_weight=args.lpips_weight,
        lpips_net=args.lpips_net,
        tv_weight=args.tv_weight,
    )
    trainable_params = count_trainable_parameters(model)
    support_stage_desc = describe_support_stages(model)
    decoder_stage_desc = describe_decoder_stages(args, model)
    train_history: list[dict[str, float]] = []
    eval_history: list[dict[str, float]] = []
    elapsed_offset = 0.0
    completed_steps = 0
    last_batch = first_batch
    if args.resume is not None:
        train_history, eval_history, elapsed_offset = load_checkpoint(args.resume, model, optimizer)
        completed_steps = int(train_history[-1]["step"]) if train_history else 0
        reference_step = max(completed_steps - 1, 0)
        last_batch = build_step_batch(
            dataset,
            step_idx=reference_step,
            batch_size=args.batch_size,
            support_refresh_period=args.support_refresh_period,
            base_seed=args.seed,
        )

    print(
        f"mock_train start | backbone={args.backbone} device={args.device} "
        f"image={args.image_size} support={args.support_views} "
        f"train_query={args.train_query_views} eval_query={args.eval_query_views} "
        f"support_refresh={args.support_refresh_period} output_dir={output_dir} "
        f"trainable_params={trainable_params} optimizer={args.optimizer} lr={args.lr} "
        f"lpips_weight={args.lpips_weight} start_lpips={args.start_lpips} "
        f"tv_weight={args.tv_weight} "
        f"epipolar_masking={args.epipolar_masking} "
        f"query_ray_skip={args.use_query_ray_skip} "
        f"support_rgb_skip={args.use_support_rgb_skip} "
        f"raw_rgb_stage1={args.use_raw_rgb_stage1}"
    )
    print(f"using DA3 support stages: {support_stage_desc}")
    print(f"using decoder stages: {decoder_stage_desc}")
    if args.resume is not None:
        print(f"resuming from {args.resume} at completed_step={completed_steps}")

    progress = tqdm(
        range(completed_steps + 1, args.steps + 1),
        desc="train",
        dynamic_ncols=True,
        leave=True,
        initial=completed_steps,
        total=args.steps,
    )
    latest_train_loss: float | None = (
        float(train_history[-1]["train_rgb_recon_loss"]) if train_history else None
    )
    latest_eval_loss: float | None = (
        float(eval_history[-1]["unseen_recon_loss"]) if eval_history else None
    )
    latest_train_lpips: float | None = (
        float(train_history[-1]["train_lpips_loss"]) if train_history and "train_lpips_loss" in train_history[-1] else None
    )
    latest_eval_lpips: float | None = (
        float(eval_history[-1]["eval_lpips_loss"]) if eval_history and "eval_lpips_loss" in eval_history[-1] else None
    )
    latest_train_tv: float | None = (
        float(train_history[-1]["train_tv_loss"]) if train_history and "train_tv_loss" in train_history[-1] else None
    )
    latest_eval_tv: float | None = (
        float(eval_history[-1]["eval_tv_loss"]) if eval_history and "eval_tv_loss" in eval_history[-1] else None
    )
    latest_lpips_weight: float = 0.0
    for step in progress:
        batch = (
            first_batch
            if step == 1 and completed_steps == 0
            else build_step_batch(
                dataset,
                step_idx=step - 1,
                batch_size=args.batch_size,
                support_refresh_period=args.support_refresh_period,
                base_seed=args.seed,
            )
        )
        last_batch = batch
        latest_lpips_weight = args.lpips_weight if step >= args.start_lpips else 0.0
        trainer.set_lpips_weight(latest_lpips_weight)

        metrics = trainer.train_step(batch)
        train_history.append(
            {
                "step": float(step),
                "train_rgb_recon_loss": metrics["train_rgb_recon_loss"],
                "train_epipolar_keep_ratio": metrics["train_epipolar_keep_ratio"],
                "lpips_weight": latest_lpips_weight,
            }
        )
        if "train_lpips_loss" in metrics:
            train_history[-1]["train_lpips_loss"] = metrics["train_lpips_loss"]
        if "train_tv_loss" in metrics:
            train_history[-1]["train_tv_loss"] = metrics["train_tv_loss"]
        latest_train_loss = metrics["train_rgb_recon_loss"]
        latest_train_lpips = metrics.get("train_lpips_loss")
        latest_train_tv = metrics.get("train_tv_loss")

        if step % max(args.eval_every, 1) == 0:
            eval_metrics = trainer.evaluate_unseen_metrics(batch)
            eval_history.append(
                {
                    "step": float(step),
                    "unseen_recon_loss": eval_metrics["unseen_recon_loss"],
                    "eval_epipolar_keep_ratio": eval_metrics["eval_epipolar_keep_ratio"],
                    "lpips_weight": latest_lpips_weight,
                }
            )
            if "eval_lpips_loss" in eval_metrics:
                eval_history[-1]["eval_lpips_loss"] = eval_metrics["eval_lpips_loss"]
            if "eval_tv_loss" in eval_metrics:
                eval_history[-1]["eval_tv_loss"] = eval_metrics["eval_tv_loss"]
            latest_eval_loss = eval_metrics["unseen_recon_loss"]
            latest_eval_lpips = eval_metrics.get("eval_lpips_loss")
            latest_eval_tv = eval_metrics.get("eval_tv_loss")

        current_lr = float(optimizer.param_groups[0]["lr"])
        postfix = {
            "lr": f"{current_lr:.2e}",
            "train_loss": f"{latest_train_loss:.6f}" if latest_train_loss is not None else "-",
            "eval_loss": f"{latest_eval_loss:.6f}" if latest_eval_loss is not None else "-",
            "lpips_w": f"{latest_lpips_weight:.3f}",
            "train_lpips": f"{latest_train_lpips:.4f}" if latest_train_lpips is not None else "-",
            "eval_lpips": f"{latest_eval_lpips:.4f}" if latest_eval_lpips is not None else "-",
        }
        if args.tv_weight > 0.0:
            postfix["train_tv"] = f"{latest_train_tv:.4f}" if latest_train_tv is not None else "-"
            postfix["eval_tv"] = f"{latest_eval_tv:.4f}" if latest_eval_tv is not None else "-"
        progress.set_postfix(postfix)

    eval_pred_rgb, eval_gt_rgb = predict_query_batch(model, last_batch, device=args.device, split="eval")
    train_pred_rgb, train_gt_rgb = predict_query_batch(model, last_batch, device=args.device, split="train")
    save_combined_visualizations(
        train_pred_rgb,
        train_gt_rgb,
        eval_pred_rgb,
        eval_gt_rgb,
        output_dir,
        support_views=last_batch.support_images.shape[1],
        train_query_views=last_batch.train_target_rgb.shape[1],
        support_refresh_period=args.support_refresh_period,
        max_views=4,
    )
    save_loss_plot(
        train_history,
        eval_history,
        output_dir / "loss_curve.png",
        support_refresh_period=args.support_refresh_period,
    )

    total_elapsed = elapsed_offset + (time.perf_counter() - start_time)
    save_checkpoint(
        model,
        optimizer,
        checkpoint_path,
        args=args,
        train_history=train_history,
        eval_history=eval_history,
        elapsed_seconds=total_elapsed,
        trainable_params=trainable_params,
    )

    metrics_payload = {
        "train_history": train_history,
        "eval_history": eval_history,
        "output_dir": str(output_dir),
        "combined_visualization_path": str(output_dir / "combined_batch_00.png"),
        "checkpoint_path": str(checkpoint_path),
        "trainable_params": trainable_params,
        "optimizer": args.optimizer,
        "resume_path": str(args.resume) if args.resume is not None else None,
        "completed_steps_before_resume": completed_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "tv_weight": args.tv_weight,
        "epipolar_masking": bool(args.epipolar_masking),
        "lpips_weight": args.lpips_weight,
        "start_lpips": args.start_lpips,
        "lpips_net": args.lpips_net,
        "elapsed_seconds": total_elapsed,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    print(f"training finished in {total_elapsed:.2f}s")
    print(f"saved checkpoint to {checkpoint_path}")
    print(f"saved loss plot to {output_dir / 'loss_curve.png'}")
    print(f"saved combined visualization to {output_dir / 'combined_batch_00.png'}")
    print(f"saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
