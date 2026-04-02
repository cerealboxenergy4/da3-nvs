from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from da3_nvs.models import DA3PatchBackbone
from da3_mask_experiment import (
    CO3DMaskReconstructionDataset,
    DA3MaskedPatchModel,
    MaskReconstructionBatch,
    MaskReconstructionDataset,
    ToyMultiStageBackbone,
    ToyPatchBackbone,
    compute_masked_patch_metrics,
    mask_batch_collate,
    move_mask_batch,
    patch_mask_to_image,
    sample_random_patch_mask,
)


def log_progress(message: str) -> None:
    print(message, flush=True)


def summarize_timing(total_seconds: float, count: int) -> dict[str, float]:
    return {
        "total_seconds": total_seconds,
        "count": float(count),
        "avg_seconds": (total_seconds / count) if count > 0 else 0.0,
    }


def build_lpips_loss_fn(device: str, net: str = "vgg") -> torch.nn.Module:
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "LPIPS loss requested but the `lpips` package is not installed."
        ) from error

    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()
    loss_fn.requires_grad_(False)
    return loss_fn


def compute_lpips_loss(
    loss_fn: torch.nn.Module,
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_views = pred_rgb.shape[:2]
    pred_flat = pred_rgb.reshape(batch_size * num_views, *pred_rgb.shape[2:]).float()
    target_flat = target_rgb.reshape(batch_size * num_views, *target_rgb.shape[2:]).float()
    pred_flat = pred_flat * 2.0 - 1.0
    target_flat = target_flat * 2.0 - 1.0
    return loss_fn(pred_flat, target_flat).mean()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DA3 masked-patch RGB reconstruction experiment.")
    parser.add_argument(
        "--dataset",
        choices=("nerf_synthetic", "co3d"),
        default="nerf_synthetic",
        help="Choose between the existing nerf_synthetic setup and a CO3D category loader.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default="/home/hunn/projects/datasets/nerf_synthetic",
        help="Dataset root. For CO3D, pass either the full root such as /home/hunn/projects/datasets/co3d_test or a single category folder such as /home/hunn/projects/datasets/co3d_test/apple.",
    )
    parser.add_argument(
        "--backbone",
        choices=("da3", "toy"),
        default="da3",
        help="Use the real frozen DA3 backbone or a toy backbone for smoke tests.",
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
    parser.add_argument(
        "--backbone-trainable",
        action="store_true",
        help="Allow gradients through the DA3 backbone instead of training only the downstream patch head.",
    )
    parser.add_argument(
        "--head-type",
        choices=("mlp", "cnn", "dpt"),
        default="dpt",
        help="Use the patch-wise MLP head, a lightweight CNN decoder, or a DPT decoder driven by stage-wise cross-attention.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--support-views", type=int, default=16)
    parser.add_argument("--train-query-views", type=int, default=4)
    parser.add_argument("--eval-query-views", type=int, default=4)
    parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default="val",
        help="Held-out viewpoints used for evaluation.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.4,
        help="Fraction of 14x14 patches to mask in each query image.",
    )
    parser.add_argument(
        "--mask-fill-value",
        type=float,
        default=0.0,
        help="Pixel value used to replace masked patches before encoding.",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=0.01,
        help="Treat nearly pure white/black patches within this threshold as background and exclude them from mask sampling.",
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=512,
        help="Hidden width of the masked-patch reconstruction MLP head.",
    )
    parser.add_argument(
        "--cross-attn-heads",
        type=int,
        default=8,
        help="Attention head count used by the DPT cross-attention head.",
    )
    parser.add_argument(
        "--cross-attn-mlp-ratio",
        type=float,
        default=4.0,
        help="Feed-forward expansion ratio used by the DPT cross-attention head.",
    )
    parser.add_argument(
        "--decoder-hidden-dim",
        type=int,
        default=128,
        help="Base hidden width used by the DPT decoder path.",
    )
    parser.add_argument(
        "--toy-embed-dim",
        type=int,
        default=192,
        help="Embedding dimension for the toy backbone.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=("adamw", "adam", "sgd"),
        help="Optimizer for trainable parameters.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--lpips-weight",
        type=float,
        default=0.1,
        help="Weight for LPIPS in the total loss: L_rgb + lpips_weight * L_lpips.",
    )
    parser.add_argument(
        "--lpips-net",
        default="vgg",
        choices=("alex", "vgg", "squeeze"),
        help="Backbone used by LPIPS.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        default=None,
        help="Restrict to nerf_synthetic scenes or CO3D category names / category-sequence ids like apple/110_13051_23361. Repeat the flag to add multiple entries.",
    )
    parser.add_argument(
        "--co3d-set-list",
        default="set_lists_manyview_dev_0.json",
        help="Preferred CO3D set_lists filename. Manyview files are used per-sequence when available; fewview-only categories fall back to sequence-wise random splitting over frames referenced by their fewview jsons.",
    )
    parser.add_argument(
        "--co3d-fallback-eval-ratio",
        type=float,
        default=0.1,
        help="Eval ratio used when a CO3D scene must be split from available frames instead of using a direct manyview train/eval split.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Run held-out masked reconstruction eval every N steps.",
    )
    parser.add_argument(
        "--max-vis-views",
        type=int,
        default=4,
        help="Maximum number of query views to show in saved visualizations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save checkpoints, metrics, plots, and visualizations.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a saved checkpoint.pt. --steps is interpreted as the final total step count.",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save checkpoint.pt at the end of training.",
    )
    parser.add_argument(
        "--da3-log-level",
        default="WARN",
        choices=("ERROR", "WARN", "INFO", "DEBUG"),
        help="Depth Anything 3 internal log level.",
    )
    return parser


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


def infer_scene_label(args: argparse.Namespace) -> str:
    if args.dataset == "co3d" and args.scenes is None:
        return slugify(Path(args.root).name)
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
    base_dir = REPO_ROOT / "outputs" / "da3_mask"
    ratio_tag = f"mr{int(round(args.mask_ratio * 100)):02d}"
    dataset_tag = "co3d" if args.dataset == "co3d" else "nerf"
    stem = f"{dataset_tag}_{infer_scene_label(args)}_{infer_model_alias(args)}_{args.head_type}_{args.steps}_{ratio_tag}"
    candidate = base_dir / stem
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        next_candidate = base_dir / f"{stem}_run{suffix}"
        if not next_candidate.exists():
            return next_candidate
        suffix += 1


def build_model(args: argparse.Namespace) -> DA3MaskedPatchModel:
    if args.backbone == "toy":
        if args.head_type == "dpt":
            backbone = ToyMultiStageBackbone(patch_size=14, embed_dim=args.toy_embed_dim)
        else:
            backbone = ToyPatchBackbone(patch_size=14, embed_dim=args.toy_embed_dim)
        return DA3MaskedPatchModel(
            patch_size=14,
            backbone=backbone,
            head_type=args.head_type,
            head_hidden_dim=args.head_hidden_dim,
            num_heads=args.cross_attn_heads,
            mlp_ratio=args.cross_attn_mlp_ratio,
            decoder_hidden_dim=args.decoder_hidden_dim,
            mask_fill_value=args.mask_fill_value,
        )

    return DA3MaskedPatchModel(
        patch_size=14,
        backbone_model_name=args.da3_model_name,
        backbone_weights_path=args.da3_weights_path,
        backbone_trainable=args.backbone_trainable,
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
        num_heads=args.cross_attn_heads,
        mlp_ratio=args.cross_attn_mlp_ratio,
        decoder_hidden_dim=args.decoder_hidden_dim,
        mask_fill_value=args.mask_fill_value,
    )


def build_optimizer(
    args: argparse.Namespace,
    model: DA3MaskedPatchModel,
) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_step_batch(
    dataset: MaskReconstructionDataset | CO3DMaskReconstructionDataset,
    *,
    step_idx: int,
    batch_size: int,
    base_seed: int,
) -> MaskReconstructionBatch:
    num_scenes = len(dataset.scenes)
    if num_scenes == 1:
        scene_slots = [0] * batch_size
    else:
        scene_rng = random.Random(base_seed + (step_idx * 10_003))
        if batch_size <= num_scenes:
            scene_slots = scene_rng.sample(range(num_scenes), k=batch_size)
        else:
            scene_slots = [scene_rng.randrange(num_scenes) for _ in range(batch_size)]

    samples = []
    for batch_idx in range(batch_size):
        scene_slot = scene_slots[batch_idx]
        cycle_rng = random.Random(base_seed + (step_idx * 1_000_003) + (batch_idx * 97) + 17)
        cycle = cycle_rng.randrange(max(dataset.size, 1))
        index = scene_slot + (cycle * num_scenes)
        samples.append(dataset[index])
    return mask_batch_collate(samples)


def build_dataset(
    args: argparse.Namespace,
) -> MaskReconstructionDataset | CO3DMaskReconstructionDataset:
    dataset_size = max(args.steps * args.batch_size, 8)
    if args.dataset == "co3d":
        return CO3DMaskReconstructionDataset(
            root=args.root,
            sequence_names=tuple(args.scenes) if args.scenes else None,
            image_size=args.image_size,
            support_views=args.support_views,
            train_query_views=args.train_query_views,
            eval_query_views=args.eval_query_views,
            set_list_name=args.co3d_set_list,
            size=dataset_size,
            fallback_eval_ratio=args.co3d_fallback_eval_ratio,
        )

    return MaskReconstructionDataset(
        root=args.root,
        scene_names=tuple(args.scenes) if args.scenes else None,
        image_size=args.image_size,
        support_views=args.support_views,
        train_query_views=args.train_query_views,
        eval_query_views=args.eval_query_views,
        eval_split=args.eval_split,
        size=dataset_size,
    )


def materialize_trainable_modules(
    model: DA3MaskedPatchModel,
    batch: MaskReconstructionBatch,
    *,
    device: str,
    mask_ratio: float,
    seed: int,
    background_threshold: float,
) -> None:
    batch = move_mask_batch(batch, device)
    model = model.to(device)
    model.eval()
    query_images = batch.train_query_images
    patch_mask = sample_random_patch_mask(
        images=query_images,
        patch_size=model.patch_size,
        mask_ratio=mask_ratio,
        seed=seed,
        background_threshold=background_threshold,
    ).to(device)
    model(
        support_images=batch.support_images,
        query_images=query_images,
        query_patch_mask=patch_mask,
        query_intrinsics=batch.train_query_intrinsics,
        query_c2w=batch.train_query_c2w,
    )


def count_trainable_parameters(model: DA3MaskedPatchModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_checkpoint_state_dict(model: DA3MaskedPatchModel) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    if backbone_should_be_saved(model):
        return {name: tensor.detach().cpu() for name, tensor in state_dict.items()}
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if not name.startswith("backbone.")
    }


def backbone_should_be_saved(model: DA3MaskedPatchModel) -> bool:
    return not (
        isinstance(model.backbone, DA3PatchBackbone)
        and not model.backbone.trainable
    )


def save_checkpoint(
    model: DA3MaskedPatchModel,
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
        "checkpoint_type": "da3-mask",
        "args": vars(args),
        "trainable_params": trainable_params,
        "elapsed_seconds": elapsed_seconds,
        "train_history": train_history,
        "eval_history": eval_history,
        "model_state_dict": build_checkpoint_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, output_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: DA3MaskedPatchModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[list[dict[str, float]], list[dict[str, float]], float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint.get("model_state_dict", {})
    current_state = model.state_dict()
    unexpected = [name for name in checkpoint_state.keys() if name not in current_state]
    if unexpected:
        raise RuntimeError(
            "Checkpoint contains unexpected keys: "
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


def mse_to_psnr(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def run_masked_step(
    model: DA3MaskedPatchModel,
    batch: MaskReconstructionBatch,
    *,
    split: str,
    device: str,
    mask_ratio: float,
    mask_seed: int,
    background_threshold: float,
    lpips_loss_fn: torch.nn.Module | None = None,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = move_mask_batch(batch, device)
    if split == "train":
        query_images = batch.train_query_images
        query_intrinsics = batch.train_query_intrinsics
        query_c2w = batch.train_query_c2w
    elif split == "eval":
        query_images = batch.eval_query_images
        query_intrinsics = batch.eval_query_intrinsics
        query_c2w = batch.eval_query_c2w
    else:
        raise ValueError(f"Unsupported split: {split}")

    patch_mask = sample_random_patch_mask(
        images=query_images,
        patch_size=model.patch_size,
        mask_ratio=mask_ratio,
        seed=mask_seed,
        background_threshold=background_threshold,
    ).to(device)
    outputs = model(
        support_images=batch.support_images,
        query_images=query_images,
        query_patch_mask=patch_mask,
        query_intrinsics=query_intrinsics,
        query_c2w=query_c2w,
    )
    patch_metrics = compute_masked_patch_metrics(
        outputs.pred_patch_rgb,
        outputs.target_patch_rgb,
        outputs.query_patch_mask,
    )
    mse = patch_metrics["mse"]
    mae = patch_metrics["mae"]
    metrics = {
        f"{split}_masked_mse": float(mse.detach().item()),
        f"{split}_masked_mae": float(mae.detach().item()),
        f"{split}_masked_psnr": mse_to_psnr(float(mse.detach().item())),
        f"{split}_masked_patch_ratio": float(outputs.query_patch_mask.float().mean().item()),
    }
    if lpips_loss_fn is not None:
        lpips_loss = compute_lpips_loss(
            lpips_loss_fn,
            outputs.reconstructed_query_images,
            query_images,
        )
        metrics[f"{split}_lpips_loss"] = float(lpips_loss.detach().item())
    return (
        metrics,
        outputs.masked_query_images.detach().cpu(),
        outputs.reconstructed_query_images.detach().cpu(),
        query_images.detach().cpu(),
        outputs.query_patch_mask.detach().cpu(),
    )


def train_step(
    model: DA3MaskedPatchModel,
    optimizer: torch.optim.Optimizer,
    lpips_loss_fn: torch.nn.Module | None,
    batch: MaskReconstructionBatch,
    *,
    device: str,
    mask_ratio: float,
    mask_seed: int,
    background_threshold: float,
    lpips_weight: float,
) -> dict[str, float]:
    batch = move_mask_batch(batch, device)
    model.train()
    patch_mask = sample_random_patch_mask(
        images=batch.train_query_images,
        patch_size=model.patch_size,
        mask_ratio=mask_ratio,
        seed=mask_seed,
        background_threshold=background_threshold,
    ).to(device)
    outputs = model(
        support_images=batch.support_images,
        query_images=batch.train_query_images,
        query_patch_mask=patch_mask,
        query_intrinsics=batch.train_query_intrinsics,
        query_c2w=batch.train_query_c2w,
    )
    patch_metrics = compute_masked_patch_metrics(
        outputs.pred_patch_rgb,
        outputs.target_patch_rgb,
        outputs.query_patch_mask,
    )
    rgb_loss = patch_metrics["mse"]
    if lpips_loss_fn is not None and lpips_weight > 0.0:
        lpips_loss = compute_lpips_loss(
            lpips_loss_fn,
            outputs.reconstructed_query_images,
            batch.train_query_images,
        )
    else:
        lpips_loss = rgb_loss.new_zeros(())
    loss = rgb_loss + (lpips_weight * lpips_loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    loss_value = float(loss.detach().item())
    return {
        "loss": loss_value,
        "train_masked_mse": float(rgb_loss.detach().item()),
        "train_masked_mae": float(patch_metrics["mae"].detach().item()),
        "train_masked_psnr": mse_to_psnr(float(rgb_loss.detach().item())),
        "train_masked_patch_ratio": float(outputs.query_patch_mask.float().mean().item()),
        "train_lpips_loss": float(lpips_loss.detach().item()),
        "train_total_loss": loss_value,
    }


def save_loss_plot(
    train_history: list[dict[str, float]],
    eval_history: list[dict[str, float]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [item["step"] for item in train_history],
        [item["train_masked_mse"] for item in train_history],
        marker="o",
        linewidth=2,
        label="train_masked_mse",
    )
    if eval_history:
        axis.plot(
            [item["step"] for item in eval_history],
            [item["eval_masked_mse"] for item in eval_history],
            marker="s",
            linewidth=2,
            label="eval_masked_mse",
        )
    axis.set_xlabel("Step")
    axis.set_ylabel("Masked Patch MSE")
    axis.set_title("da3-mask loss history")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_psnr_plot(
    train_history: list[dict[str, float]],
    eval_history: list[dict[str, float]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [item["step"] for item in train_history],
        [item["train_masked_psnr"] for item in train_history],
        marker="o",
        linewidth=2,
        label="train_masked_psnr",
    )
    if eval_history:
        axis.plot(
            [item["step"] for item in eval_history],
            [item["eval_masked_psnr"] for item in eval_history],
            marker="s",
            linewidth=2,
            label="eval_masked_psnr",
        )
    axis.set_xlabel("Step")
    axis.set_ylabel("PSNR")
    axis.set_title("da3-mask PSNR history")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def select_view_indices(num_views: int, max_views: int) -> list[int]:
    if num_views <= max_views:
        return list(range(num_views))
    return torch.linspace(0, num_views - 1, steps=max_views).round().long().tolist()


def save_combined_visualization(
    *,
    train_masked_input_rgb: torch.Tensor,
    train_recon_rgb: torch.Tensor,
    train_target_rgb: torch.Tensor,
    train_patch_mask: torch.Tensor,
    eval_masked_input_rgb: torch.Tensor,
    eval_recon_rgb: torch.Tensor,
    eval_target_rgb: torch.Tensor,
    eval_patch_mask: torch.Tensor,
    output_dir: Path,
    patch_size: int,
    max_views: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    train_diff_rgb = (train_recon_rgb - train_target_rgb).abs().clamp(0.0, 1.0)
    eval_diff_rgb = (eval_recon_rgb - eval_target_rgb).abs().clamp(0.0, 1.0)
    train_mask_rgb = patch_mask_to_image(
        train_patch_mask,
        patch_size=patch_size,
        image_size=train_target_rgb.shape[-2:],
    ).repeat(1, 1, 3, 1, 1)
    eval_mask_rgb = patch_mask_to_image(
        eval_patch_mask,
        patch_size=patch_size,
        image_size=eval_target_rgb.shape[-2:],
    ).repeat(1, 1, 3, 1, 1)

    saved_paths: list[str] = []
    batch_size = min(train_masked_input_rgb.shape[0], eval_masked_input_rgb.shape[0])
    for batch_idx in range(batch_size):
        eval_view_indices = select_view_indices(eval_masked_input_rgb.shape[1], max_views)
        train_view_indices = select_view_indices(train_masked_input_rgb.shape[1], len(eval_view_indices))
        num_rows = max(len(eval_view_indices), len(train_view_indices), 1)
        figure, axes = plt.subplots(num_rows, 10, figsize=(30, max(num_rows, 1) * 3))
        if num_rows == 1:
            axes = axes[None, :]

        for row_idx in range(num_rows):
            if row_idx < len(train_view_indices):
                train_view_idx = train_view_indices[row_idx]
                train_images = (
                    train_mask_rgb[batch_idx, train_view_idx],
                    train_masked_input_rgb[batch_idx, train_view_idx],
                    train_recon_rgb[batch_idx, train_view_idx],
                    train_target_rgb[batch_idx, train_view_idx],
                    train_diff_rgb[batch_idx, train_view_idx],
                )
                train_titles = (
                    f"Train Mask {train_view_idx}",
                    f"Train Masked Input {train_view_idx}",
                    f"Train Recon {train_view_idx}",
                    f"Train GT {train_view_idx}",
                    f"Train Diff {train_view_idx}",
                )
                for col_idx, (image, title) in enumerate(zip(train_images, train_titles)):
                    axes[row_idx, col_idx].imshow(image.permute(1, 2, 0).numpy())
                    axes[row_idx, col_idx].set_title(title)
                    axes[row_idx, col_idx].axis("off")
            else:
                for col_idx in range(5):
                    axes[row_idx, col_idx].axis("off")

            if row_idx < len(eval_view_indices):
                eval_view_idx = eval_view_indices[row_idx]
                eval_images = (
                    eval_mask_rgb[batch_idx, eval_view_idx],
                    eval_masked_input_rgb[batch_idx, eval_view_idx],
                    eval_recon_rgb[batch_idx, eval_view_idx],
                    eval_target_rgb[batch_idx, eval_view_idx],
                    eval_diff_rgb[batch_idx, eval_view_idx],
                )
                eval_titles = (
                    f"Eval Mask {eval_view_idx}",
                    f"Eval Masked Input {eval_view_idx}",
                    f"Eval Recon {eval_view_idx}",
                    f"Eval GT {eval_view_idx}",
                    f"Eval Diff {eval_view_idx}",
                )
                for offset, (image, title) in enumerate(zip(eval_images, eval_titles), start=5):
                    axes[row_idx, offset].imshow(image.permute(1, 2, 0).numpy())
                    axes[row_idx, offset].set_title(title)
                    axes[row_idx, offset].axis("off")
            else:
                for col_idx in range(5, 10):
                    axes[row_idx, col_idx].axis("off")

        figure.tight_layout()
        output_path = output_dir / f"combined_batch_{batch_idx:02d}.png"
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(str(output_path))
    return saved_paths


def main() -> None:
    args = build_argparser().parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.mask_ratio <= 0.0 or args.mask_ratio > 1.0:
        raise ValueError("--mask-ratio must be in the range (0, 1]")
    if args.co3d_fallback_eval_ratio <= 0.0 or args.co3d_fallback_eval_ratio >= 1.0:
        raise ValueError("--co3d-fallback-eval-ratio must be in the range (0, 1)")
    if args.lpips_weight < 0.0:
        raise ValueError("--lpips-weight must be non-negative")
    if args.background_threshold < 0.0 or args.background_threshold >= 0.5:
        raise ValueError("--background-threshold must be in the range [0, 0.5)")
    if args.image_size % 14 != 0:
        raise ValueError("--image-size must be divisible by 14 for the DA3/DINOv2 patch grid")
    if args.resume is not None and not args.resume.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")

    os.environ["DA3_LOG_LEVEL"] = args.da3_log_level
    log_progress("starting da3-mask training")
    log_progress(
        f"args parsed | dataset={args.dataset} backbone={args.backbone} head={args.head_type} "
        f"steps={args.steps} batch_size={args.batch_size} root={args.root}"
    )
    if args.resume is not None and args.output_dir is None:
        output_dir = args.resume.resolve().parent
    else:
        output_dir = args.output_dir if args.output_dir is not None else build_default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    log_progress(f"output dir ready: {output_dir}")
    start_time = time.perf_counter()

    log_progress("loading dataset...")
    dataset_load_start = time.perf_counter()
    dataset = build_dataset(args)
    dataset_load_seconds = time.perf_counter() - dataset_load_start
    log_progress(f"dataset loaded | scenes={len(dataset.scenes)} size={len(dataset)}")
    log_progress(f"dataset loading took {dataset_load_seconds:.2f}s")

    model_build_start = time.perf_counter()
    model = build_model(args)
    model_build_seconds = time.perf_counter() - model_build_start
    log_progress("model object created")
    log_progress(f"model construction took {model_build_seconds:.2f}s")

    log_progress("building first batch...")
    first_batch_start = time.perf_counter()
    first_batch = build_step_batch(
        dataset,
        step_idx=0,
        batch_size=args.batch_size,
        base_seed=args.seed,
    )
    first_batch_seconds = time.perf_counter() - first_batch_start
    log_progress(
        "first batch ready | "
        f"support={tuple(first_batch.support_images.shape)} "
        f"train_query={tuple(first_batch.train_query_images.shape)} "
        f"eval_query={tuple(first_batch.eval_query_images.shape)}"
    )
    log_progress(f"first batch construction took {first_batch_seconds:.2f}s")
    log_progress("materializing trainable modules...")
    materialize_start = time.perf_counter()
    materialize_trainable_modules(
        model,
        first_batch,
        device=args.device,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
        background_threshold=args.background_threshold,
    )
    materialize_seconds = time.perf_counter() - materialize_start
    log_progress("materialization done")
    log_progress(f"materialization took {materialize_seconds:.2f}s")
    log_progress("building optimizer...")
    optimizer_build_start = time.perf_counter()
    optimizer = build_optimizer(args, model)
    optimizer_build_seconds = time.perf_counter() - optimizer_build_start
    log_progress("optimizer ready")
    log_progress(f"optimizer construction took {optimizer_build_seconds:.2f}s")
    lpips_build_seconds = 0.0
    lpips_loss_fn: torch.nn.Module | None = None
    if args.lpips_weight > 0.0:
        log_progress("building LPIPS loss...")
        lpips_build_start = time.perf_counter()
        lpips_loss_fn = build_lpips_loss_fn(args.device, net=args.lpips_net)
        lpips_build_seconds = time.perf_counter() - lpips_build_start
        log_progress(f"LPIPS ready | net={args.lpips_net} build_time={lpips_build_seconds:.2f}s")
    trainable_params = count_trainable_parameters(model)
    train_history: list[dict[str, float]] = []
    eval_history: list[dict[str, float]] = []
    elapsed_offset = 0.0
    completed_steps = 0
    last_batch = first_batch

    if args.resume is not None:
        log_progress(f"loading checkpoint from {args.resume}...")
        train_history, eval_history, elapsed_offset = load_checkpoint(args.resume, model, optimizer)
        completed_steps = int(train_history[-1]["step"]) if train_history else 0
        reference_step = max(completed_steps - 1, 0)
        last_batch = build_step_batch(
            dataset,
            step_idx=reference_step,
            batch_size=args.batch_size,
            base_seed=args.seed,
        )
        log_progress(f"checkpoint loaded | completed_steps={completed_steps}")

    print(
        f"da3-mask start | dataset={args.dataset} backbone={args.backbone} device={args.device} "
        f"head={args.head_type} "
        f"image={args.image_size} support={args.support_views} "
        f"train_query={args.train_query_views} eval_query={args.eval_query_views} "
        f"eval_split={'test' if args.dataset == 'co3d' else args.eval_split} mask_ratio={args.mask_ratio:.3f} "
        f"bg_thresh={args.background_threshold:.3f} "
        f"output_dir={output_dir} trainable_params={trainable_params} "
        f"optimizer={args.optimizer} lr={args.lr} backbone_trainable={args.backbone_trainable}"
    )
    if args.dataset == "co3d" and hasattr(dataset, "available_categories"):
        print(
            f"co3d categories used={len(dataset.available_categories)} "
            f"skipped_missing_set_list={len(getattr(dataset, 'skipped_categories', ()))}"
        )
        if getattr(dataset, "skipped_categories", ()):
            preview = ", ".join(list(dataset.skipped_categories)[:10])
            suffix = " ..." if len(dataset.skipped_categories) > 10 else ""
            log_progress(f"co3d skipped categories preview: {preview}{suffix}")
        if getattr(dataset, "category_set_lists", {}):
            preview_items = list(dataset.category_set_lists.items())[:10]
            preview = ", ".join(
                f"{category}:{'|'.join(names[:3])}{'...' if len(names) > 3 else ''}"
                for category, names in preview_items
            )
            suffix = " ..." if len(dataset.category_set_lists) > 10 else ""
            log_progress(f"co3d set_lists preview: {preview}{suffix}")
    if args.resume is not None:
        print(f"resuming from {args.resume} at completed_step={completed_steps}")

    log_progress("entering training loop")
    progress = tqdm(
        range(completed_steps + 1, args.steps + 1),
        desc="train",
        dynamic_ncols=True,
        leave=True,
        initial=completed_steps,
        total=args.steps,
    )
    latest_train_mse: float | None = (
        float(train_history[-1]["train_masked_mse"]) if train_history else None
    )
    latest_eval_mse: float | None = (
        float(eval_history[-1]["eval_masked_mse"]) if eval_history else None
    )
    latest_train_psnr: float | None = (
        float(train_history[-1]["train_masked_psnr"]) if train_history else None
    )
    latest_eval_psnr: float | None = (
        float(eval_history[-1]["eval_masked_psnr"]) if eval_history else None
    )
    latest_train_lpips: float | None = (
        float(train_history[-1]["train_lpips_loss"]) if train_history and "train_lpips_loss" in train_history[-1] else None
    )
    latest_eval_lpips: float | None = (
        float(eval_history[-1]["eval_lpips_loss"]) if eval_history and "eval_lpips_loss" in eval_history[-1] else None
    )
    batch_build_seconds_total = 0.0
    batch_build_count = 0
    train_step_seconds_total = 0.0
    train_step_count = 0
    eval_step_seconds_total = 0.0
    eval_step_count = 0

    for step in progress:
        batch_build_start = time.perf_counter()
        batch = (
            first_batch
            if step == 1 and completed_steps == 0
            else build_step_batch(
                dataset,
                step_idx=step - 1,
                batch_size=args.batch_size,
                base_seed=args.seed,
            )
        )
        batch_build_seconds_total += time.perf_counter() - batch_build_start
        batch_build_count += 1
        last_batch = batch
        train_step_start = time.perf_counter()
        train_metrics = train_step(
            model,
            optimizer,
            lpips_loss_fn,
            batch,
            device=args.device,
            mask_ratio=args.mask_ratio,
            mask_seed=args.seed + (step * 1009),
            background_threshold=args.background_threshold,
            lpips_weight=args.lpips_weight,
        )
        train_step_seconds_total += time.perf_counter() - train_step_start
        train_step_count += 1
        train_history.append({"step": float(step), **train_metrics})
        latest_train_mse = train_metrics["train_masked_mse"]
        latest_train_psnr = train_metrics["train_masked_psnr"]
        latest_train_lpips = train_metrics["train_lpips_loss"]
        if step == completed_steps + 1 or step % 50 == 0 or step == args.steps:
            log_progress(
                f"step {step}/{args.steps} | "
                f"train_mse={latest_train_mse:.6f} "
                f"train_psnr={latest_train_psnr:.2f} "
                f"train_lpips={latest_train_lpips:.4f}"
            )

        if step % max(args.eval_every, 1) == 0:
            model.eval()
            eval_step_start = time.perf_counter()
            with torch.no_grad():
                eval_metrics, _, _, _, _ = run_masked_step(
                    model,
                    batch,
                    split="eval",
                    device=args.device,
                    mask_ratio=args.mask_ratio,
                    mask_seed=args.seed + (step * 2003) + 1,
                    background_threshold=args.background_threshold,
                    lpips_loss_fn=lpips_loss_fn,
                )
            eval_step_seconds_total += time.perf_counter() - eval_step_start
            eval_step_count += 1
            eval_history.append({"step": float(step), **eval_metrics})
            latest_eval_mse = eval_metrics["eval_masked_mse"]
            latest_eval_psnr = eval_metrics["eval_masked_psnr"]
            latest_eval_lpips = eval_metrics.get("eval_lpips_loss")

        current_lr = float(optimizer.param_groups[0]["lr"])
        progress.set_postfix(
            {
                "lr": f"{current_lr:.2e}",
                "train_mse": f"{latest_train_mse:.6f}" if latest_train_mse is not None else "-",
                "eval_mse": f"{latest_eval_mse:.6f}" if latest_eval_mse is not None else "-",
                "train_psnr": f"{latest_train_psnr:.2f}" if latest_train_psnr is not None else "-",
                "eval_psnr": f"{latest_eval_psnr:.2f}" if latest_eval_psnr is not None else "-",
                "train_lpips": f"{latest_train_lpips:.4f}" if latest_train_lpips is not None else "-",
                "eval_lpips": f"{latest_eval_lpips:.4f}" if latest_eval_lpips is not None else "-",
            }
        )

    model.eval()
    log_progress("running final visualization passes...")
    final_pass_start = time.perf_counter()
    with torch.no_grad():
        train_vis_metrics, train_masked_input, train_recon, train_gt, train_mask = run_masked_step(
            model,
            last_batch,
            split="train",
            device=args.device,
            mask_ratio=args.mask_ratio,
            mask_seed=args.seed + 90_001,
            background_threshold=args.background_threshold,
            lpips_loss_fn=lpips_loss_fn,
        )
        eval_vis_metrics, eval_masked_input, eval_recon, eval_gt, eval_mask = run_masked_step(
            model,
            last_batch,
            split="eval",
            device=args.device,
            mask_ratio=args.mask_ratio,
            mask_seed=args.seed + 90_002,
            background_threshold=args.background_threshold,
            lpips_loss_fn=lpips_loss_fn,
        )
    final_pass_seconds = time.perf_counter() - final_pass_start

    log_progress("saving visualizations and metrics...")
    save_artifacts_start = time.perf_counter()
    combined_vis_paths = save_combined_visualization(
        train_masked_input_rgb=train_masked_input,
        train_recon_rgb=train_recon,
        train_target_rgb=train_gt,
        train_patch_mask=train_mask,
        eval_masked_input_rgb=eval_masked_input,
        eval_recon_rgb=eval_recon,
        eval_target_rgb=eval_gt,
        eval_patch_mask=eval_mask,
        output_dir=output_dir,
        patch_size=model.patch_size,
        max_views=args.max_vis_views,
    )
    save_loss_plot(train_history, eval_history, output_dir / "loss_curve.png")
    save_psnr_plot(train_history, eval_history, output_dir / "psnr_curve.png")
    save_artifacts_seconds = time.perf_counter() - save_artifacts_start

    total_elapsed = elapsed_offset + (time.perf_counter() - start_time)
    if args.save_checkpoint:
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
        "checkpoint_path": str(checkpoint_path) if args.save_checkpoint else None,
        "loss_curve_path": str(output_dir / "loss_curve.png"),
        "psnr_curve_path": str(output_dir / "psnr_curve.png"),
        "combined_visualization_paths": combined_vis_paths,
        "trainable_params": trainable_params,
        "optimizer": args.optimizer,
        "resume_path": str(args.resume) if args.resume is not None else None,
        "completed_steps_before_resume": completed_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "mask_ratio": args.mask_ratio,
        "mask_fill_value": args.mask_fill_value,
        "background_threshold": args.background_threshold,
        "dataset": args.dataset,
        "root": str(args.root),
        "co3d_set_list": args.co3d_set_list if args.dataset == "co3d" else None,
        "co3d_fallback_eval_ratio": args.co3d_fallback_eval_ratio if args.dataset == "co3d" else None,
        "co3d_available_categories": list(getattr(dataset, "available_categories", ())),
        "co3d_skipped_categories": list(getattr(dataset, "skipped_categories", ())),
        "co3d_category_set_lists": dict(getattr(dataset, "category_set_lists", {})),
        "head_type": args.head_type,
        "head_hidden_dim": args.head_hidden_dim,
        "cross_attn_heads": args.cross_attn_heads,
        "cross_attn_mlp_ratio": args.cross_attn_mlp_ratio,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "backbone": args.backbone,
        "backbone_trainable": args.backbone_trainable,
        "elapsed_seconds": total_elapsed,
        "timing": {
            "dataset_load": summarize_timing(dataset_load_seconds, 1),
            "model_build": summarize_timing(model_build_seconds, 1),
            "first_batch": summarize_timing(first_batch_seconds, 1),
            "materialize": summarize_timing(materialize_seconds, 1),
            "optimizer_build": summarize_timing(optimizer_build_seconds, 1),
            "lpips_build": summarize_timing(lpips_build_seconds, 1),
            "batch_build": summarize_timing(batch_build_seconds_total, batch_build_count),
            "train_step": summarize_timing(train_step_seconds_total, train_step_count),
            "eval_step": summarize_timing(eval_step_seconds_total, eval_step_count),
            "final_pass": summarize_timing(final_pass_seconds, 1),
            "save_artifacts": summarize_timing(save_artifacts_seconds, 1),
        },
        "lpips_weight": args.lpips_weight,
        "lpips_net": args.lpips_net,
        "final_train_metrics": train_vis_metrics,
        "final_eval_metrics": eval_vis_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    timing = metrics_payload["timing"]
    log_progress(
        "timing summary | "
        f"dataset_load={timing['dataset_load']['total_seconds']:.2f}s "
        f"model_build={timing['model_build']['total_seconds']:.2f}s "
        f"first_batch={timing['first_batch']['total_seconds']:.2f}s "
        f"materialize={timing['materialize']['total_seconds']:.2f}s "
        f"batch_build_avg={timing['batch_build']['avg_seconds']:.2f}s "
        f"train_step_avg={timing['train_step']['avg_seconds']:.2f}s "
        f"eval_step_avg={timing['eval_step']['avg_seconds']:.2f}s "
        f"final_pass={timing['final_pass']['total_seconds']:.2f}s "
        f"save_artifacts={timing['save_artifacts']['total_seconds']:.2f}s"
    )
    print(f"training finished in {total_elapsed:.2f}s")
    if args.save_checkpoint:
        print(f"saved checkpoint to {checkpoint_path}")
    print(f"saved loss plot to {output_dir / 'loss_curve.png'}")
    print(f"saved psnr plot to {output_dir / 'psnr_curve.png'}")
    if combined_vis_paths:
        print(f"saved combined visualization to {combined_vis_paths[0]}")
    print(f"saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
