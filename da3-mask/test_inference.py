from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from train import (
    build_argparser as build_train_argparser,
    build_dataset,
    build_lpips_loss_fn,
    build_model,
    build_step_batch,
    log_progress,
    materialize_trainable_modules,
    run_masked_step,
    save_combined_visualization,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run checkpointed DA3 masked-patch inference on new data without additional training."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a saved da3-mask checkpoint.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for inference metrics and visualizations. Defaults next to the checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        choices=("nerf_synthetic", "co3d"),
        default=None,
        help="Optional dataset override. Defaults to the dataset stored in the checkpoint args.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional dataset root override.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        default=None,
        help="Optional scene/category override. Repeat to add multiple entries.",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--support-views", type=int, default=None)
    parser.add_argument("--train-query-views", type=int, default=None)
    parser.add_argument("--eval-query-views", type=int, default=None)
    parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default=None,
        help="Override the held-out split for nerf_synthetic runs.",
    )
    parser.add_argument(
        "--co3d-set-list",
        default=None,
        help="Override the CO3D set list filename.",
    )
    parser.add_argument(
        "--co3d-fallback-eval-ratio",
        type=float,
        default=None,
        help="Override the CO3D fallback eval ratio.",
    )
    parser.add_argument(
        "--cache-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override image caching for inference.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size used when sampling inference batches.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Override the mask ratio used during inference.",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=None,
        help="Override the background threshold used for mask sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed used for deterministic batch/mask selection.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of sampled batches to evaluate when --batch-index is not provided.",
    )
    parser.add_argument(
        "--batch-index",
        action="append",
        dest="batch_indices",
        type=int,
        default=None,
        help="Specific deterministic batch indices to evaluate. Repeat to add more.",
    )
    parser.add_argument(
        "--max-vis-views",
        type=int,
        default=None,
        help="Override the maximum number of query views shown in saved visualizations.",
    )
    parser.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute LPIPS during inference when the package is available.",
    )
    return parser


def load_checkpoint_payload(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type != "da3-mask":
        raise RuntimeError(f"Unsupported checkpoint type: {checkpoint_type!r}")
    return checkpoint


def resolve_runtime_args(cli_args: argparse.Namespace, checkpoint_args: dict) -> argparse.Namespace:
    train_defaults = vars(build_train_argparser().parse_args([]))
    defaults = dict(train_defaults)
    defaults.update(checkpoint_args)
    runtime = argparse.Namespace(**defaults)
    checkpoint_dataset = checkpoint_args.get("dataset")

    override_fields = [
        "dataset",
        "root",
        "scenes",
        "image_size",
        "support_views",
        "train_query_views",
        "eval_query_views",
        "eval_split",
        "co3d_set_list",
        "co3d_fallback_eval_ratio",
        "cache_images",
        "device",
        "batch_size",
        "mask_ratio",
        "background_threshold",
        "seed",
        "max_vis_views",
    ]
    for field in override_fields:
        value = getattr(cli_args, field)
        if value is not None:
            setattr(runtime, field, value)

    if (
        cli_args.dataset is not None
        and cli_args.dataset != checkpoint_dataset
        and cli_args.root is None
    ):
        runtime.root = train_defaults["root"]

    selected_indices = (
        sorted(set(cli_args.batch_indices))
        if cli_args.batch_indices
        else list(range(max(cli_args.num_batches, 1)))
    )
    if not selected_indices:
        raise ValueError("No batch indices selected for inference")
    if any(index < 0 for index in selected_indices):
        raise ValueError("--batch-index values must be non-negative")

    runtime.steps = max(selected_indices) + 1
    runtime.batch_indices = selected_indices
    runtime.resume = cli_args.checkpoint
    runtime.save_checkpoint = False
    runtime.output_dir = cli_args.output_dir
    runtime.compute_lpips = cli_args.lpips
    return runtime


def default_output_dir(checkpoint_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return checkpoint_path.resolve().parent / f"inference_{timestamp}"


def load_model_checkpoint(
    checkpoint: dict,
    model,
) -> int:
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
    train_history = list(checkpoint.get("train_history", []))
    return int(train_history[-1]["step"]) if train_history else 0


def main() -> None:
    cli_args = build_argparser().parse_args()
    checkpoint = load_checkpoint_payload(cli_args.checkpoint)
    checkpoint_args = checkpoint.get("args", {})
    runtime_args = resolve_runtime_args(cli_args, checkpoint_args)

    output_dir = runtime_args.output_dir or default_output_dir(cli_args.checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_progress(
        f"inference setup | checkpoint={cli_args.checkpoint} dataset={runtime_args.dataset} "
        f"root={runtime_args.root} batch_size={runtime_args.batch_size} cache_images={runtime_args.cache_images}"
    )
    log_progress("loading dataset...")
    dataset = build_dataset(runtime_args)
    log_progress(f"dataset ready | scenes={len(dataset.scenes)} size={len(dataset)}")

    log_progress("building model...")
    model = build_model(runtime_args)

    first_step_idx = runtime_args.batch_indices[0]
    first_batch = build_step_batch(
        dataset,
        step_idx=first_step_idx,
        batch_size=runtime_args.batch_size,
        base_seed=runtime_args.seed,
    )
    log_progress("materializing modules on the first inference batch...")
    materialize_trainable_modules(
        model,
        first_batch,
        device=runtime_args.device,
        mask_ratio=runtime_args.mask_ratio,
        seed=runtime_args.seed,
        background_threshold=runtime_args.background_threshold,
    )

    completed_steps = load_model_checkpoint(checkpoint, model)
    model = model.to(runtime_args.device)
    model.eval()
    log_progress(f"checkpoint loaded | trained_steps={completed_steps}")

    lpips_loss_fn = None
    if runtime_args.compute_lpips and runtime_args.lpips_weight > 0.0:
        try:
            lpips_loss_fn = build_lpips_loss_fn(runtime_args.device, net=runtime_args.lpips_net)
            log_progress(f"LPIPS enabled for inference | net={runtime_args.lpips_net}")
        except RuntimeError as error:
            log_progress(f"LPIPS unavailable for inference, continuing without it: {error}")

    batch_results: list[dict[str, object]] = []
    log_progress(f"running inference for batch indices: {runtime_args.batch_indices}")
    with torch.no_grad():
        for step_idx in runtime_args.batch_indices:
            batch = build_step_batch(
                dataset,
                step_idx=step_idx,
                batch_size=runtime_args.batch_size,
                base_seed=runtime_args.seed,
            )
            train_metrics, train_masked_input, train_recon, train_gt, train_mask = run_masked_step(
                model,
                batch,
                split="train",
                device=runtime_args.device,
                mask_ratio=runtime_args.mask_ratio,
                mask_seed=runtime_args.seed + (step_idx * 1009),
                background_threshold=runtime_args.background_threshold,
                lpips_loss_fn=lpips_loss_fn,
            )
            eval_metrics, eval_masked_input, eval_recon, eval_gt, eval_mask = run_masked_step(
                model,
                batch,
                split="eval",
                device=runtime_args.device,
                mask_ratio=runtime_args.mask_ratio,
                mask_seed=runtime_args.seed + (step_idx * 2003) + 1,
                background_threshold=runtime_args.background_threshold,
                lpips_loss_fn=lpips_loss_fn,
            )
            batch_output_dir = output_dir / f"batch_{step_idx:04d}"
            vis_paths = save_combined_visualization(
                train_masked_input_rgb=train_masked_input,
                train_recon_rgb=train_recon,
                train_target_rgb=train_gt,
                train_patch_mask=train_mask,
                eval_masked_input_rgb=eval_masked_input,
                eval_recon_rgb=eval_recon,
                eval_target_rgb=eval_gt,
                eval_patch_mask=eval_mask,
                output_dir=batch_output_dir,
                patch_size=model.patch_size,
                max_views=runtime_args.max_vis_views,
            )
            batch_results.append(
                {
                    "batch_index": step_idx,
                    "train_metrics": train_metrics,
                    "eval_metrics": eval_metrics,
                    "visualization_paths": vis_paths,
                }
            )
            log_progress(
                f"batch {step_idx} done | "
                f"train_mse={train_metrics['train_masked_mse']:.6f} "
                f"eval_mse={eval_metrics['eval_masked_mse']:.6f}"
            )

    metrics_payload = {
        "checkpoint_path": str(cli_args.checkpoint),
        "trained_steps": completed_steps,
        "output_dir": str(output_dir),
        "runtime_args": {
            "dataset": runtime_args.dataset,
            "root": str(runtime_args.root),
            "scenes": list(runtime_args.scenes) if runtime_args.scenes else None,
            "image_size": runtime_args.image_size,
            "support_views": runtime_args.support_views,
            "train_query_views": runtime_args.train_query_views,
            "eval_query_views": runtime_args.eval_query_views,
            "eval_split": runtime_args.eval_split,
            "co3d_set_list": runtime_args.co3d_set_list,
            "co3d_fallback_eval_ratio": runtime_args.co3d_fallback_eval_ratio,
            "cache_images": runtime_args.cache_images,
            "device": runtime_args.device,
            "batch_size": runtime_args.batch_size,
            "mask_ratio": runtime_args.mask_ratio,
            "background_threshold": runtime_args.background_threshold,
            "seed": runtime_args.seed,
            "batch_indices": runtime_args.batch_indices,
            "max_vis_views": runtime_args.max_vis_views,
            "lpips_enabled": lpips_loss_fn is not None,
        },
        "batch_results": batch_results,
    }
    metrics_path = output_dir / "inference_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"saved inference metrics to {metrics_path}")
    if batch_results and batch_results[0]["visualization_paths"]:
        first_path = batch_results[0]["visualization_paths"][0]
        print(f"saved first visualization to {first_path}")


if __name__ == "__main__":
    main()
