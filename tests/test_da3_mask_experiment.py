from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "da3-mask" / "da3_mask_experiment.py"
    spec = importlib.util.spec_from_file_location("da3_mask_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sample_random_patch_mask_respects_requested_ratio() -> None:
    experiment = _load_module()
    images = torch.rand(2, 3, 3, 28, 28)
    mask = experiment.sample_random_patch_mask(
        images=images,
        patch_size=14,
        mask_ratio=0.5,
        seed=123,
    )

    assert mask.shape == (2, 3, 4)
    assert mask.dtype == torch.bool
    assert torch.all(mask.sum(dim=-1) == 2)


def test_da3_mask_model_reconstructs_masked_query_shapes() -> None:
    experiment = _load_module()
    model = experiment.DA3MaskedPatchModel(
        patch_size=14,
        backbone=experiment.ToyPatchBackbone(patch_size=14, embed_dim=64),
        head_hidden_dim=128,
    )

    support_images = torch.rand(1, 2, 3, 28, 28)
    query_images = torch.rand(1, 3, 3, 28, 28)
    query_patch_mask = experiment.sample_random_patch_mask(
        images=query_images,
        patch_size=14,
        mask_ratio=0.25,
        seed=7,
    )
    outputs = model(
        support_images=support_images,
        query_images=query_images,
        query_patch_mask=query_patch_mask,
    )

    assert outputs.masked_query_images.shape == query_images.shape
    assert outputs.pred_query_images.shape == query_images.shape
    assert outputs.reconstructed_query_images.shape == query_images.shape
    assert outputs.query_tokens.shape == (1, 3, 4, 64)
    assert outputs.pred_patch_rgb.shape == (1, 3, 4, 3 * 14 * 14)
    assert torch.equal(outputs.query_patch_mask.cpu(), query_patch_mask)


def test_da3_mask_model_supports_dpt_head_shapes() -> None:
    experiment = _load_module()
    model = experiment.DA3MaskedPatchModel(
        patch_size=14,
        backbone=experiment.ToyMultiStageBackbone(patch_size=14, embed_dim=64),
        head_type="dpt",
        decoder_hidden_dim=64,
    )

    support_images = torch.rand(1, 2, 3, 28, 28)
    query_images = torch.rand(1, 3, 3, 28, 28)
    query_patch_mask = experiment.sample_random_patch_mask(
        images=query_images,
        patch_size=14,
        mask_ratio=0.5,
        seed=19,
    )
    outputs = model(
        support_images=support_images,
        query_images=query_images,
        query_patch_mask=query_patch_mask,
    )

    assert outputs.pred_query_images.shape == query_images.shape
    assert outputs.reconstructed_query_images.shape == query_images.shape
    assert outputs.pred_patch_rgb.shape == (1, 3, 4, 3 * 14 * 14)
    assert outputs.query_tokens.shape == (1, 3, 4, 64)


def test_da3_mask_model_supports_cnn_head_shapes() -> None:
    experiment = _load_module()
    model = experiment.DA3MaskedPatchModel(
        patch_size=14,
        backbone=experiment.ToyPatchBackbone(patch_size=14, embed_dim=64),
        head_type="cnn",
        decoder_hidden_dim=64,
    )

    support_images = torch.rand(1, 2, 3, 28, 28)
    query_images = torch.rand(1, 3, 3, 28, 28)
    query_patch_mask = experiment.sample_random_patch_mask(
        images=query_images,
        patch_size=14,
        mask_ratio=0.5,
        seed=23,
    )
    outputs = model(
        support_images=support_images,
        query_images=query_images,
        query_patch_mask=query_patch_mask,
    )

    assert outputs.pred_query_images.shape == query_images.shape
    assert outputs.reconstructed_query_images.shape == query_images.shape
    assert outputs.pred_patch_rgb.shape == (1, 3, 4, 3 * 14 * 14)
    assert outputs.query_tokens.shape == (1, 3, 4, 64)


def test_sample_random_patch_mask_skips_white_and_black_background() -> None:
    experiment = _load_module()
    images = torch.full((1, 1, 3, 28, 28), 0.5)
    images[:, :, :, :14, :14] = 1.0
    images[:, :, :, :14, 14:] = 0.0
    mask = experiment.sample_random_patch_mask(
        images=images,
        patch_size=14,
        mask_ratio=1.0,
        seed=5,
        background_threshold=0.01,
    )

    assert mask.shape == (1, 1, 4)
    assert torch.equal(mask[0, 0], torch.tensor([False, False, True, True]))


def test_co3d_intrinsics_and_pose_helpers_return_expected_shapes() -> None:
    experiment = _load_module()
    intrinsics = experiment._co3d_intrinsics_from_viewpoint(
        {
            "focal_length": [2.3, 2.3],
            "principal_point": [0.0, 0.0],
            "intrinsics_format": "ndc_isotropic",
        },
        annotation_image_size=[640, 479],
        output_image_size=(224, 224),
    )
    c2w = experiment._co3d_c2w_from_viewpoint(
        {
            "R": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "T": [0.1, -0.2, 1.5],
        }
    )

    assert intrinsics.shape == (3, 3)
    assert intrinsics.dtype == torch.float32
    assert c2w.shape == (4, 4)
    assert c2w.dtype == torch.float32
    assert torch.allclose(c2w[:3, 3], torch.tensor([-0.1, 0.2, -1.5]))


def test_resolve_requested_co3d_scene_ids_supports_category_and_fq_names() -> None:
    experiment = _load_module()
    available_scene_ids = [
        "apple/110_13051_23361",
        "apple/189_20393_38136",
        "banana/seq_001",
    ]

    resolved = experiment._resolve_requested_co3d_scene_ids(
        ("apple", "banana/seq_001"),
        available_scene_ids=available_scene_ids,
    )

    assert resolved == (
        "apple/110_13051_23361",
        "apple/189_20393_38136",
        "banana/seq_001",
    )
