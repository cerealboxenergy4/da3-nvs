from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SceneBatch:
    support_images: torch.Tensor
    support_intrinsics: torch.Tensor
    support_c2w: torch.Tensor
    train_query_intrinsics: torch.Tensor
    train_query_c2w: torch.Tensor
    train_target_rgb: torch.Tensor
    eval_query_intrinsics: torch.Tensor
    eval_query_c2w: torch.Tensor
    eval_target_rgb: torch.Tensor

