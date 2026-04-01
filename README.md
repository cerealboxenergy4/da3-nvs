# da3-nvs

`da3-nvs`는 [`outline.md`](./outline.md)에 적힌 PoC를 바로 시작할 수 있도록 만든 초기 코드베이스입니다.

현재 포함된 구성은 다음과 같습니다.

- `src/da3_nvs/data/rays.py`: `ttt-nvs`와 같은 규약의 ray map 유틸리티
- `src/da3_nvs/data/nerf_synthetic.py`: `nerf_synthetic`용 POC dataset loader
- `src/da3_nvs/models/ray_encoder.py`: ray map patch embedding
- `src/da3_nvs/models/da3_backbone.py`: `Depth-Anything-3` 백본 토큰 추출 래퍼
- `src/da3_nvs/models/nvs_head.py`: cross-attention 기반 NVS head
- `src/da3_nvs/models/da3_nvs.py`: DA3 token + ray lifting + query rendering 조립
- `src/da3_nvs/train/engine.py`: RGB MSE 기반 최소 trainer

기본 ray map은 `ttt-nvs`와 맞춰 9D `(origin, direction, origin x direction)`를 사용합니다. 필요하면 `include_moment=False`로 6D `(origin, direction)`도 사용할 수 있습니다.

현재 POC 기본값은 `nerf_synthetic`의 8개 오브젝트에 대해 `224x224`, `support 64 views`, `query 16 views`입니다.

## Local DA3 Dependency

`DA3PatchBackbone`은 아래 둘 중 하나를 가정합니다.

- `depth_anything_3`가 현재 Python 환경에 설치되어 있음
- 현재 프로젝트와 같은 부모 디렉터리에 `Depth-Anything-3/src`가 존재함

현재 워크스페이스에서는 두 번째 경로를 자동으로 찾도록 구현했습니다.

## Quick Sketch

```python
import torch

from da3_nvs import DA3NVSModel, NerfSyntheticSceneDataset, scene_batch_collate

model = DA3NVSModel(
    patch_size=14,
    backbone_model_name="da3-large",
)

dataset = NerfSyntheticSceneDataset(
    root="/home/hunn/projects/datasets/nerf_synthetic",
)
batch = scene_batch_collate([dataset[0]])

scene = model.encode_support(
    support_images=batch.support_images,
    support_intrinsics=batch.support_intrinsics,
    support_c2w=batch.support_c2w,
)
outputs = model.render_queries(
    scene,
    query_intrinsics=batch.train_query_intrinsics,
    query_c2w=batch.train_query_c2w,
    query_image_size=batch.train_target_rgb.shape[-2:],
)
print(outputs.pred_rgb.shape)
```

## Tests

```bash
pytest /home/hunn/projects/da3-nvs/tests
```

테스트는 실제 DA3 가중치를 요구하지 않도록 mock backbone으로 shape 계약만 검증합니다.

## Mock Train

가장 가벼운 파이프라인 smoke는 toy backbone으로 돌리면 됩니다.

```bash
cd /home/hunn/projects/da3-nvs
PYTHONPATH=src conda run -n tttnvs python scripts/mock_train.py --backbone toy --steps 2 --device cuda
```

실제 frozen DA3 backbone 경로는 아래처럼 돌릴 수 있습니다.

```bash
cd /home/hunn/projects/da3-nvs
PYTHONPATH=src conda run -n tttnvs python scripts/mock_train.py --backbone da3 --steps 1 --device cuda
```

필요하면 `--scene chair --scene lego`, `--image-size 224`, `--support-views 64`, `--train-query-views 16` 같은 식으로 그대로 덮어쓰면 됩니다.
