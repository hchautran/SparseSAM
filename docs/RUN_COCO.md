# Run COCO Evaluation

This guide runs COCO instance-segmentation evaluation for SparseSAM algorithms using detector-proposed boxes. The runner uses the local configs in `tasks/sam_coco/configs/` and the algorithm registry in `algos/registry.py`; it does not use quantization, calibration, DUO, or DiffDUO paths.

## What It Runs

The pipeline is:

1. Build an MMDetection detector wrapper from a local COCO config.
2. Build a SAM-HQ predictor from `--model-ckt`.
3. Patch the SAM-HQ image encoder with `piecewise`, `sparge`, `sparsesam`, `tome`, or `gradtome`.
4. Replace the detector wrapper's bootstrap SAM predictor with the patched SAM-HQ predictor.
5. Run COCO `segm` evaluation and write mAP results.

Supported ViT-L detector configs are included locally:

```text
tasks/sam_coco/configs/focalnet_dino/focalnet-l-dino_sam-vit-l.py
tasks/sam_coco/configs/hdetr/r50-hdetr_sam-vit-l.py
tasks/sam_coco/configs/yolox/yolo_l-sam-vit-l.py
```

## Environment

Install the normal SparseSAM dependencies first, then install SAM-HQ and any optional algorithm kernels required by the algorithms you plan to run:

```bash
pip install -e .
pip install -e algos/3rd_party/sam-hq
pip install -e algos/3rd_party/piecewise-sparse-attention

# SpargeAttn commonly needs a CUDA build:
# python algos/3rd_party/SpargeAttn/setup.py develop
```

The COCO detector wrappers also depend on PTQ4SAM's `projects/instance_segment_anything` package and its deformable-attention CUDA op. Put a PTQ4SAM checkout anywhere and point `SAM_QUANT_ROOT` to the directory that contains `PTQ4SAM/`:

```bash
export SAM_QUANT_ROOT=/path/to/PTQ4SAM-parent
```

Expected layout:

```text
$SAM_QUANT_ROOT/PTQ4SAM/projects/instance_segment_anything/
$SAM_QUANT_ROOT/PTQ4SAM/projects/instance_segment_anything/ops/
```

Make PTQ4SAM and the ops importable:

```bash
export PYTHONPATH=$PWD:$PWD/algos/3rd_party/sam-hq:$SAM_QUANT_ROOT/PTQ4SAM:$SAM_QUANT_ROOT/PTQ4SAM/projects/instance_segment_anything/ops:$PYTHONPATH
```

Build the deformable-attention op if it is not already built for your Python/CUDA/PyTorch version:

```bash
cd $SAM_QUANT_ROOT/PTQ4SAM/projects/instance_segment_anything/ops
python setup.py build_ext --inplace
cd -
```

You also need compatible `mmcv`, `mmdet`, `pycocotools`, `opencv-python`, `timm`, and PyTorch/CUDA versions. Use the PTQ4SAM/MMDetection versions required by your PTQ4SAM checkout.

## Data And Checkpoints

Prepare COCO validation data in this layout:

```text
/path/to/coco/
├── annotations/
│   └── instances_val2017.json
└── val2017/
    ├── 000000000139.jpg
    └── ...
```

Prepare checkpoints:

```text
/path/to/ckpts/
├── sam_hq_vit_l.pth          # SAM-HQ checkpoint patched/evaluated by SparseSAM
├── sam_vit_l_0b3195.pth      # original SAM checkpoint used only to bootstrap detector wrapper
├── focalnet_l_dino.pth       # for --detector dino
├── r50_hdetr.pth             # for --detector hdetr
└── yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth  # for --detector yolox
```

`--model-ckt` and `--det-sam-ckt` are intentionally separate:

- `--model-ckt` is the SAM-HQ checkpoint that receives the SparseSAM algorithm patch.
- `--det-sam-ckt` is the original SAM checkpoint needed only because the MMDet wrapper constructs a bootstrap predictor before the runner replaces it.

## Smoke Test

Run 50 COCO validation images with YOLOX, ViT-L, and 25% sparsity:

```bash
export SAM_QUANT_ROOT=/path/to/PTQ4SAM-parent
export COCO_ROOT=/path/to/coco
export CKPT_ROOT=/path/to/ckpts

python tasks/sam_coco/eval_coco.py \
  --data-root $COCO_ROOT \
  --detector yolox \
  --det-checkpoint $CKPT_ROOT/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth \
  --model-type vit_l \
  --model-ckt $CKPT_ROOT/sam_hq_vit_l.pth \
  --det-sam-ckt $CKPT_ROOT/sam_vit_l_0b3195.pth \
  --algos piecewise sparge sparsesam tome gradtome \
  --ratios 0.25 \
  --batch-sizes 1 \
  --num-samples 50 \
  --workers-per-gpu 0
```

Equivalent wrapper usage:

```bash
SAM_QUANT_ROOT=/path/to/PTQ4SAM-parent \
DATA_ROOT=/path/to/coco \
MODEL_CKT=/path/to/ckpts/sam_hq_vit_l.pth \
DET_SAM_CKT=/path/to/ckpts/sam_vit_l_0b3195.pth \
DETECTOR=yolox \
DET_CHECKPOINT=/path/to/ckpts/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth \
ALGOS="piecewise sparge sparsesam tome gradtome" \
RATIOS="0.25" \
NUM_SAMPLES=50 \
sh tasks/sam_coco/eval_coco.sh
```

## Common Runs

DINO detector:

```bash
python tasks/sam_coco/eval_coco.py \
  --data-root /path/to/coco \
  --detector dino \
  --det-checkpoint /path/to/ckpts/focalnet_l_dino.pth \
  --model-type vit_l \
  --model-ckt /path/to/ckpts/sam_hq_vit_l.pth \
  --det-sam-ckt /path/to/ckpts/sam_vit_l_0b3195.pth \
  --algos none piecewise sparge sparsesam tome gradtome \
  --ratios 0.75 0.5 0.25 \
  --batch-sizes 1
```

HDETR detector:

```bash
python tasks/sam_coco/eval_coco.py \
  --data-root /path/to/coco \
  --detector hdetr \
  --det-checkpoint /path/to/ckpts/r50_hdetr.pth \
  --model-type vit_l \
  --model-ckt /path/to/ckpts/sam_hq_vit_l.pth \
  --det-sam-ckt /path/to/ckpts/sam_vit_l_0b3195.pth \
  --algos none piecewise sparge sparsesam tome gradtome \
  --ratios 0.25 \
  --batch-sizes 1
```

YOLOX detector:

```bash
python tasks/sam_coco/eval_coco.py \
  --data-root /path/to/coco \
  --detector yolox \
  --det-checkpoint /path/to/ckpts/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth \
  --model-type vit_l \
  --model-ckt /path/to/ckpts/sam_hq_vit_l.pth \
  --det-sam-ckt /path/to/ckpts/sam_vit_l_0b3195.pth \
  --algos none piecewise sparge sparsesam tome gradtome \
  --ratios 0.25 \
  --batch-sizes 1
```

## Outputs

By default, results are written to:

```text
benchmark_results/sam_coco/
```

Each run writes:

- A CSV named `sam_coco_eval_*.csv` with COCO metrics, detector name, model type, sparsity ratio, image count, and config path.
- Raw MMDetection outputs under `demo/coco/` unless `--out` is changed.

Use `--output-dir` and `--out` to redirect outputs.

## Notes

- COCO uses `batch-size=1` by default because the detector wrapper calls SAM per image. Larger batch sizes are exposed but not the primary supported path.
- `sparsesam` uses an fp16 image-encoder path in the COCO runner because its fused kernel supports fp16/bf16 only.
- `gradtome` can be sensitive to very high sparsity on COCO. Use `--ratios 0.25` for the smoke-tested setting.
- If a detector config path is not provided, the runner picks the local config from `tasks/sam_coco/configs/` based on `--detector` and `--model-type`.
