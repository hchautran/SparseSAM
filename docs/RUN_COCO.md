# Run COCO Evaluation

This guide runs COCO instance-segmentation evaluation for SparseSAM algorithms using detector-proposed boxes. The runner uses the local configs in `tasks/sam_coco/configs/` and the algorithm registry in `algos/registry.py`; it does not use quantization, calibration, DUO, or DiffDUO paths.

## What It Runs

The pipeline is:

1. Build an MMDetection detector wrapper from a local COCO config.
2. Build a SAM-HQ predictor from `--model-ckt`.
3. Patch the SAM-HQ image encoder with `piecewise`, `sparge`, `sparsesam`, `tome`, or `gradtome`.
4. Replace the detector wrapper's bootstrap SAM predictor with the patched SAM-HQ predictor.
5. Run COCO `segm` evaluation and write CSV results with mAP, encoder latency, and peak-memory metrics.

Supported detector configs are included locally under `tasks/sam_coco/configs/` for `dino`, `hdetr`, and `yolox`, with SAM-HQ `vit_b`, `vit_l`, and `vit_h` variants.

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
export PYTHONPATH=$PWD:$PWD/algos/3rd_party/sam-hq:$SAM_QUANT_ROOT:$SAM_QUANT_ROOT/PTQ4SAM:$SAM_QUANT_ROOT/PTQ4SAM/projects/instance_segment_anything/ops:$PYTHONPATH
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
├── sam_hq_vit_b.pth
├── sam_hq_vit_l.pth
├── sam_hq_vit_h.pth
├── sam_vit_l_0b3195.pth
├── focalnet_l_dino.pth
├── r50_hdetr.pth
└── yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
```

`--model-ckt` and `--det-sam-ckt` are intentionally separate:

- `--model-ckt` is the SAM-HQ checkpoint that receives the algorithm patch.
- `--det-sam-ckt` is the original SAM checkpoint used only because the MMDet wrapper constructs a bootstrap predictor before the runner replaces it.
- For `vit_b` and `vit_h`, the default wrapper path uses `DET_SAM_CKT=none` because the injected predictor replaces the bootstrap predictor before inference.

## Portable Wrapper Usage

The wrapper is portable on purpose. You are expected to provide your own paths:

```bash
SAM_QUANT_ROOT=/path/to/PTQ4SAM-parent \
DATA_ROOT=/path/to/coco \
CKPT_ROOT=/path/to/ckpts \
MODEL_TYPE=vit_l \
DETECTOR=dino \
sh tasks/sam_coco/eval_coco.sh
```

Common env knobs:

- `ALGOS="none sparsesam gradtome tome sparge piecewise"`
- `RATIOS="0.70 0.50 0.30"`
- `BATCH_SIZES="1"`
- `NUM_SAMPLES=500`
- `MODEL_TYPE=vit_b|vit_l|vit_h`
- `DETECTOR=dino|hdetr|yolox`
- `MLP_MERGE=yes|no`
- `MODEL_CKT=/path/to/ckpts/sam_hq_vit_l.pth`
- `DET_CKPT=/path/to/ckpts/focalnet_l_dino.pth`
- `DET_SAM_CKT=/path/to/ckpts/sam_vit_l_0b3195.pth`
- `OUT_DIR=/path/to/output_dir`

## Smoke Test

Run 50 COCO validation images with YOLOX, ViT-L, and 25% sparsity:

```bash
export SAM_QUANT_ROOT=/path/to/PTQ4SAM-parent
export DATA_ROOT=/path/to/coco
export CKPT_ROOT=/path/to/ckpts

python tasks/sam_coco/eval_coco.py \
  --sam-quant-root $SAM_QUANT_ROOT \
  --data-root $DATA_ROOT \
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
CKPT_ROOT=/path/to/ckpts \
MODEL_TYPE=vit_l \
DETECTOR=yolox \
ALGOS="piecewise sparge sparsesam tome gradtome" \
RATIOS="0.25" \
NUM_SAMPLES=50 \
sh tasks/sam_coco/eval_coco.sh
```

## Common Runs

DINO detector:

```bash
python tasks/sam_coco/eval_coco.py \
  --sam-quant-root /path/to/PTQ4SAM-parent \
  --data-root /path/to/coco \
  --detector dino \
  --det-checkpoint /path/to/ckpts/focalnet_l_dino.pth \
  --model-type vit_l \
  --model-ckt /path/to/ckpts/sam_hq_vit_l.pth \
  --det-sam-ckt /path/to/ckpts/sam_vit_l_0b3195.pth \
  --algos none piecewise sparge sparsesam tome gradtome \
  --ratios 0.70 0.50 0.30 \
  --batch-sizes 1
```

HDETR detector:

```bash
python tasks/sam_coco/eval_coco.py \
  --sam-quant-root /path/to/PTQ4SAM-parent \
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
  --sam-quant-root /path/to/PTQ4SAM-parent \
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

- A CSV named `sam_coco_eval_*.csv` with COCO metrics, detector name, model type, sparsity ratio, image count, config path, encoder latency, and peak-memory metrics.
- Raw MMDetection outputs if `--out` is provided.

Use `--output-dir` and `--out` to redirect outputs.

Measured summaries for the current DINO runs live in [`tasks/sam_coco/RESULTS.md`](../tasks/sam_coco/RESULTS.md).

## Notes

- COCO uses `batch-size=1` by default because the detector wrapper calls SAM per image. Larger batch sizes are exposed but are not the primary supported path.
- `sparsesam` uses an fp16 image-encoder path in the COCO runner because its fused kernel supports fp16/bf16 only.
- `gradtome` can be sensitive to very high sparsity on COCO. Use `--ratios 0.25` for the smoke-tested setting.
- If a detector config path is not provided, the runner picks the local config from `tasks/sam_coco/configs/` based on `--detector` and `--model-type`.
- To replicate the SAM-H results reported in the paper, run the model back in fp32. `sparge` does not support non-`2**k` hidden dimensions, so its SAM-H paper entry is intentionally left blank.
