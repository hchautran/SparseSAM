#!/bin/bash
# SAM2 HQ44K throughput + mIoU sweep with the sparsesam Hiera patch.
#
# Defaults are HQ-Hiera-large (the only checkpoint variant whose decoder
# accepts the `hq_token_only` flag the SAM-HQ2 predictor passes). To eval
# a non-HQ SAM2 checkpoint (sam2.1_hiera_t/s/b+/l), use the upstream
# facebookresearch/segment-anything-2 predictor instead — sam-hq2's
# predictor is HQ-specific.
#
# Knobs (env-overridable):
#   RATIOS        per-block keep fractions (sparsesam)
#   BATCH_SIZES   batch sizes
#   NUM_SAMPLES   images per dataset (0 = full val set)
#   CFG, CKPT     model config + ckpt
#   AMP           bfloat16 (default) | float16 | float32
#   MLP_MERGE     yes (default) | no  (no = sparse-attn only, full MLP)
#   DATASET_IDX   space-separated dataset indices (default: all)
#   START_BLOCK   only patch blocks with idx >= this (default 0 = all).
#                 Set to 24 for HQ-Hiera-L to skip stages 0–1 + first global
#                 (the first global is at block 23, so 24 = "right after").

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RATIOS=${RATIOS:-"0.7 0.5 0.3"}
BATCH_SIZES=${BATCH_SIZES:-"4"}
NUM_SAMPLES=${NUM_SAMPLES:-100}
CFG=${CFG:-./sam-hq/sam-hq2/sam2/configs/sam2.1/sam2.1_hq_hiera_l.yaml}
CKPT=${CKPT:-./ckts/sam2.1_hq_hiera_large.pt}
AMP=${AMP:-bfloat16}
MLP_MERGE=${MLP_MERGE:-yes}
DATASET_IDX=${DATASET_IDX:-}
START_BLOCK=${START_BLOCK:-0}

if [ "$MLP_MERGE" = "no" ] || [ "$MLP_MERGE" = "false" ] || [ "$MLP_MERGE" = "0" ]; then
    MLP_FLAG="--no-mlp-merge"
else
    MLP_FLAG=""
fi

DSI_FLAG=""
if [ -n "$DATASET_IDX" ]; then
    DSI_FLAG="--dataset-idx $DATASET_IDX"
fi

python "$SCRIPT_DIR/eval_sam2_hq44k.py" \
    --model-cfg     "$CFG" \
    --checkpoint    "$CKPT" \
    --ratios        $RATIOS \
    --batch-sizes   $BATCH_SIZES \
    --num-samples   "$NUM_SAMPLES" \
    --amp-dtype     "$AMP" \
    --start-block   "$START_BLOCK" \
    $MLP_FLAG \
    $DSI_FLAG

# CSV lands in ./benchmark_results/sam2_hq44k_<timestamp>.csv.
