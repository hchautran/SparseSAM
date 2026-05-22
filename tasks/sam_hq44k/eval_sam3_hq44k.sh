#!/usr/bin/env bash
# SAM3 HQ44K mIoU sweep — baseline vs sparsesam (partial-MLP, optional flex sparse-attn).
#
# Knobs (env-overridable):
#   RATIOS         per-block keep fractions (sparsesam family only)
#   BATCH_SIZES    space-separated batch sizes
#   NUM_SAMPLES    images per dataset (0 = full val set)
#   MODEL          SAM3 hf model id
#   TEXT_PROMPT    open-vocab text prompt (default "object")
#   AMP            float16 (default) | bfloat16 | float32
#   MLP_MERGE      yes (default) | no
#   SPARSE_ATTN    no (default) | yes  (flex_attention path, requires torch ≥ 2.5)
#   DATASET_IDX    space-separated dataset indices (default: all)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RATIOS=${RATIOS:-"0.5 0.3"}
BATCH_SIZES=${BATCH_SIZES:-"1"}
NUM_SAMPLES=${NUM_SAMPLES:-100}
MODEL=${MODEL:-facebook/sam3}
CHECKPOINT=${CHECKPOINT:-}
TEXT_PROMPT=${TEXT_PROMPT:-object}
AMP=${AMP:-float16}
MLP_MERGE=${MLP_MERGE:-yes}
SPARSE_ATTN=${SPARSE_ATTN:-no}
DATASET_IDX=${DATASET_IDX:-}

EXTRA=""
case "$MLP_MERGE" in
  no|false|0) EXTRA="$EXTRA --no-mlp-merge" ;;
esac
case "$SPARSE_ATTN" in
  yes|true|1) EXTRA="$EXTRA --sparse-attn" ;;
esac
DSI_FLAG=""
[ -n "$DATASET_IDX" ] && DSI_FLAG="--dataset-idx $DATASET_IDX"
CKPT_FLAG=""
[ -n "$CHECKPOINT" ] && CKPT_FLAG="--checkpoint $CHECKPOINT"

python "$SCRIPT_DIR/eval_sam3_hq44k.py" \
    --model         "$MODEL" \
    --text-prompt   "$TEXT_PROMPT" \
    --ratios        $RATIOS \
    --batch-sizes   $BATCH_SIZES \
    --num-samples   "$NUM_SAMPLES" \
    --amp-dtype     "$AMP" \
    $CKPT_FLAG \
    $EXTRA \
    $DSI_FLAG

# Examples:
#   bash tasks/sam_hq44k/eval_sam3_hq44k.sh                                  # default
#   NUM_SAMPLES=0 BATCH_SIZES=2 bash tasks/sam_hq44k/eval_sam3_hq44k.sh      # full val
#   DATASET_IDX=0 NUM_SAMPLES=200 bash tasks/sam_hq44k/eval_sam3_hq44k.sh    # DIS5K-VD only
#   SPARSE_ATTN=yes RATIOS="0.5 0.3 0.2" bash tasks/sam_hq44k/eval_sam3_hq44k.sh
#   MLP_MERGE=no  bash tasks/sam_hq44k/eval_sam3_hq44k.sh                    # baseline-equiv
