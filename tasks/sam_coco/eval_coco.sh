#!/bin/bash
# SAM-HQ COCO evaluation sweep.
#
# Expected:
#   - SAM-HQ checkpoints available under CKPT_ROOT
#   - COCO-style dataset root with val2017/ and annotations/
#   - PTQ4SAM dependencies available through SAM_QUANT_ROOT
#
# Required env vars:
#   SAM_QUANT_ROOT  path to the directory that contains PTQ4SAM/
#   DATA_ROOT       path to the COCO dataset root
#   CKPT_ROOT       path to detector and SAM checkpoints
#
# Optional knobs (env-overridable):
#   ALGOS          space-separated algos
#   RATIOS         per-block keep fractions
#   BATCH_SIZES    space-separated batch sizes
#   NUM_SAMPLES    optional image limit
#   MODEL_TYPE     vit_l / vit_b / vit_h
#   DETECTOR       dino / yolox / hdetr
#   MLP_MERGE      "yes" (default) or "no"
#   MODEL_CKT      explicit SAM-HQ checkpoint path
#   DET_CKPT       explicit detector checkpoint path
#   DET_SAM_CKT    original SAM checkpoint used to bootstrap the detector wrapper
#   OUT_DIR        output directory for CSV results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SAM_QUANT_ROOT="${SAM_QUANT_ROOT:-<path-to-ptq4sam-parent>}"
DATA_ROOT="${DATA_ROOT:-<path-to-coco-data>}"
CKPT_ROOT="${CKPT_ROOT:-<path-to-checkpoints>}"
ALGOS="${ALGOS:-none sparsesam gradtome tome sparge piecewise}"
RATIOS="${RATIOS:-0.70 0.50 0.30}"
BATCH_SIZES="${BATCH_SIZES:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
MODEL_TYPE="${MODEL_TYPE:-vit_l}"
DETECTOR="${DETECTOR:-dino}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-0}"
ENCODER_WARMUP_BATCHES="${ENCODER_WARMUP_BATCHES:-3}"
MLP_MERGE="${MLP_MERGE:-yes}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/benchmark_results/sam_coco}"

case "${MODEL_TYPE}" in
  vit_b) DEFAULT_MODEL_CKT="${CKPT_ROOT}/sam_hq_vit_b.pth" ;;
  vit_h) DEFAULT_MODEL_CKT="${CKPT_ROOT}/sam_hq_vit_h.pth" ;;
  vit_l) DEFAULT_MODEL_CKT="${CKPT_ROOT}/sam_hq_vit_l.pth" ;;
  *) echo "Unsupported MODEL_TYPE=${MODEL_TYPE}; expected vit_b, vit_l, or vit_h" >&2; exit 2 ;;
esac
MODEL_CKT="${MODEL_CKT:-${DEFAULT_MODEL_CKT}}"

case "${DETECTOR}" in
  dino) DEFAULT_DET_CKPT="${CKPT_ROOT}/focalnet_l_dino.pth" ;;
  hdetr) DEFAULT_DET_CKPT="${CKPT_ROOT}/r50_hdetr.pth" ;;
  yolox) DEFAULT_DET_CKPT="${CKPT_ROOT}/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth" ;;
  *) echo "Unsupported DETECTOR=${DETECTOR}; expected dino, yolox, or hdetr" >&2; exit 2 ;;
esac
DET_CKPT="${DET_CKPT:-${DEFAULT_DET_CKPT}}"

case "${MODEL_TYPE}" in
  vit_l) DEFAULT_DET_SAM_CKT="${CKPT_ROOT}/sam_vit_l_0b3195.pth" ;;
  *) DEFAULT_DET_SAM_CKT="none" ;;
esac
DET_SAM_CKT="${DET_SAM_CKT:-${DEFAULT_DET_SAM_CKT}}"

if [ "$MLP_MERGE" = "no" ] || [ "$MLP_MERGE" = "false" ] || [ "$MLP_MERGE" = "0" ]; then
    MLP_FLAG="--no-mlp-merge"
else
    MLP_FLAG="--mlp-merge"
fi

case "${SAM_QUANT_ROOT}" in
  *"<path-to-ptq4sam-parent>"*|"")
    echo "Set SAM_QUANT_ROOT to the directory that contains PTQ4SAM/." >&2
    exit 2
    ;;
esac
case "${DATA_ROOT}" in
  *"<path-to-coco-data>"*|"")
    echo "Set DATA_ROOT to the COCO dataset root." >&2
    exit 2
    ;;
esac
case "${CKPT_ROOT}" in
  *"<path-to-checkpoints>"*|"")
    echo "Set CKPT_ROOT to the checkpoint directory, or override MODEL_CKT / DET_CKPT / DET_SAM_CKT explicitly." >&2
    exit 2
    ;;
esac

export SAM_QUANT_ROOT

if [ -f "${SAM_QUANT_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${SAM_QUANT_ROOT}/.venv/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/algos/3rd_party/sam-hq:${SAM_QUANT_ROOT}:${SAM_QUANT_ROOT}/PTQ4SAM:${SAM_QUANT_ROOT}/PTQ4SAM/projects/instance_segment_anything/ops:${PYTHONPATH:-}"

python "$SCRIPT_DIR/eval_coco.py" \
    --sam-quant-root "${SAM_QUANT_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --model-type "${MODEL_TYPE}" \
    --model-ckt "${MODEL_CKT}" \
    --detector "${DETECTOR}" \
    --det-checkpoint "${DET_CKPT}" \
    --det-sam-ckt "${DET_SAM_CKT}" \
    --algos ${ALGOS} \
    --ratios ${RATIOS} \
    --batch-sizes ${BATCH_SIZES} \
    --num-samples "${NUM_SAMPLES}" \
    --workers-per-gpu "${WORKERS_PER_GPU}" \
    --encoder-warmup-batches "${ENCODER_WARMUP_BATCHES}" \
    --output-dir "${OUT_DIR}" \
    "$MLP_FLAG"
