#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAM_QUANT_ROOT="${SAM_QUANT_ROOT:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization}"
export SAM_QUANT_ROOT

source "${SAM_QUANT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/algos/3rd_party/sam-hq:${SAM_QUANT_ROOT}:${SAM_QUANT_ROOT}/PTQ4SAM:${SAM_QUANT_ROOT}/PTQ4SAM/projects/instance_segment_anything/ops:${PYTHONPATH:-}"

python "${REPO_ROOT}/tasks/sam_coco/eval_coco.py" \
  --sam-quant-root "${SAM_QUANT_ROOT}" \
  --detector "${DETECTOR:-dino}" \
  --model-type "${MODEL_TYPE:-vit_l}" \
  --model-ckt "${MODEL_CKT:-${SAM_QUANT_ROOT}/ckts/sam_hq_vit_l.pth}" \
  --det-sam-ckt "${DET_SAM_CKT:-${SAM_QUANT_ROOT}/ckts/sam_vit_l_0b3195.pth}" \
  --algos ${ALGOS:-none sparsesam tome gradtome sparge piecewise} \
  --ratios ${RATIOS:-0.75} \
  --batch-sizes ${BATCH_SIZES:-1} \
  --workers-per-gpu "${WORKERS_PER_GPU:-2}" \
  --profile-warmup-calls "${PROFILE_WARMUP_CALLS:-10}" \
  ${NUM_SAMPLES:+--num-samples ${NUM_SAMPLES}} \
  ${DATA_ROOT:+--data-root ${DATA_ROOT}} \
  ${CONFIG:+--config ${CONFIG}} \
  ${DET_CHECKPOINT:+--det-checkpoint ${DET_CHECKPOINT}} \
  ${PROFILE_IMAGE_ENCODER:+--profile-image-encoder} \
  "$@"
