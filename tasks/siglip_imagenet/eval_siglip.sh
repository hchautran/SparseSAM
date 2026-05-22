#!/usr/bin/env bash
# SigLIP / SigLIP 2 zero-shot ImageNet wrapper.
#
# Env vars (with defaults):
#   MODEL         HF model id (default: google/siglip2-so400m-patch14-384)
#   DATA_ROOT     ImageNet-1k root (default: ./tasks/pe_imagenet/data/imagenet)
#   BATCH         Batch size (default: 32)
#   DTYPE         fp32|fp16|bf16 (default: bf16)
#   OUTPUT_DIR    CSV destination (default: ./benchmark_results)

set -eu

MODEL=${MODEL:-google/siglip2-so400m-patch14-384}
DATA_ROOT=${DATA_ROOT:-./tasks/pe_imagenet/data/imagenet}
BATCH=${BATCH:-32}
DTYPE=${DTYPE:-bf16}
OUTPUT_DIR=${OUTPUT_DIR:-./benchmark_results}

python tasks/siglip_imagenet/eval_siglip.py \
    --model "$MODEL" \
    --dataset-root "$DATA_ROOT" \
    --batch-size "$BATCH" \
    --dtype "$DTYPE" \
    --output-dir "$OUTPUT_DIR"
