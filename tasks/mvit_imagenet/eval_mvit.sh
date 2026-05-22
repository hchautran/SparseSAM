#!/usr/bin/env bash
# timm ImageNet eval wrapper.
#
# Env vars (with defaults):
#   MODEL         timm model id (default: mvitv2_tiny.fb_in1k)
#   VAL_ROOT      ImageNet val dir (default: ./tasks/pe_imagenet/data/imagenet/val)
#   BATCH         Batch size (default: 64)
#   DTYPE         fp32|fp16|bf16 (default: bf16)
#   OUTPUT_DIR    CSV destination (default: ./benchmark_results)

set -eu

MODEL=${MODEL:-mvitv2_tiny.fb_in1k}
VAL_ROOT=${VAL_ROOT:-./tasks/pe_imagenet/data/imagenet/val}
BATCH=${BATCH:-64}
DTYPE=${DTYPE:-bf16}
OUTPUT_DIR=${OUTPUT_DIR:-./benchmark_results}

python tasks/mvit_imagenet/eval_mvit.py \
    --model "$MODEL" \
    --val-root "$VAL_ROOT" \
    --batch-size "$BATCH" \
    --dtype "$DTYPE" \
    --output-dir "$OUTPUT_DIR"
