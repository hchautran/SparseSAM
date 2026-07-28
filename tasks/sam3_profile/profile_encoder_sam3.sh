#!/usr/bin/env bash
# Profile the vendored SAM3 ViT (mock weights so no HF auth is needed) and
# compare baseline against the SparseSAM MLP-merge patch.
set -euo pipefail

cd "$(dirname "$0")"

python profile_encoder_sam3.py \
    --mock \
    --batch-size 1 \
    --img-size 1008 \
    --n-warmup 3 \
    --n-runs 10 \
    --dtype bf16 \
    --algo sparsesam \
    --ratio 0.5
