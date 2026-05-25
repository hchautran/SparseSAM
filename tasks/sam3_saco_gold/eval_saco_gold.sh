#!/usr/bin/env bash
# Run one SA-Co/Gold subset, baseline first then with the SparseSAM patch.
# Edit the paths below to match your local data layout.
#
# Prereqs (one-time):
#   hf auth login                                  # accept facebook/sam3 license first
#   download SACo-Gold annotations + images        # see SAM3 README scripts/eval/gold
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

# ── paths (EDIT THESE) ────────────────────────────────────────────────────────
SACO_ANN=/data/SACo-Gold/annotations           # *.json from facebook/SACo-Gold
SACO_METACLIP_IMGS=/data/SACo-Gold/images_metaclip
SACO_SA1B_IMGS=/data/SACo-Gold/images_sa1b     # only needed for sa1b_nps
LOG_DIR=./sam3_logs/saco_gold
# Optional local ckpt; leave empty to auto-download from HF.
CKPT=""

# ── what to run ───────────────────────────────────────────────────────────────
SUBSET="${1:-metaclip_nps}"   # metaclip_nps | sa1b_nps | attributes | crowded |
                              # wiki_common | fg_food | fg_sports
RATIO="${2:-0.5}"             # 1.0 = baseline (no patch); 0<r<1 = MLP-merge

ckpt_arg=()
[ -n "$CKPT" ] && ckpt_arg=(--checkpoint-path "$CKPT")

img_args=(--metaclip-img-path "$SACO_METACLIP_IMGS")
[ "$SUBSET" = "sa1b_nps" ] && img_args+=(--sa1b-img-path "$SACO_SA1B_IMGS")

# Baseline (ratio=1.0 disables the patch entirely).
python tasks/sam3_saco_gold/eval_saco_gold.py \
    --subset "$SUBSET" \
    --base-annotation-path    "$SACO_ANN" \
    --base-experiment-log-dir "$LOG_DIR/baseline_${SUBSET}" \
    "${img_args[@]}" "${ckpt_arg[@]}" \
    --num-gpus 1 \
    --ratio 1.0

# Patched.
python tasks/sam3_saco_gold/eval_saco_gold.py \
    --subset "$SUBSET" \
    --base-annotation-path    "$SACO_ANN" \
    --base-experiment-log-dir "$LOG_DIR/sparsesam_r${RATIO}_${SUBSET}" \
    "${img_args[@]}" "${ckpt_arg[@]}" \
    --num-gpus 1 \
    --ratio "$RATIO"
