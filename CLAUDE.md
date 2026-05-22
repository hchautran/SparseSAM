# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for quantization and optimization of Segment Anything Models (SAM-1, SAM-HQ, SAM-2, SAM-2.1) plus Perception Encoder / SigLIP / MViTv2. The main techniques explored are token merging (ToMe/PiToMe/SparseSAM/GradToMe) and quantization to reduce compute and memory while maintaining segmentation quality. All algorithm patches live in the [algos/](algos/) package at the repo root (cute kernels under [algos/kernels/](algos/kernels/)); evaluation/profile entry points live under [tasks/](tasks/).

## Common Commands

Entry points live under `tasks/`; each ships both a `*.py` and a `*.sh` wrapper.

### Run token merging benchmark (SAM-HQ on HQ44K)
```bash
python tasks/sam_hq44k/eval_hq44k.py \
    --algos none tome pitome sparsesam \
    --ratios 0.9 0.8 0.7 \
    --batch-sizes 1 2 4 \
    --model-ckt ./ckts/sam_hq_vit_l.pth \
    --model-type vit_l \
    --num-samples 100 --no-wandb
# or use the wrapper:
sh tasks/sam_hq44k/eval_hq44k.sh
```

### Profile encoder (SAM-HQ baseline vs. patched)
```bash
python tasks/sam_profile/profile_encoder.py --version sam1 \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l
python tasks/sam_profile/profile_encoder.py --version sam1 \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
    --tome-algo pitome --tome-ratio 0.8
```

### Evaluate SAM2 on HQ44k
```bash
python tasks/sam_hq44k/eval_sam2_hq44k.py \
    --model-cfg ./sam2_configs/sam2.1/sam2.1_hiera_b+.yaml \
    --checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt \
    --num-samples 100
# or use the wrapper:
sh tasks/sam_hq44k/eval_sam2_hq44k.sh
```

## Architecture

### In-repo packages
- **`algos/`** — Token compression + attention patches (ToMe, PiToMe, SparseSAM, GradToMe, SpargeAttn). Unified registry in `algos/registry.py` with one `AlgoSpec` per `(backbone, name)`. Public API: `apply_pe / apply_sam / apply_siglip / apply_mvit` and matching `remove_all_*`. Cute kernels live under `algos/kernels/`.
- **`tasks/`** — Eval / profile entry points, grouped by task (sam_hq44k, sam_profile, pe_imagenet, siglip_imagenet, mvit_imagenet).
- **`sam_engine.py` / `data_utils.py`** — Shared SAM-HQ data pipeline + default dataset configs.

### Submodules (vendored under `algos/3rd_party/`)
- **`algos/3rd_party/sam-hq/`** — SAM-HQ model with training support; provides `SamPredictor`, checkpoint loading, and mask decoder.
- **`algos/3rd_party/perception_models/`** — Meta's Perception Encoder (PE) source.
- **`algos/3rd_party/SpargeAttn/`** — SpargeAttn block-sparse attention kernels.
- **`algos/3rd_party/lmms-eval/`** — VQA eval harness (currently unused).

### Adding a new algorithm
Write the patch in `algos/<myalgo>/`, then add one `register(AlgoSpec(...))` call in `algos.registry._register_builtins`. See `docs/ADDING_ALGORITHMS.md` for the full walkthrough.

### Token Merging Pattern
All algorithms follow the same three-step flow injected around every transformer block:
1. **Merge** — reduce token count via bipartite matching (metric varies by algorithm)
2. **MLP forward** — run on reduced token set (main speedup)
3. **Unmerge** — expand tokens back to full count

Key parameters:
- `--tome-ratio` (0–1): fraction of tokens to keep; lower = more compression
- `--tome-algo`: `none | tome | pitome | sparsesam | gradtome`
- `--tome-margin`: energy threshold for PiToMe

### Data Pipeline
```
Dataset (DIS5K / ThinObject5K / CascadePSP)
  → OnlineDataset (data_utils.py)
  → DataLoader
  → SAM model (optionally patched with ToMe)
  → SamPredictor / SAM2ImagePredictor
  → compute_iou / compute_boundary_iou
  → wandb + benchmark_results/
```

### Datasets (expected under `/data/`)
- `/data/DIS5K/` — high-detail segmentation (train + validation)
- `/data/thin_object_detection/` — COIFT, HRSOD, ThinObject5K
- `/data/cascade_psp/` — salient object detection benchmarks

### Checkpoints
- SAM-HQ checkpoints: `./ckts/sam_hq_vit_{t,b,l,h}.pth`
- SAM2/2.1 checkpoints: `./sam2_ckts/`
- SAM2 model configs: `./sam2_configs/sam2.1/`

## Environment

- Python 3.10 (`.python-version`)
- Key deps: `torch`, `torchvision`, `transformers>=4.55.4`, `timm>=1.0.19`, `accelerate`, `flash-attn2` (via `kernels`), `wandb`, `pycocotools`
- Install: `pip install -e .` (uses `pyproject.toml`)

## Metrics

Benchmarks report: **mIoU**, **Boundary IoU**, **throughput (img/s)**, **latency (ms)**, **GPU memory (MB)**. Results saved to `benchmark_results/` as CSV and optionally logged to wandb (disable with `--no-wandb`).

## A few more notes
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.