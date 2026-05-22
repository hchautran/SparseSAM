---
license: apache-2.0
---

<br />
<p align="center">
  <h1 align="center">SparseSAM: Structured Sparsification of Activations<br/>in Segment Anything Models</h1>

  <p align="center">
    <a href="https://scholar.google.com/citations?user=FZH2vcEAAAAJ&hl=en"><strong>Hoai-Chau Tran</strong></a>
    ·
    <strong>Chi H. Nguyen</strong>
    ·
    <a href="https://duyhominhnguyen.github.io/"><strong>Duy M. H. Nguyen</strong></a>
    ·
    <a href="https://www.matlog.net/"><strong>Mathias Niepert</strong></a>
    ·
    <a href="https://www.matlog.net/"><strong>Fan Lai</strong></a>
    ·
    <a href="https://www.matlog.net/"><strong>Khoa D Doan</strong></a>
  </p>

  <p align="center">
    <a href="https://arxiv.org/abs/2605.17633">
      <img src="https://img.shields.io/badge/arXiv-2605.17633-b31b1b.svg" alt="arXiv"></a>
    <a href="./2605.17633v1.pdf">
      <img src="https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green" alt="Paper PDF"></a>
    <a href="https://www.python.org/downloads/release/python-3120/">
      <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python"></a>
    <a href="https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg?logo=pytorch" alt="PyTorch"></a>
    <a href="https://opensource.org/licenses/Apache-2.0">
      <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  </p>
</p>
<br />

<p align="center">
  <img src="image.png" width="100%"
       alt="Segmentation outputs of baseline / ToMe / SpargeAttn / SparseSAM at density 0.3">
</p>

This repository contains the official PyTorch implementation of [SparseSAM](./2605.17633v1.pdf), a training-free framework that accelerates the Segment Anything Model (SAM) by **2×** with **2.8× memory reduction** and only **<1% IoU loss**.  All algorithm implementations live under [`algos/`](algos/) and can be patched on top of the original checkpoints without retraining.

## Table of Contents

- [Abstract](#abstract)
- [Folder layout](#folder-layout)
- [Installation](#installation)
- [Supported tasks](#supported-tasks)
  - [SAM HQ-44K segmentation](#sam-hq-44k-segmentation)
  - [SAM MS-COCO box-prompted](#sam-ms-coco-box-prompted)
  - [Perception Encoder ImageNet zero-shot](#perception-encoder-imagenet-zero-shot)
- [Results](#results)
- [Profiling](#profiling)
- [Adding a new algorithm](#adding-a-new-algorithm)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Abstract

The Segment Anything Model (SAM) achieves strong open-vocabulary segmentation,
but its ViT-based image encoders dominate inference latency and memory. Existing
activation-compression methods such as token merging reduce token length yet
introduce non-trivial runtime overhead and suffer catastrophic quality drops
under high compression. Sparse-attention methods, on the other hand, focus on
attention alone and leave the MLP fully dense, capping achievable speedup.

We propose **SparseSAM**, a *training-free structured sparsification* framework
that jointly accelerates attention **and** MLP layers while preserving token
identity. SparseSAM introduces:

- **Stripe-Sort Attention** — a deterministic Z-order permutation that
  transforms dense attention into a static hardware-friendly sparse pattern,
  eliminating dynamic masking overhead.
- **Residual-Consistency MLP** — routes only informative tokens through the
  MLP while propagating remaining tokens through the residual pathway.

Across four segmentation benchmarks SparseSAM loses only **0.004 mIoU** at 0.4
density and **0.021 mIoU** at 0.3 — a **2.10× reduction in accuracy loss**
versus token-merging — while delivering **2× faster inference and 2.8× memory
reduction**.

---

## Folder layout

```
algos/                          # all algorithm code + vendored upstream models
├── registry.py                   unified AlgoSpec + register() for SAM / PE
├── tome/                         Token Merging (bipartite soft matching)
├── gradtome/                     Gradient-aware bipartite matching (StructSAM)
├── sparsesam/                    SparseSAM Stripe-Sort attn + Residual-Consistency MLP
├── sparge/                       SpargeAttn drop-in sparse attention (integration layer)
├── kernels/                      fused cutlass-DSL CUDA kernels (FA2 + rel-pos / RoPE)
└── 3rd_party/                    upstream model sources (vendored submodules)
    ├── sam-hq/                     SAM-HQ model + predictor + train pipeline
    ├── perception_models/          Meta's Perception Encoder source
    ├── SpargeAttn/                 SpargeAttn block-sparse attention kernels (pip install -e)
    └── lmms-eval/                  (unused; kept for archival)

tasks/                          # eval / profile entry points, grouped by task
├── sam_hq44k/                    SAM-HQ on HQ-44K 
├── sam_coco/                     SAM-HQ on MS-COCO val2017 with GT-box prompts
├── sam_profile/                  SAM per-component / per-attn-layer profilers
├── pe_imagenet/                  PE zero-shot CLIP eval + per-block profiler

utils/                          # shared data loading + benchmark helpers
docs/                           # contributor docs — start here when adding an algo
benchmark_results/              # CSV outputs + saved plots
ckts/  sam2_ckts/  sam2_configs/   # checkpoints + configs
data/                              # DIS5K, thin_object_detection, coco, imagenet, …
```

All compression algorithms are runtime patches: they monkey-patch the encoder's
transformer blocks at apply time and revert cleanly, so the original checkpoints
stay unchanged and a single eval run can sweep several `(algo, ratio)` configs
back-to-back. Each task ships both a `*.py` entry point and a `*.sh` wrapper;
most knobs (model, batch size, algos, ratios) are env-overridable from the wrapper.

---

## Installation

Maintainer env: **Python 3.12**, **PyTorch 2.5.1 + CUDA 12.1**, **NVIDIA A100**.
The code runs on Python 3.10–3.12; pick whichever matches your CUDA toolchain.

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo-url> SAM_Quantization
cd SAM_Quantization
# (or, if already cloned)
git submodule update --init --recursive

# 2. Env + PyTorch (must match your CUDA)
conda create -n sam python=3.12 -y && conda activate sam
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Repo + extra deps
pip install -e .
pip install -r requirements.txt

# 4. Vendored submodules that ship as Python packages
pip install -e algos/3rd_party/sam-hq
pip install -e algos/3rd_party/perception_models
```

Optional / kernel deps:

```bash
# Flash-Attention 2 (used by some PE patches + the FA2+RoPE fused kernel)
pip install flash-attn==2.8.3 --no-build-isolation

# xFormers (memory-efficient attention; required by perception_models)
pip install xformers==0.0.35

# SpargeAttn — CUDA extension. The HF `kernels` package can break setuptools
# egg_info on some envs; if `pip install -e` fails, run setup.py develop directly:
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 python algos/3rd_party/SpargeAttn/setup.py develop
```

Expected layout for data and checkpoints:

```
ckts/                            SAM-HQ: sam_hq_vit_{t,b,l,h}.pth
sam2_ckts/                       SAM2 / SAM2.1
data/DIS5K/                      high-detail segmentation
data/thin_object_detection/      COIFT, HRSOD, ThinObject5K
data/coco/                       COCO val2017 + annotations
data/imagenet/                   ImageNet1k for PE zero-shot eval
```

---

## Supported tasks

All registered algorithms are accessed through one unified registry in [`algos/registry.py`](algos/registry.py). Apply a patch with three lines:

```python
from segment_anything import sam_model_registry
from algos.registry import apply_sam, remove_all_sam

sam = sam_model_registry["vit_l"](checkpoint="./ckts/sam_hq_vit_l.pth")
apply_sam(sam.image_encoder, name="sparsesam", ratio=0.5)   # density 50%
# ... run inference ...
remove_all_sam(sam.image_encoder)                            # revert to baseline
```

`apply_pe`, `apply_siglip`, `apply_mvit` (+ matching `remove_all_*`) follow the same shape. The registry advertises every algorithm: `sparsesam`, `sparsesam_pitome`, `sparsesam_random`, `tome`, `pitome`, `gradtome`, `gradtome_pitome`, `gradtome_hilbert`, `sparge`. See [`docs/ADDING_ALGORITHMS.md`](docs/ADDING_ALGORITHMS.md) for adding new ones.

> **Interactive demo:** [`notebooks/sparsesam_demo.ipynb`](notebooks/sparsesam_demo.ipynb) — applies SparseSAM on a single image, sweeps density, and runs a per-block profile (attention vs MLP, windowed vs global) with side-by-side mask plots.

### SAM HQ-44K segmentation

High-fidelity segmentation on DIS5K-VD, COIFT, ThinObject5K-TE, HRSOD (HQ-44K). Patches `model.image_encoder` (SAM-HQ ViT) or `model.image_encoder.trunk` (SAM2.1 Hiera).

```python
from algos.registry import apply_sam
apply_sam(sam.image_encoder, "sparsesam", ratio=0.5, mlp_merge=True)
```

Sweep CLI:

```bash
python tasks/sam_hq44k/eval_hq44k.py \
    --algos none sparsesam tome gradtome sparge \
    --ratios 0.25 0.50 0.75 \
    --batch-sizes 1 --num-samples 470 \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
    --dataset-idx 0 1
# or the wrapper (env-overridable knobs):
ALGOS="sparsesam tome" RATIOS="0.5" sh tasks/sam_hq44k/eval_hq44k.sh
```

Reports **mIoU**, **Boundary IoU**, throughput, encoder latency, peak GPU memory. CSVs land in `benchmark_results/`.

### SAM MS-COCO box-prompted

Zero-shot box-prompted segmentation on COCO val2017 (GT boxes, or detections from DINO / H-DETR / YOLOX).

> **Note** — `tasks/sam_coco/eval_coco.py` is referenced by the paper but **not yet ported into this repo**; the directory currently ships empty. Add the entry-point script following the pattern of `tasks/sam_hq44k/eval_hq44k.py` to enable this task.

### Perception Encoder ImageNet zero-shot

Zero-shot CLIP on ImageNet-1k (and CIFAR10/100, ImageNet-V2, MS-COCO retrieval) with PE-Core-B16 / L14-336. Patches `model.visual` via the partial-token-count variants.

```python
import core.vision_encoder.pe as pe
from algos.registry import apply_pe

model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
apply_pe(model.visual, "sparsesam_partial",
         ratio=0.5, group_size=4, start_block=0, mlp_merge=True)
```

Sweep CLI:

```bash
ALGOS_SWEEP="sparsesam_partial tome_partial" RATIOS_SWEEP="0.25 0.5 0.75" \
    sh tasks/pe_imagenet/eval_pe.sh
# or directly
python tasks/pe_imagenet/eval_pe_clip.py --model PE-Core-L14-336 \
    --dataset imagenet1k --dataset-root ./data/imagenet \
    --algorithm sparsesam_partial --ratio 0.5
```

Reports **Top-1 / Top-5 accuracy** plus the timing/memory triple.

---

## Example Results

All numbers were **measured in this repo** (not copied from the paper) on
**NVIDIA A100X-20C (sm80)** · PyTorch 2.5.1 + CUDA 12.1 · SAM-HQ ViT-L ·
batch=8 · full datasets (470 imgs DIS5K-VD, 500 imgs ThinObject5K-TE).

### Dense baseline

| Dataset           |   mIoU |  B-IoU | enc/img | Peak GPU |
|---                |   ---: |   ---: |    ---: |     ---: |
| DIS5K-VD          | 0.7863 | 0.7060 | 55.7 ms | 14130 MB |
| ThinObject5K-TE   | 0.8956 | 0.7962 | 52.9 ms | 14130 MB |

ΔmIoU below is absolute vs the dense baseline. Speedup is
`baseline_enc_ms / algo_enc_ms` (>1 = faster, **bold** = the best speedup in column).

### SparseSAM — Stripe-Sort attention (± Residual-Consistency MLP)

Ablation: attention-only (Stripe-Sort) vs. full SparseSAM (Stripe-Sort + Residual-Consistency MLP).

| Density | Variant            |    DIS5K-VD mIoU (Δ) |   Speedup |    ThinObj mIoU (Δ) |   Speedup | Peak GPU |
|---:     | :---               |                 ---: |      ---: |                ---: |      ---: |     ---: |
|     30% | attn only          |      0.7718 (−0.015) |     ×1.71 |     0.8907 (−0.005) |     ×1.66 |  2271 MB |
|     30% | attn + MLP (full)  |      0.7500 (−0.036) | **×2.04** |     0.8756 (−0.020) | **×2.02** |  2272 MB |
|     50% | attn only          |      0.7838 (−0.003) |     ×1.64 |     0.8941 (−0.001) |     ×1.59 |  2277 MB |
|     50% | attn + MLP (full)  |      0.7756 (−0.011) | **×1.89** |     0.8921 (−0.003) | **×1.85** |  2277 MB |
|     70% | attn only          |  **0.7847 (−0.002)** |     ×1.61 | **0.8965 (+0.001)** |     ×1.56 |  2282 MB |
|     70% | attn + MLP (full)  |      0.7819 (−0.004) | **×1.78** |     0.8960 (+0.000) | **×1.73** |  2282 MB |

Toggle via `mlp_merge=True/False` in `apply_sam(..., name="sparsesam", mlp_merge=...)`
or `--mlp-merge` / `--no-mlp-merge` on the CLI.

The full Residual-Consistency MLP path adds another **~0.2× speedup** on top
of attention-only at every density. Quality cost is small at high density
(−0.003 mIoU at 70%) and grows with compression (−0.022 mIoU on DIS5K-VD at
30%). Either variant is the only algorithm here that simultaneously drops
both latency and memory — **~2× encoder speedup with ~84% less GPU memory**
vs the dense baseline.

### ToMe — bipartite soft matching (attn + MLP)

| Density |  DIS5K-VD mIoU (Δ) |       ThinObj mIoU (Δ) | Speedup (DIS / Thin) | Peak GPU |
|---:     |               ---: |                   ---: |                 ---: |     ---: |
|     30% |    0.6970 (−0.089) |        0.8428 (−0.053) |         ×0.79 / ×0.82 | 12129 MB |
|     50% |    0.6970 (−0.089) |        0.8428 (−0.053) |         ×0.80 / ×0.82 | 12129 MB |
|     70% |    0.7650 (−0.021) |        0.8906 (−0.005) |         ×0.49 / ×0.50 | 13795 MB |

Merge + unmerge overhead exceeds the savings → encoder runs slower than
dense; memory drops only modestly.

### GradToMe / StructSAM — gradient-aware bipartite matching

| Density |  DIS5K-VD mIoU (Δ) |       ThinObj mIoU (Δ) | Speedup (DIS / Thin) | Peak GPU |
|---:     |               ---: |                   ---: |                 ---: |     ---: |
|     30% |    0.5608 (−0.226) |        0.7138 (−0.182) |         ×0.72 / ×0.74 | 12366 MB |
|     50% |    0.6850 (−0.101) |        0.8268 (−0.069) |         ×0.65 / ×0.65 | 14033 MB |
|     70% |    0.7561 (−0.030) |        0.8830 (−0.013) |         ×0.56 / ×0.58 | 15694 MB |

Catastrophic quality drop at low density (especially thin structures); also
runs slower than dense and at 70% density uses *more* GPU memory than baseline.

### SpargeAttn — top-k sparse attention (no token reduction)

| Density |  DIS5K-VD mIoU (Δ) |       ThinObj mIoU (Δ) | Speedup (DIS / Thin) | Peak GPU |
|---:     |               ---: |                   ---: |                 ---: |     ---: |
|     30% |    0.7350 (−0.051) |        0.8539 (−0.042) |         ×1.12 / ×1.13 | 14027 MB |
|     50% |    0.7709 (−0.015) |        0.8882 (−0.007) |         ×1.10 / ×1.13 | 14027 MB |
|     70% |    0.7713 (−0.015) |        0.8898 (−0.006) |         ×1.12 / ×1.11 | 14027 MB |

Quality close to dense and a modest ~1.1× speedup, but no token reduction
→ memory is essentially the dense baseline.

> **Reproduce:**
> ```bash
> python tasks/sam_hq44k/eval_hq44k.py --algos none sparsesam tome gradtome sparge \
>     --ratios 0.30 0.50 0.70 --batch-sizes 8 --num-samples 500 \
>     --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
>     --dataset-idx 0 1 --no-wandb --no-plot --mlp-merge
> # then for the attn-only ablation, swap --mlp-merge for --no-mlp-merge with --algos sparsesam
> ```
> Raw CSVs:
> [`benchmark_results/tome_benchmark_20260522_191155.csv`](benchmark_results/tome_benchmark_20260522_191155.csv) (main),
> [`benchmark_results/tome_benchmark_20260522_191816.csv`](benchmark_results/tome_benchmark_20260522_191816.csv) (no-MLP ablation).

### MS-COCO box-prompted segmentation

`tasks/sam_coco/eval_coco.py` is not yet ported into this repo (the directory
ships empty). Once added — following the pattern of `eval_hq44k.py` plus
`pycocotools` for mAP — this section will hold the SAM-B / SAM-L / SAM-H ×
density sweep.

### Perception Encoder ImageNet zero-shot

Blocked in this env: `perception_models` requires `timm.layers` (timm ≥0.6)
but the current conda env has `timm 0.4.12`. Run `pip install -U timm` and
then `sh tasks/pe_imagenet/eval_pe.sh` to populate this section.

---

## Profiling

| Target | Entry point | Wrapper |
|---|---|---|
| SAM encoder per-component | [profile_encoder.py](tasks/sam_profile/profile_encoder.py) | [profile.sh](tasks/sam_profile/profile.sh) |
| SAM per-attention-layer | [profile_attn_layers.py](tasks/sam_profile/profile_attn_layers.py) | — |
| PE per-block latency | [profile_pe.py](tasks/pe_imagenet/profile_pe.py) | [profile_pe.sh](tasks/pe_imagenet/profile_pe.sh) |

```bash
# SAM-HQ encoder, baseline vs. patched
python tasks/sam_profile/profile_encoder.py --version sam1 \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
    --tome-algo sparsesam --tome-ratio 0.5

# PE per-block
python tasks/pe_imagenet/profile_pe.py --tome-algo sparsesam_partial --tome-ratio 0.5
```

---

## Adding a new algorithm

The contributor docs in [docs/](docs/) cover this end-to-end:

* **[docs/ADDING_ALGORITHMS.md](docs/ADDING_ALGORITHMS.md)** — overview: how the
  models works, file layout under
  [algos/](algos/), naming conventions (`pe_compress.py` / `pe_partial.py` /
  `sam.py` / `merge.py`), and which doc to read for each backbone.
* **[docs/ADDING_SAM.md](docs/ADDING_SAM.md)** — SAM patches: the
  subclass-and-swap template, three-step patch → register → run example,
  smoke test, and gotchas.
* **[docs/ADDING_PE.md](docs/ADDING_PE.md)** — PE patches: both flavors
  (stage-compression and partial / full-token-count), the `_pe_stage.py`
  base classes, `kwargs_from_args` builders, sweep + plot.

Once registered, your algorithm appears as a choice in `--algos` /
`--algorithm` for every eval and profile script automatically — no
changes to the entry-point scripts needed.

---

## Citation

```bibtex
@article{tran2026sparsesam,
  title   = {SparseSAM: Structured Sparsification of Activations in Segment Anything Models},
  author  = {Tran, Hoai-Chau and Nguyen, Chi H. and Nguyen, Duy M. H. and
             Niepert, Mathias and Lai, Fan and Doan, Khoa D.},
  journal = {arXiv preprint arXiv:2605.17633},
  year    = {2026}
}
```

For correspondence: tranhoaichau.00@gmail.com, chauht2@illinois.edu

---

## Acknowledgement

This work builds on:
- **[SAM-HQ](https://github.com/SysCV/sam-hq)** — high-quality SAM checkpoints, predictor, and HQ-44K training pipeline.
- **[ToMe](https://arxiv.org/abs/2210.09461)** — bipartite-soft-matching token merging; baseline + the file layout that `algos/` follows.
- **[PiToMe](https://github.com/hchautran/PiToMe)** (NeurIPS 2024) — sister project, energy-margin variant of ToMe; the registry + per-algo file conventions in this repo are direct descendants.
- **[SpargeAttn](https://github.com/thu-ml/SpargeAttn)** — top-k attention-mass sparsification kernel, integrated as the `sparge` baseline.
- **[StructSAM (GradToMe)](https://arxiv.org/abs/2603.07307)** — gradient-aware bipartite matching variant, integrated as the `gradtome` baseline.
- **[Perception Encoder](https://github.com/facebookresearch/perception_models)** — Meta's PE backbone used for the ImageNet / SigLIP / VQA evaluation tracks.
