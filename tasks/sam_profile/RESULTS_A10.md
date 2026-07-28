# SparseSAM benchmark on NVIDIA A10 (SM86)

SAM-HQ **ViT-L**, GPU: NVIDIA A10 (23 GB, cc 8.6), CUDA 12.8, torch 2.8.0+cu128,
nvidia-cutlass-dsl 4.6.1, cuda-python 12.9.0. Same scripts and settings as
[RESULTS_3090.md](RESULTS_3090.md) / [RESULTS_L4.md](RESULTS_L4.md).

The SparseSAM FA2 CUTE kernel (written for Ampere SM80) compiles and runs on the
A10's SM86 with no code change. Device-independent metrics — peak memory, GMAC,
FLOP saving — reproduce the 3090/L4 numbers **exactly**. Latency speedup lands
between the L4 and the 3090: **1.77×** at density 0.25 (vs 2.28× L4, 1.70× 3090),
because the A10 sits between them on the bandwidth-vs-compute axis and has a
*slower* dense baseline (130 ms vs 100 ms on the 3090), leaving more headroom.

## Headline (density ↔ speed ↔ memory ↔ FLOPs)

| density  | enc speedup | peak-mem saving | GFLOP saving |
|----------|-------------|-----------------|--------------|
| baseline | 1.00×       | 1.0×            | 1.0×         |
| 0.75     | 1.31×       | 2.9×            | 1.13×        |
| 0.50     | 1.50×       | 2.9×            | 1.30×        |
| 0.25     | 1.77×       | 2.9×            | 1.54×        |

(Memory saving grows to **7.2×** at batch=8. Speedup is encoder-only; full e2e
including pre/post-processing is **1.38–1.71×** at 1 prompt/image.)

Latency/memory/GFLOPs use random-init weights (valid — these don't depend on
trained weights); input 1024×1024, fp16, batch=1 unless noted.

## Encoder: latency / memory / GFLOPs (batch=1)

| algo      | density | latency ms (median) | speedup | peak mem MB | mem saving | GMAC   | FLOP saving |
|-----------|---------|---------------------|---------|-------------|-----------|--------|-------------|
| baseline  | —       | 130.4               | 1.00×   | 2221.5      | 1.0×      | 1487.9 | 1.0×        |
| sparsesam | 0.25    | 73.5                | 1.77×   | 765.3       | 2.9×      | 968.8  | 1.54×       |
| sparsesam | 0.50    | 87.1                | 1.50×   | 765.3       | 2.9×      | 1141.8 | 1.30×       |
| sparsesam | 0.75    | 99.7                | 1.31×   | 765.3       | 2.9×      | 1314.8 | 1.13×       |

## Batch=8

| algo      | density | latency ms (median) | speedup | peak mem MB | mem saving |
|-----------|---------|---------------------|---------|-------------|-----------|
| baseline  | —       | 981.8               | 1.00×   | 13766.2     | 1.0×      |
| sparsesam | 0.25    | 549.1               | 1.79×   | 1908.1      | 7.2×      |
| sparsesam | 0.50    | 648.3               | 1.51×   | 1908.1      | 7.2×      |
| sparsesam | 0.75    | 739.8               | 1.33×   | 1908.1      | 7.2×      |

Peak-memory figures (765.3 MB / 1908.1 MB) are byte-identical to the 3090 and L4
runs — allocator behaviour is device-independent.

## End-to-end vs encoder-only (batch=1, encoder+decoder)

**1 prompt/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 129.7  | 6.0    | 135.7  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 73.3   | 6.0    | 79.3   | 1.77×       | 1.71×       |
| sparsesam 0.50 | 86.5   | 6.0    | 92.4   | 1.50×       | 1.47×       |

**10 prompts/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 130.2  | 29.0   | 159.2  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 73.6   | 29.5   | 103.1  | 1.77×       | 1.54×       |
| sparsesam 0.50 | 86.8   | 29.5   | 116.3  | 1.50×       | 1.37×       |

## Full end-to-end, incl. pre/post-processing (batch=1, real images from `input_imgs/`)

**1 prompt/image** (ms per image, median of 10)
| algo           | preproc | encoder | decoder | postproc | d2h  | **e2e ms** | e2e spd | img/s |
|----------------|---------|---------|---------|----------|------|------------|---------|-------|
| baseline       | 6.48    | 128.92  | 5.34    | 0.10     | 0.20 | **141.7**  | 1.00×   | 7.1   |
| sparsesam 0.25 | 6.71    | 69.74   | 5.30    | 0.10     | 0.20 | **82.7**   | 1.71×   | 12.1  |
| sparsesam 0.50 | 6.93    | 80.04   | 5.35    | 0.10     | 0.20 | **93.3**   | 1.52×   | 10.8  |
| sparsesam 0.75 | 6.67    | 89.84   | 5.41    | 0.11     | 0.20 | **102.9**  | 1.38×   | 9.7   |

Baseline non-encoder floor: 12.30 ms/image (8.7% of e2e), Amdahl ceiling 11.5×.

**10 prompts/image**
| algo           | preproc | encoder | decoder | postproc | d2h  | **e2e ms** | e2e spd | img/s |
|----------------|---------|---------|---------|----------|------|------------|---------|-------|
| baseline       | 6.88    | 128.96  | 27.93   | 0.46     | 0.96 | **165.9**  | 1.00×   | 6.0   |
| sparsesam 0.25 | 6.88    | 70.08   | 28.61   | 0.45     | 0.95 | **107.6**  | 1.54×   | 9.3   |
| sparsesam 0.50 | 6.62    | 80.23   | 28.74   | 0.45     | 0.95 | **117.7**  | 1.40×   | 8.5   |
| sparsesam 0.75 | 6.72    | 89.68   | 28.90   | 0.46     | 0.95 | **127.4**  | 1.30×   | 7.9   |

Baseline non-encoder floor: 36.63 ms/image (22.1% of e2e), Amdahl ceiling 4.5×.

## Implementation gain vs algorithmic gain (fused-dense control)

ViT-L, batch=1, 1024², median of 20 iters:

| config                 | what it is                                  | lat ms | peak MB | vs A  | vs B  |
|------------------------|---------------------------------------------|--------|---------|-------|-------|
| **A** baseline         | stock attention (manual QKᵀ + rel-pos + PV) | 130.83 | 2221.5  | 1.00× | 0.87× |
| **B** fused-dense      | same CUTE kernel, dense, full MLP           | 113.65 | 765.3   | 1.15× | 1.00× |
| **C** attn-sparse 0.25 | A-shape sparse attention, full MLP          | 91.01  | 765.3   | 1.44× | 1.25× |
| **C** attn-sparse 0.50 | "                                           | 99.51  | 765.3   | 1.31× | 1.14× |
| **D** sparsesam 0.25   | + keep-token MLP (the full method)          | 74.07  | 765.3   | 1.77× | 1.53× |
| **D** sparsesam 0.50   | "                                           | 87.18  | 765.3   | 1.50× | 1.30× |
| **E** tome 0.25        | baseline algorithm, stock attention         | 194.50 | 1992.3  | 0.67× | 0.58× |
| **E** tome 0.50        | "                                           | 193.83 | 1992.3  | 0.67× | 0.59× |

```
sparsesam 0.25:  1.77x total  =  1.15x implementation  x  1.53x algorithmic
sparsesam 0.50:  1.50x total  =  1.15x implementation  x  1.30x algorithmic
```

The custom kernel is worth **1.15×**; the algorithm (token sparsification +
keep-token MLP) is worth **1.53×**. As on the 3090, the ToMe reference
implementation is slower than not merging at all (matching overhead dominates), so
latency comparisons against ToMe should be made on density/GMAC, not wall clock.

## Reproduce

```
# deps (Python 3.12, torch 2.8.0+cu128)
git submodule update --init algos/3rd_party/sam-hq
pip install fvcore opencv-python-headless timm einops pandas scikit-image pycocotools matplotlib
pip install "nvidia-cutlass-dsl==4.6.1" "cuda-python==12.9.0"   # cuda-python 12.x REQUIRED (13.x → cudaErrorInsufficientDriver on a CUDA-12.8 driver)
pip install -e algos/3rd_party/sam-hq --no-deps

export PYTHONPATH=$PWD
python tasks/sam_profile/bench_encoder_l4.py  --model-type vit_l --ratios 0.25 0.5 0.75
python tasks/sam_profile/bench_encoder_l4.py  --model-type vit_l --ratios 0.25 0.5 0.75 --batch-sizes 8 --iters 10 --warmup 3
python tasks/sam_profile/flops_encoder_l4.py  --model-type vit_l --ratios 0.25 0.5 0.75
python tasks/sam_profile/bench_e2e_l4.py      --model-type vit_l --ratios 0.25 0.5 --prompts 1 10
python tasks/sam_profile/bench_e2e_full.py    --model-type vit_l --ratios 0.25 0.5 0.75 --prompts 1 10
python tasks/sam_profile/bench_impl_ablation.py --model-type vit_l --ratios 0.25 0.5
```

Raw run logs: `tasks/sam_profile/a10_run/`.

Accuracy (mIoU on COIFT) is **not** included here — it needs the
`sam_hq_vit_l.pth` checkpoint + the COIFT dataset. On device-independent metrics
(peak mem, GMAC) the A10 already matches the 3090/L4 exactly, and mIoU is
device-independent, so the published −0.12% / −0.38% / −1.67% Δ-mIoU at density
0.75 / 0.50 / 0.25 carry over. Add the checkpoint + data and run
`tasks/sam_hq44k/eval_miou_l4.py --num-samples 280 --ratios 0.25 0.5 0.75` to
confirm.
