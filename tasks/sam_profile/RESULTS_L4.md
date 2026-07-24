# SparseSAM benchmark on NVIDIA L4 (SM89)

SAM-HQ **ViT-L**, GPU: NVIDIA L4 (23 GB, cc 8.9), CUDA 12.8, torch 2.8.0+cu128,
nvidia-cutlass-dsl 4.6.1, cuda-python 12.9.0. Same scripts and settings as
[RESULTS_3090.md](RESULTS_3090.md) / [RESULTS_A10.md](RESULTS_A10.md).

The SparseSAM FA2 CUTE kernel (written for Ampere SM80) compiles and runs on the
L4's Ada SM89 with no code change, reproducing the paper's ~2× speed / 2.9× memory
headline. Device-independent metrics — peak memory, GMAC, FLOP saving — are
**byte-identical** to the 3090/A10 runs. This is a fresh re-run of the efficiency
suite on an L4; numbers were measured with `torch 2.8.0+cu128`.

> **Note on this L4 unit.** The L4 is a 72 W single-slot card and throttles quickly
> under sustained fp16 load (idle floor ~65 °C, climbing to ~85 °C within a bench).
> Encoder-only and enc+dec latencies below are measured with a cooldown before each
> benchmark, at the default warmup, so they reflect steady-state clocks. The full
> pre/post-processing e2e numbers are more thermally sensitive — see that section.
> Memory, GMAC and mIoU are throttle-independent and reproduce exactly.

## Headline (density ↔ speed ↔ memory ↔ accuracy)

| density  | enc speedup | peak-mem saving | GFLOP saving | mIoU (COIFT) | Δ mIoU  |
|----------|-------------|-----------------|--------------|--------------|---------|
| baseline | 1.00×       | 1.0×            | 1.0×         | 0.9455       | —       |
| 0.75     | 1.74×       | 2.9×            | 1.13×        | 0.9444       | −0.11%  |
| 0.50     | 1.92×       | 2.9×            | 1.30×        | 0.9419       | −0.38%  |
| 0.25     | 2.16×       | 2.9×            | 1.54×        | 0.9297       | −1.67%  |

(Memory saving grows to **7.2×** at batch=8 — see below. Speedup is encoder-only;
enc+dec end-to-end is **2.19×** at density 0.25, 1 prompt/image. mIoU is carried
over from the prior run — it is device-independent and needs the checkpoint + COIFT
data to re-measure; see [Accuracy](#accuracy--miou-coift-box-prompted-sam-hq-vit-l).)

---

Details below. Latency / memory / GFLOPs use random-init weights (valid — these
don't depend on trained weights); input 1024×1024, fp16, batch=1 unless noted.

## Encoder: latency / memory / GFLOPs (batch=1)

| algo      | density | latency ms (median) | speedup | peak mem MB | mem saving | GMAC   | FLOP saving |
|-----------|---------|---------------------|---------|-------------|-----------|--------|-------------|
| baseline  | —       | 190.9               | 1.00×   | 2221.5      | 1.0×      | 1487.9 | 1.0×        |
| sparsesam | 0.25    | 88.5                | 2.16×   | 765.3       | 2.9×      | 968.8  | 1.54×       |
| sparsesam | 0.50    | 99.6                | 1.92×   | 765.3       | 2.9×      | 1141.8 | 1.30×       |
| sparsesam | 0.75    | 110.0               | 1.74×   | 765.3       | 2.9×      | 1314.8 | 1.13×       |

## Batch=8 (same model/input)

| algo      | density | latency ms (median) | speedup | peak mem MB | mem saving |
|-----------|---------|---------------------|---------|-------------|-----------|
| baseline  | —       | 1664.8              | 1.00×   | 13766.2     | 1.0×      |
| sparsesam | 0.25    | 806.0               | 2.07×   | 1908.1      | 7.2×      |
| sparsesam | 0.50    | 920.5               | 1.81×   | 1908.1      | 7.2×      |
| sparsesam | 0.75    | 1031.6              | 1.61×   | 1908.1      | 7.2×      |

**Memory saving grows with batch:** baseline peak scales ~6× from batch 1→8
(2221→13766 MB) while sparsesam scales only ~2.5× (765→1908 MB), so the mem
advantage widens 2.9× → 7.2×. Latency speedup stays ~2×. Peak-memory figures
(765.3 / 1908.1 MB) are byte-identical to the 3090 and A10 runs — allocator
behaviour is device-independent.

Notes
- Latency speedup exceeds FLOP reduction → gain is from the fused CUTE FA2 kernel +
  reduced memory traffic (token merge), not FLOPs alone — quantified in
  [Implementation vs algorithmic gain](#implementation-gain-vs-algorithmic-gain).
- GFLOPs use the MAC convention; baseline cross-checks against fvcore (1487.9 vs
  1493.8 GMAC, 0.4%). fvcore can't trace sparsesam's custom kernel, so sparsesam
  FLOPs are hook-measured Linear/Conv MACs + density-scaled attention.

## End-to-end vs encoder-only (batch=1, encoder+decoder)

Full SAM-HQ compute path on an already-resident GPU tensor (image encoder + prompt
encoder + mask decoder), random box prompts. Encoder fp16; decoder fp32 (this
sam-hq HEAD is not half-clean).

**1 prompt/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 188.7  | 5.8    | 194.5  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 83.2   | 5.8    | 88.9   | 2.27×       | 2.19×       |
| sparsesam 0.50 | 97.1   | 5.8    | 102.9  | 1.94×       | 1.89×       |

**10 prompts/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 191.1  | 46.4   | 237.5  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 85.2   | 46.7   | 132.0  | 2.24×       | 1.80×       |
| sparsesam 0.50 | 97.5   | 46.5   | 144.0  | 1.96×       | 1.65×       |

The encoder dominates, so with few prompts e2e ≈ encoder speedup (2.2×). The
decoder is unaccelerated fixed cost, so many prompts/image dilute the speedup
(Amdahl): 10 prompts → e2e 1.80×.

## Full end-to-end, incl. pre/post-processing (batch=1, real images from `input_imgs/`)

The table above starts from a GPU-resident tensor. This one walks the whole
pipeline an application pays for, matching `SamPredictor.set_image` / `.predict`
step for step (`bench_e2e_full.py`): 10 real images (480×640 … 1365×2048), ms per
image, median of 10.

**1 prompt/image**
| algo           | preproc | encoder | decoder | postproc | d2h  | **e2e ms** | e2e spd | img/s |
|----------------|---------|---------|---------|----------|------|------------|---------|-------|
| baseline       | 7.66    | 188.15  | 5.01    | 0.12     | 0.20 | **201.9**  | 1.00×   | 4.9   |
| sparsesam 0.25 | 7.66    | 83.39   | 5.27    | 0.12     | 0.20 | **97.4**   | 2.07×   | 10.2  |

Baseline non-encoder floor: **17.1 ms/image (8.3 % of e2e)**, Amdahl ceiling 12.0×.

- **Preprocessing (7.7 ms), not the decoder, is the largest fixed cost at 1 prompt.**
  It is single-threaded CPU work (PIL bilinear resize in `ResizeLongestSide`) and is
  untouched by sparsification — identical across every row.
- **Post-processing (0.12 ms) and the D2H mask copy (0.20 ms) are nearly free**, even
  though postproc upscales to full native resolution. Not worth optimizing.
- **With JPEG/PNG decode included** (`--include-load`): `cv2.imread` adds **17.0 ms**
  on top, so baseline 220.6 ms → sparsesam 0.25 153.1 ms, **1.44×** (4.5 → 6.5 img/s).
  In a serving loop this decode is prefetchable, which is why it is off by default.
- **At 10 prompts the floor grows to the decoder** (~46 ms, from the enc+dec table):
  the non-encoder share rises to ~28 %, Amdahl ceiling ~3.6×, and e2e speedup falls
  to ~1.8× at density 0.25.

> **Thermal caveat.** `bench_e2e_full.py` times the dense baseline first, then each
> sparse config, with no cooldown between; on this throttle-prone L4 that means the
> sparse configs are timed on a hotter card, which *depresses* their measured
> full-pipeline speedup at densities 0.50 / 0.75 (and at 10 prompts) below what the
> cooled encoder-only table reports. Density 0.25 above is the clean, representative
> case; treat the [encoder-only](#encoder-latency--memory--gflops-batch1) and
> [enc+dec](#end-to-end-vs-encoder-only-batch1-encoderdecoder) tables as the
> reference for per-density speedup. Raw per-ratio logs: `l4_run/e2e_full.txt`.

## Implementation gain vs algorithmic gain (fused-dense control)

The headline speedup compares a fused CUTLASS-DSL kernel against stock SAM-HQ
attention, mixing "better kernel" with "better algorithm". `bench_impl_ablation.py`
separates them with a **fused-dense control** (B): the identical CUTE kernel at
`ratio=1.0` — dense mask, full MLP, no sparsification. It differs from the baseline
*only* in implementation.

ViT-L, batch=1, 1024², median of 20 iters:

| config                 | what it is                                  | lat ms | peak MB | vs A  | vs B  |
|------------------------|---------------------------------------------|--------|---------|-------|-------|
| **A** baseline         | stock attention (manual QKᵀ + rel-pos + PV) | 191.51 | 2221.5  | 1.00× | 0.66× |
| **B** fused-dense      | same CUTE kernel, dense, full MLP           | 127.05 | 765.3   | 1.51× | 1.00× |
| **C** attn-sparse 0.25 | A-shape sparse attention, full MLP          | 116.79 | 765.3   | 1.64× | 1.09× |
| **C** attn-sparse 0.50 | "                                           | 119.47 | 765.3   | 1.60× | 1.06× |
| **D** sparsesam 0.25   | + keep-token MLP (the full method)          | 89.45  | 765.3   | 2.14× | 1.42× |
| **D** sparsesam 0.50   | "                                           | 99.65  | 765.3   | 1.92× | 1.27× |
| **E** tome 0.25        | baseline algorithm, stock attention         | 259.48 | 1992.3  | 0.74× | 0.49× |
| **E** tome 0.50        | "                                           | 261.84 | 1992.3  | 0.73× | 0.49× |

```
sparsesam 0.25:  2.14x total  =  1.51x implementation  x  1.42x algorithmic
sparsesam 0.50:  1.92x total  =  1.51x implementation  x  1.27x algorithmic
```

**On the L4 the custom kernel is worth 1.51×** — far more than on the compute-rich
3090 (1.12×) or A10 (1.15×). This is the L4 story: it is bandwidth-starved (~300
GB/s), so the fused CUTE kernel — which never materializes the B×H×N×N attention
matrix and slashes memory traffic — recovers a large fraction of the win before any
token is even merged. The algorithm (token sparsification + keep-token MLP) adds a
further 1.42× on top.

Two things this ablation shows:
1. **The memory saving is implementation, not algorithm.** B already drops peak
   memory to 765.3 MB — the full 2.9× — with zero sparsification, because the fused
   kernel never materializes the attention matrix. Every memory number here is
   attributable to the kernel, not to token merging.
2. **The ToMe reference is not latency-competitive, and that is an implementation
   artifact.** At ~260 ms it is *slower than not merging at all* (191 ms): it runs
   full-N attention then pays a per-block bipartite-matching cost that exceeds the
   MLP saving. Its near-identical latency at ratio 0.25 and 0.50 (259.5 / 261.8 ms)
   confirms matching overhead, not token count, dominates. Baseline comparisons
   should be made on density / GMAC / accuracy — which are implementation-free — not
   on wall clock.

## Accuracy — mIoU (COIFT, box-prompted, SAM-HQ ViT-L)

280 COIFT images, box prompts from GT. Higher density = less sparsification.

| density  | mIoU   | Δ mIoU            | Boundary IoU |
|----------|--------|-------------------|--------------|
| baseline | 0.9455 | —                 | 0.8959       |
| 0.75     | 0.9444 | −0.0010 (−0.11%)  | 0.8946       |
| 0.50     | 0.9419 | −0.0036 (−0.38%)  | 0.8915       |
| 0.25     | 0.9297 | −0.0158 (−1.67%)  | 0.8720       |

Confirms the paper's "<1% IoU loss" at density ≥0.5. Clean accuracy/speed tradeoff:
density 0.25 → 2.16× encoder speedup at −1.67% mIoU; density 0.75 → essentially
lossless (−0.1%).

> mIoU was **not** re-measured in this efficiency run — it needs the
> `sam_hq_vit_l.pth` checkpoint (1.25 GB) + the COIFT dataset. It is device- and
> throttle-independent, and the L4 already matches the 3090/A10 on every
> device-independent metric (peak mem, GMAC) byte-for-byte, so the Δ-mIoU above
> carries over unchanged. Re-confirm with the command below.

## Reproduce

```
# deps (Python 3.12, torch 2.8.0+cu128)
git submodule update --init algos/3rd_party/sam-hq
pip install fvcore opencv-python-headless timm einops pandas scikit-image pycocotools matplotlib
pip install "nvidia-cutlass-dsl==4.6.1" "cuda-python==12.9.0"   # cuda-python 12.x REQUIRED on a CUDA-12.8 driver

export PYTHONPATH=$PWD
# latency / memory / FLOPs (no checkpoint or data needed)
python tasks/sam_profile/bench_encoder_l4.py    --model-type vit_l --ratios 0.25 0.5 0.75
python tasks/sam_profile/bench_encoder_l4.py    --model-type vit_l --ratios 0.25 0.5 0.75 --batch-sizes 8 --iters 10 --warmup 3
python tasks/sam_profile/flops_encoder_l4.py    --model-type vit_l --ratios 0.25 0.5 0.75
python tasks/sam_profile/bench_e2e_l4.py        --model-type vit_l --ratios 0.25 0.5 --prompts 1 10
python tasks/sam_profile/bench_impl_ablation.py --model-type vit_l --ratios 0.25 0.5
python tasks/sam_profile/bench_e2e_full.py      --model-type vit_l --ratios 0.25 0.5 0.75 --prompts 1 10
python tasks/sam_profile/bench_e2e_full.py      --model-type vit_l --ratios 0.25 --prompts 1 --include-load

# accuracy — needs ckts/sam_hq_vit_l.pth + data/thin_object_detection/COIFT
python tasks/sam_hq44k/eval_miou_l4.py --num-samples 280 --ratios 0.25 0.5 0.75
```

On this 72 W L4, insert a short GPU cooldown between benchmarks (and run
`bench_e2e_full` one ratio at a time) for steady, comparable absolute latencies —
see the thermal notes above. Raw run logs: `l4_run/`.

Assets used for the accuracy run:
- checkpoint: `huggingface.co/lkeab/hq-sam` → `sam_hq_vit_l.pth` (1.25 GB) into `ckts/`
- COIFT: `thin_object_detection.zip` → extract `COIFT/` into `data/thin_object_detection/`
  (280 images + 280 masks)
