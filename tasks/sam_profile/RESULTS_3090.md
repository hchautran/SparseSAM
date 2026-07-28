# SparseSAM benchmark on NVIDIA RTX 3090 (SM86)

SAM-HQ **ViT-L**, GPU: NVIDIA GeForce RTX 3090 (24 GB, cc 8.6), CUDA 12.8,
torch 2.8.0+cu128, cutlass-dsl 4.6.1. Same scripts and same settings as
[RESULTS_L4.md](RESULTS_L4.md), re-run on consumer Ampere.

The SparseSAM FA2 CUTE kernel was written for Ampere SM80 (A100). It compiles and
runs correctly on SM86 as well (no code change) — memory, FLOPs, and accuracy
reproduce the L4 numbers exactly; the **latency speedup is lower** on this GPU
(1.7× vs 2.3×). See [Why the speedup is lower here](#why-the-speedup-is-lower-here).

## Headline (density ↔ speed ↔ memory ↔ accuracy)

| density  | enc speedup | peak-mem saving | GFLOP saving | mIoU (COIFT) | Δ mIoU  |
|----------|-------------|-----------------|--------------|--------------|---------|
| baseline | 1.00×       | 1.0×            | 1.0×         | 0.9455       | —       |
| 0.75     | 1.26×       | 2.9×            | 1.13×        | 0.9444       | −0.12%  |
| 0.50     | 1.44×       | 2.9×            | 1.30×        | 0.9419       | −0.38%  |
| 0.25     | 1.70×       | 2.9×            | 1.54×        | 0.9296       | −1.67%  |

(Memory saving grows to **7.2×** at batch=8 — see below. Speedup is encoder-only;
full end-to-end including preprocessing and post-processing is **1.49–1.60×** at
1 prompt/image and 1.51× at 10 — see
[Full end-to-end](#full-end-to-end-including-prepost-processing-batch1).)

---

Details below. Latency / memory / GFLOPs use random-init weights (valid — these
don't depend on trained weights); input 1024×1024, fp16, batch=1 unless noted.

## Encoder: latency / memory / GFLOPs (batch=1)

| algo      | density | latency ms (median) | speedup | peak mem MB | mem saving | GMAC   | FLOP saving |
|-----------|---------|---------------------|---------|-------------|-----------|--------|-------------|
| baseline  | —       | 100.1               | 1.00×   | 2221.5      | 1.0×      | 1487.9 | 1.0×        |
| sparsesam | 0.25    | 58.9                | 1.70×   | 765.3       | 2.9×      | 968.8  | 1.54×       |
| sparsesam | 0.50    | 69.5                | 1.44×   | 765.3       | 2.9×      | 1141.8 | 1.30×       |
| sparsesam | 0.75    | 79.2                | 1.26×   | 765.3       | 2.9×      | 1314.8 | 1.13×       |

## Batch=8 (same model/input)

| algo      | density | latency ms (median) | speedup | peak mem MB | mem saving |
|-----------|---------|---------------------|---------|-------------|-----------|
| baseline  | —       | 720.9               | 1.00×   | 13766.2     | 1.0×      |
| sparsesam | 0.25    | 425.4               | 1.69×   | 1908.1      | 7.2×      |
| sparsesam | 0.50    | 505.3               | 1.43×   | 1908.1      | 7.2×      |
| sparsesam | 0.75    | 583.3               | 1.24×   | 1908.1      | 7.2×      |

**Memory saving grows with batch** exactly as on the L4: baseline peak scales ~6×
from batch 1→8 (2221→13766 MB) while sparsesam scales only ~2.5× (765→1908 MB),
widening the advantage 2.9× → 7.2×. Peak-memory figures are byte-identical to the
L4 run — allocator behaviour here is device-independent.

## End-to-end vs encoder-only (batch=1)

Full SAM-HQ pipeline (image encoder + prompt encoder + mask decoder), random box
prompts. Encoder runs fp16; decoder fp32 (this sam-hq HEAD is not half-clean).

**1 prompt/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 99.2   | 4.9    | 104.1  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 58.6   | 5.4    | 64.1   | 1.69×       | 1.62×       |
| sparsesam 0.50 | 69.4   | 5.9    | 75.3   | 1.43×       | 1.38×       |

**10 prompts/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 99.8   | 18.9   | 118.7  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 59.4   | 19.2   | 78.6   | 1.68×       | 1.51×       |
| sparsesam 0.50 | 70.0   | 19.2   | 89.1   | 1.43×       | 1.33×       |

Same Amdahl story as the L4: the encoder dominates, so with 1 prompt e2e ≈ encoder
speedup; the unaccelerated decoder dilutes it at 10 prompts (1.68× → 1.51×). The
decoder is much cheaper here than on the L4 (18.9 ms vs 44.2 ms at 10 prompts), so
the dilution is milder in relative terms.

## Full end-to-end, including pre/post-processing (batch=1)

The table above starts from a GPU-resident tensor. This one is the *whole* thing an
application actually pays for, matching `SamPredictor.set_image` / `.predict`
step for step (`bench_e2e_full.py`):

| stage        | what runs                                                              | where |
|--------------|------------------------------------------------------------------------|-------|
| `load`       | `cv2.imread` + BGR→RGB                                                 | disk + CPU |
| `preprocess` | `ResizeLongestSide` (PIL bilinear) → HWC→CHW → H2D → normalize + pad → fp16 | CPU + copy |
| `encoder`    | `image_encoder` (+ fp16→fp32 feature bridge)                           | GPU |
| `prompt`     | `transform.apply_boxes` + `prompt_encoder`                             | CPU + GPU |
| `decoder`    | `mask_decoder` (`hq_token_only`)                                       | GPU |
| `postproc`   | `postprocess_masks` (2× interpolate to original size) + threshold      | GPU |
| `d2h`        | masks + IoU → CPU numpy                                                | copy |

10 real images from `input_imgs/` at native resolution (480×640 … 1365×2048), ms
per image, median of 10 iterations. `wall` is a separate sync-free pass — the true
wall clock; `sum` is the staged pass, which pays an extra `cudaSynchronize` per
boundary. They agree to well under 1%, so the breakdown is trustworthy.

**1 prompt/image**
| algo           | load | preproc | encoder | prompt | decoder | postproc | d2h  | **e2e ms** | e2e speedup | img/s |
|----------------|------|---------|---------|--------|---------|----------|------|------------|-------------|-------|
| baseline       | —    | 6.84    | 99.20   | 0.72   | 5.03    | 0.10     | 0.17 | **111.2**  | 1.00×       | 9.0   |
| sparsesam 0.25 | —    | 6.84    | 57.06   | 0.72   | 5.06    | 0.10     | 0.17 | **69.7**   | 1.60×       | 14.4  |
| sparsesam 0.50 | —    | 6.81    | 65.85   | 0.71   | 4.99    | 0.10     | 0.16 | **78.6**   | 1.41×       | 12.7  |
| sparsesam 0.75 | —    | 6.44    | 74.27   | 0.68   | 4.65    | 0.10     | 0.16 | **86.4**   | 1.29×       | 11.6  |

**10 prompts/image**
| algo           | load | preproc | encoder | prompt | decoder | postproc | d2h  | **e2e ms** | e2e speedup | img/s |
|----------------|------|---------|---------|--------|---------|----------|------|------------|-------------|-------|
| baseline       | —    | 6.86    | 100.95  | 0.72   | 18.14   | 0.29     | 0.76 | **127.5**  | 1.00×       | 7.8   |
| sparsesam 0.25 | —    | 6.75    | 57.96   | 0.70   | 18.35   | 0.29     | 0.75 | **84.4**   | 1.51×       | 11.8  |
| sparsesam 0.50 | —    | 6.58    | 66.60   | 0.68   | 18.39   | 0.29     | 0.75 | **93.3**   | 1.37×       | 10.7  |
| sparsesam 0.75 | —    | 6.43    | 74.70   | 0.66   | 18.46   | 0.29     | 0.74 | **101.1**  | 1.26×       | 9.9   |

**With JPEG/PNG decode included** (`--include-load`, 1 prompt/image): `load` costs
16.2 ms/image on top, so baseline 130.1 ms → sparsesam 0.25 87.3 ms, **1.49×**.

### What this changes vs the encoder-only story

- **The non-encoder floor is real but small at 1 prompt.** Everything except the
  encoder costs 12.0 ms (10.8% of e2e), so the Amdahl ceiling is 9.3× — the 1.74×
  encoder speedup measured here lands almost intact as 1.60× e2e.
- **Preprocessing, not the decoder, is the largest fixed cost at 1 prompt**
  (6.8 ms vs 5.0 ms). It is single-threaded CPU work — PIL bilinear resize
  dominates — and is entirely untouched by sparsification, so it is identical
  across every row. Moving it to `apply_image_torch` on GPU, or overlapping it
  with the previous image's encoder pass, would recover most of it.
- **Post-processing is nearly free** (0.10–0.29 ms) even though it upscales to
  full native resolution; the D2H mask copy is 0.17 ms at 1 prompt and 0.76 ms at
  10. Neither is worth optimizing.
- **Decode dominates the floor once you count disk I/O**: 16.2 ms of `cv2.imread`
  is larger than preprocess + decoder combined, and drops e2e speedup 1.60× →
  1.49×. In a real serving loop this is prefetchable, which is why it is off by
  default.
- **At 10 prompts the floor triples to 26.6 ms (20.8%)**, ceiling 4.8×, and e2e
  speedup falls to 1.51×. Nearly all of that growth is the decoder (5.0 → 18.1 ms);
  preprocess is flat by construction.

Practical read: on this GPU SparseSAM turns a 9 img/s pipeline into 14.4 img/s at
1 prompt, or 7.8 → 11.8 img/s at 10 prompts. The remaining headroom is in the
CPU-side preprocess and the unaccelerated decoder, not in the encoder.

## Implementation gain vs algorithmic gain

The headline speedup compares a fused CUTLASS-DSL kernel against stock SAM-HQ
attention, so it mixes "better kernel" with "better algorithm". `bench_impl_ablation.py`
separates them with a **fused-dense control**: the identical CUTE kernel at
`ratio=1.0`, i.e. dense mask, full MLP, no sparsification. It differs from the
baseline *only* in implementation.

ViT-L, batch=1, 1024², median of 20 iters:

| config              | what it is                                   | lat ms | peak MB | vs A  | vs B  |
|---------------------|----------------------------------------------|--------|---------|-------|-------|
| **A** baseline      | stock attention (manual QKᵀ + rel-pos + PV)  | 100.17 | 2221.5  | 1.00× | 0.89× |
| **B** fused-dense   | same CUTE kernel, dense, full MLP            | 89.36  | 765.3   | 1.12× | 1.00× |
| **C** attn-sparse 0.25 | A-shape sparse attention, full MLP        | 73.61  | 765.3   | 1.36× | 1.21× |
| **C** attn-sparse 0.50 | "                                         | 79.09  | 765.3   | 1.27× | 1.13× |
| **D** sparsesam 0.25 | + keep-token MLP (the full method)          | 58.55  | 765.3   | 1.71× | 1.53× |
| **D** sparsesam 0.50 | "                                           | 69.17  | 765.3   | 1.45× | 1.29× |
| **E** tome 0.25     | baseline algorithm, stock attention          | 168.86 | 1992.3  | 0.59× | 0.53× |
| **E** tome 0.50     | "                                            | 169.23 | 1992.3  | 0.59× | 0.53× |

**B is a valid control**: its output matches the baseline to `max_abs=0.0088`,
`rel=0.0017` on features with `std=1.0` — pure fp16 accumulation noise, so it
computes the same dense attention, just faster.

### The decomposition

```
sparsesam 0.25:  1.71x total  =  1.12x implementation  x  1.53x algorithmic
sparsesam 0.50:  1.45x total  =  1.12x implementation  x  1.29x algorithmic
```

**The custom kernel is worth 1.12×; the algorithm is worth 1.53×.** Handing the
baseline an equally good attention implementation removes only about a fifth of
the reported speedup — the remainder is token sparsification and the keep-token
MLP, which would transfer to any equally-optimized backend. Within the
algorithmic 1.53×, the A-shape sparse attention contributes 1.21× and the
keep-token MLP the remaining 1.26×.

Caveat on B: it still pays SparseSAM's token-permutation cost, which a pure
kernel swap would not. That inflates B's latency, so 1.12× is a *lower* bound on
the implementation gain and 1.53× an upper bound on the algorithmic one.

### Two things this ablation does *not* let us claim

1. **The memory saving is implementation, not algorithm.** B already drops peak
   memory to 765.3 MB — the full 2.9× — with zero sparsification, because the
   fused kernel never materializes the B×H×N×N attention matrix. Every memory
   number in this document is attributable to the kernel, not to token merging.
2. **The ToMe baseline here is not latency-competitive, and that is an
   implementation artifact.** At 168.9 ms it is *slower than not merging at all*
   (100.2 ms): it runs full-N attention and then pays a per-block bipartite
   matching cost that exceeds the MLP saving it buys. Its near-identical latency
   at ratio 0.25 and 0.50 (168.86 / 169.23 ms) confirms matching overhead, not
   token count, dominates. Any head-to-head *latency* claim against ToMe measures
   our kernel against its reference implementation. Baseline comparisons should
   be made on density / GMAC / accuracy — which are implementation-free — or with
   the baselines given comparable kernels.

## Accuracy — mIoU (COIFT, box-prompted, SAM-HQ ViT-L)

280 COIFT images, box prompts from GT. Higher density = less sparsification.

| density  | mIoU   | Δ mIoU            | Boundary IoU |
|----------|--------|-------------------|--------------|
| baseline | 0.9455 | —                 | 0.8959       |
| 0.75     | 0.9444 | −0.0011 (−0.12%)  | 0.8946       |
| 0.50     | 0.9419 | −0.0036 (−0.38%)  | 0.8914       |
| 0.25     | 0.9296 | −0.0158 (−1.67%)  | 0.8718       |

Matches the L4 table to within ±0.0001 mIoU (the residual is fp16 non-determinism,
not a hardware difference), confirming the kernel is numerically correct on SM86.
The paper's "<1% IoU loss at density ≥0.5" holds.

## Why the speedup is lower here

Everything device-independent reproduces exactly (memory, GMAC, mIoU); only wall
clock differs. The gap comes from the *baseline* getting much faster on this GPU,
not from sparsesam getting slower:

| | L4 (SM89) | RTX 3090 (SM86) | ratio |
|---|---|---|---|
| baseline enc, bs=1     | 184.7 ms | 100.1 ms | 1.85× faster |
| sparsesam 0.25, bs=1   | 80.9 ms  | 58.9 ms  | 1.37× faster |
| resulting speedup      | 2.28×    | 1.70×    | — |

SparseSAM's gain exceeds its FLOP reduction (1.70× speedup at 1.54× fewer MACs
here) because most of it comes from cutting memory traffic — the fused CUTE FA2
kernel plus token merging. That lever is worth most on a bandwidth-starved GPU.
The 3090 has roughly 3× the memory bandwidth of an L4 (~936 vs ~300 GB/s) at
comparable fp16 tensor throughput, so the dense baseline is far less
bandwidth-bound here and has less headroom for SparseSAM to recover. The
compute-bound part of the win (the 1.54× FLOP cut) still lands; the traffic-bound
part largely doesn't.

Practical read: SparseSAM's speedup tracks how bandwidth-limited the target GPU
is. Memory savings (2.9× / 7.2×) and accuracy are unaffected and transfer as-is.

Notes
- GFLOPs use the MAC convention; baseline cross-checks against fvcore (1487.9 vs
  1493.8 GMAC, 0.4%). fvcore can't trace sparsesam's custom kernel, so sparsesam
  FLOPs are hook-measured Linear/Conv MACs + density-scaled attention.
- Latency is median over 20 iters (10 at batch=8) after warmup; run-to-run spread
  was under 1% (median vs min ≈ 0.5%).

## Reproduce

```
# deps (torch 2.8.0+cu128 already present)
git submodule update --init algos/3rd_party/sam-hq
pip install fvcore opencv-python-headless timm einops pandas scikit-image pycocotools matplotlib
pip install "nvidia-cutlass-dsl==4.6.1"

# latency / memory / FLOPs (no checkpoint or data needed)
PYTHONPATH=/SparseSAM python tasks/sam_profile/bench_encoder_l4.py --model-type vit_l --ratios 0.25 0.5 0.75
PYTHONPATH=/SparseSAM python tasks/sam_profile/bench_encoder_l4.py --model-type vit_l --ratios 0.25 0.5 0.75 \
    --batch-sizes 8 --iters 10 --warmup 3
PYTHONPATH=/SparseSAM python tasks/sam_profile/flops_encoder_l4.py --model-type vit_l --ratios 0.25 0.5 0.75
PYTHONPATH=/SparseSAM python tasks/sam_profile/bench_e2e_l4.py --model-type vit_l --ratios 0.25 0.5 --prompts 1 10

# implementation vs algorithmic gain (fused-dense control)
PYTHONPATH=/SparseSAM python tasks/sam_profile/bench_impl_ablation.py --model-type vit_l \
    --ratios 0.25 0.5

# full end-to-end with per-stage breakdown (uses real images from input_imgs/)
PYTHONPATH=/SparseSAM python tasks/sam_profile/bench_e2e_full.py --model-type vit_l \
    --ratios 0.25 0.5 0.75 --prompts 1 10
PYTHONPATH=/SparseSAM python tasks/sam_profile/bench_e2e_full.py --model-type vit_l \
    --ratios 0.25 --prompts 1 --include-load          # add JPEG/PNG decode

# accuracy — needs ckts/sam_hq_vit_l.pth + data/thin_object_detection/COIFT
PYTHONPATH=/SparseSAM python tasks/sam_hq44k/eval_miou_l4.py --num-samples 280 --ratios 0.25 0.5 0.75
```

Assets used for the accuracy run:
- checkpoint: `huggingface.co/lkeab/hq-sam` → `sam_hq_vit_l.pth` (1.25 GB) into `ckts/`
- COIFT: `thin_object_detection.zip` → extract `COIFT/` into `data/thin_object_detection/`
  (280 images + 280 masks)
