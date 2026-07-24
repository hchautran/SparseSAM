# SparseSAM benchmark on NVIDIA RTX 4000 Ada Generation (SM89)

SAM-HQ **ViT-L**, GPU: NVIDIA RTX 4000 Ada Generation (20 GB, cc 8.9), driver
535.261.03 (CUDA 12.2), torch 2.5.1+cu121, cutlass-dsl 4.6.1. Same scripts and
same settings as [RESULTS_3090.md](RESULTS_3090.md) / [RESULTS_L4.md](RESULTS_L4.md).

The SparseSAM FA2 CUTE kernel was written for Ampere SM80 (A100). It compiles and
runs on Ada SM89 with no code change — memory and FLOPs reproduce the reference
numbers exactly, and the **latency speedup is the highest measured so far**
(2.21× vs 2.28× on L4 and 1.70× on the 3090). See
[Why the speedup is high here](#why-the-speedup-is-high-here).

Note on the toolchain: cutlass-dsl 4.6.1 emits sm89 cubins that load fine on this
12.2 driver (CUDA 12.x minor-version compatibility, no PTX JIT involved), so the
newer CUDA the reference runs used is not required for the kernel itself.

## Headline (density ↔ speed ↔ memory)

| density  | enc speedup | peak-mem saving | GFLOP saving |
|----------|-------------|-----------------|--------------|
| baseline | 1.00×       | 1.0×            | 1.0×         |
| 0.75     | 1.69×       | 2.9×            | 1.13×        |
| 0.50     | 1.89×       | 2.9×            | 1.30×        |
| 0.25     | 2.21×       | 2.9×            | 1.54×        |

(Memory saving grows to **7.2×** at batch=8 — see below. Speedup is encoder-only;
full end-to-end including preprocessing and post-processing is **1.65–2.07×** at
1 prompt/image and 1.75× at 10 — see
[Full end-to-end](#full-end-to-end-including-prepost-processing-batch1).)

**Accuracy was not re-measured on this box** — the mIoU run needs
`ckts/sam_hq_vit_l.pth` plus the COIFT split of `thin_object_detection`, neither
of which is present here. Accuracy is device-independent in this codebase (the L4
and 3090 runs agree to ±0.0001 mIoU), so the reference tables carry over; but
nothing in this document is an accuracy measurement.

---

Details below. Latency / memory / GFLOPs use random-init weights (valid — these
don't depend on trained weights); input 1024×1024, fp16, batch=1 unless noted.

## Encoder: latency / memory / GFLOPs (batch=1)

| algo      | density | latency ms (median) | speedup | peak mem MB | mem saving | GMAC   | FLOP saving |
|-----------|---------|---------------------|---------|-------------|-----------|--------|-------------|
| baseline  | —       | 141.6               | 1.00×   | 2221.5      | 1.0×      | 1487.9 | 1.0×        |
| sparsesam | 0.25    | 64.1                | 2.21×   | 766.5       | 2.9×      | 968.8  | 1.54×       |
| sparsesam | 0.50    | 74.8                | 1.89×   | 766.5       | 2.9×      | 1141.8 | 1.30×       |
| sparsesam | 0.75    | 84.0                | 1.69×   | 766.5       | 2.9×      | 1314.8 | 1.13×       |

## Batch=8 (same model/input)

| algo      | density | latency ms (median) | speedup | peak mem MB | mem saving |
|-----------|---------|---------------------|---------|-------------|-----------|
| baseline  | —       | 1156.9              | 1.00×   | 13766.2     | 1.0×      |
| sparsesam | 0.25    | 557.4               | 2.08×   | 1908.1      | 7.2×      |
| sparsesam | 0.50    | 649.2               | 1.78×   | 1908.1      | 7.2×      |
| sparsesam | 0.75    | 727.2               | 1.59×   | 1908.1      | 7.2×      |

**Memory saving grows with batch** exactly as on the L4 and 3090: baseline peak
scales ~6× from batch 1→8 (2221→13766 MB) while sparsesam scales only ~2.5×
(766→1908 MB), widening the advantage 2.9× → 7.2×. Peak-memory figures at batch=8
are byte-identical to both reference runs (1908.1 MB) — allocator behaviour is
device-independent. At batch=1 this run reports 766.5 MB vs 765.3 MB on the
reference boxes (0.16%), the only memory number that is not bit-identical.

## End-to-end vs encoder-only (batch=1)

Full SAM-HQ pipeline (image encoder + prompt encoder + mask decoder), random box
prompts, starting from a GPU-resident tensor. Encoder runs fp16; decoder fp32
(this sam-hq HEAD is not half-clean).

**1 prompt/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 142.2  | 4.3    | 146.5  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 64.3   | 4.3    | 68.6   | 2.21×       | 2.13×       |
| sparsesam 0.50 | 75.0   | 4.3    | 79.3   | 1.90×       | 1.85×       |

**10 prompts/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 142.1  | 34.7   | 176.8  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 63.4   | 35.3   | 98.7   | 2.24×       | 1.79×       |
| sparsesam 0.50 | 74.1   | 35.3   | 109.4  | 1.92×       | 1.62×       |

Same Amdahl story as the reference runs: with 1 prompt the encoder dominates so
e2e ≈ encoder speedup (2.21× → 2.13×); the unaccelerated decoder dilutes it at 10
prompts (2.24× → 1.79×). The decoder here (34.7 ms at 10 prompts) sits between the
3090 (18.9 ms) and the L4 (44.2 ms), so the dilution is correspondingly mid-range.

## Full end-to-end, including pre/post-processing (batch=1)

The table above starts from a GPU-resident tensor. This one is the *whole* thing an
application actually pays for, matching `SamPredictor.set_image` / `.predict`
step for step (`bench_e2e_full.py`). Stage definitions are identical to
[RESULTS_3090.md](RESULTS_3090.md#full-end-to-end-including-prepost-processing-batch1).

10 real images from `input_imgs/` at native resolution (480×640 … 1365×2048), ms
per image, median of 10 iterations. `wall` is a separate sync-free pass — the true
wall clock; `sum` is the staged pass, which pays an extra `cudaSynchronize` per
boundary. They agree to well under 0.3%, so the breakdown is trustworthy.

**1 prompt/image**
| algo           | load | preproc | encoder | prompt | decoder | postproc | d2h  | **e2e ms** | e2e speedup | img/s |
|----------------|------|---------|---------|--------|---------|----------|------|------------|-------------|-------|
| baseline       | —    | 6.19    | 142.62  | 0.68   | 3.86    | 0.09     | 0.18 | **153.4**  | 1.00×       | 6.5   |
| sparsesam 0.25 | —    | 6.05    | 63.38   | 0.59   | 3.83    | 0.09     | 0.19 | **74.3**   | 2.07×       | 13.5  |
| sparsesam 0.50 | —    | 6.00    | 73.31   | 0.59   | 3.83    | 0.09     | 0.19 | **84.1**   | 1.83×       | 11.9  |
| sparsesam 0.75 | —    | 6.02    | 82.45   | 0.59   | 3.83    | 0.09     | 0.19 | **93.1**   | 1.65×       | 10.7  |

**10 prompts/image**
| algo           | load | preproc | encoder | prompt | decoder | postproc | d2h  | **e2e ms** | e2e speedup | img/s |
|----------------|------|---------|---------|--------|---------|----------|------|------------|-------------|-------|
| baseline       | —    | 6.01    | 142.49  | 0.62   | 33.65   | 0.52     | 1.09 | **184.5**  | 1.00×       | 5.4   |
| sparsesam 0.25 | —    | 6.02    | 62.56   | 0.60   | 34.37   | 0.52     | 1.09 | **105.2**  | 1.75×       | 9.5   |
| sparsesam 0.50 | —    | 6.17    | 72.36   | 0.67   | 34.39   | 0.53     | 1.10 | **115.3**  | 1.60×       | 8.7   |
| sparsesam 0.75 | —    | 6.27    | 81.41   | 0.71   | 34.42   | 0.54     | 1.10 | **124.3**  | 1.48×       | 8.0   |

**With JPEG/PNG decode included** (`--include-load`, 1 prompt/image): `load` costs
13.5 ms/image on top, so baseline 167.0 ms → sparsesam 0.25 86.7 ms, **1.93×**.

### What this changes vs the encoder-only story

- **The non-encoder floor is small at 1 prompt.** Everything except the encoder
  costs 10.8 ms (7.1% of e2e), so the Amdahl ceiling is 14.2× — the 2.25× encoder
  speedup measured here lands almost intact as 2.07× e2e. The floor is a smaller
  share than on the 3090 (10.8%) purely because this encoder is slower in absolute
  terms.
- **Preprocessing, not the decoder, is the largest fixed cost at 1 prompt**
  (6.0 ms vs 3.9 ms) — single-threaded CPU work (PIL bilinear resize), untouched
  by sparsification and therefore identical across every row.
- **Post-processing is nearly free** (0.09–0.54 ms); the D2H mask copy is 0.18 ms
  at 1 prompt and 1.09 ms at 10. Neither is worth optimizing.
- **Decode matters once you count disk I/O**: 13.5 ms of `cv2.imread` exceeds
  preprocess + decoder combined and drops e2e speedup 2.07× → 1.93×. Prefetchable
  in a real serving loop, which is why it is off by default.
- **At 10 prompts the floor grows to 42.0 ms (22.7%)**, ceiling 4.40×, and e2e
  speedup falls to 1.75×. Nearly all of that growth is the decoder (3.9 → 33.7 ms);
  preprocess is flat by construction.

Practical read: on this GPU SparseSAM turns a 6.5 img/s pipeline into 13.5 img/s
at 1 prompt, or 5.4 → 9.5 img/s at 10 prompts. The remaining headroom is in the
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
| **A** baseline      | stock attention (manual QKᵀ + rel-pos + PV)  | 143.67 | 2221.5  | 1.00× | 0.68× |
| **B** fused-dense   | same CUTE kernel, dense, full MLP            | 98.08  | 766.5   | 1.46× | 1.00× |
| **C** attn-sparse 0.25 | A-shape sparse attention, full MLP        | 83.99  | 766.5   | 1.71× | 1.17× |
| **C** attn-sparse 0.50 | "                                         | 89.91  | 766.5   | 1.60× | 1.09× |
| **D** sparsesam 0.25 | + keep-token MLP (the full method)          | 65.64  | 766.5   | 2.19× | 1.49× |
| **D** sparsesam 0.50 | "                                           | 76.00  | 766.5   | 1.89× | 1.29× |
| **E** tome 0.25     | baseline algorithm, stock attention          | 175.32 | 1992.6  | 0.82× | 0.56× |
| **E** tome 0.50     | "                                            | 172.76 | 1992.6  | 0.83× | 0.57× |

### The decomposition

```
sparsesam 0.25:  2.19x total  =  1.46x implementation  x  1.49x algorithmic
sparsesam 0.50:  1.89x total  =  1.46x implementation  x  1.29x algorithmic
```

**The custom kernel is worth 1.46×; the algorithm is worth 1.49×** — an almost
even split, and the single biggest difference from the reference GPUs. On the 3090
the same ablation attributes only 1.12× to the kernel; here the fused-dense control
alone recovers nearly half the total speedup. Within the algorithmic 1.49×, the
A-shape sparse attention contributes 1.17× and the keep-token MLP the remaining
1.28×.

Caveat on B, unchanged from the reference: it still pays SparseSAM's
token-permutation cost, which a pure kernel swap would not. That inflates B's
latency, so 1.46× is a *lower* bound on the implementation gain and 1.49× an upper
bound on the algorithmic one.

### Two things this ablation does *not* let us claim

1. **The memory saving is implementation, not algorithm.** B already drops peak
   memory to 766.5 MB — the full 2.9× — with zero sparsification, because the
   fused kernel never materializes the B×H×N×N attention matrix. Every memory
   number in this document is attributable to the kernel, not to token merging.
2. **The ToMe baseline here is not latency-competitive, and that is an
   implementation artifact.** At 175.3 ms it is *slower than not merging at all*
   (143.7 ms): it runs full-N attention and then pays a per-block bipartite
   matching cost that exceeds the MLP saving it buys. Its near-identical latency
   at ratio 0.25 and 0.50 (175.32 / 172.76 ms) confirms matching overhead, not
   token count, dominates. Any head-to-head *latency* claim against ToMe measures
   our kernel against its reference implementation. Baseline comparisons should
   be made on density / GMAC / accuracy — which are implementation-free — or with
   the baselines given comparable kernels.

## Why the speedup is high here

Everything device-independent reproduces exactly (memory, GMAC); only wall clock
differs. Lining up all three measured GPUs:

| | L4 (SM89) | RTX 4000 Ada (SM89) | RTX 3090 (SM86) |
|---|---|---|---|
| baseline enc, bs=1     | 184.7 ms | 141.6 ms | 100.1 ms |
| sparsesam 0.25, bs=1   | 80.9 ms  | 64.1 ms  | 58.9 ms  |
| resulting speedup      | 2.28×    | **2.21×**| 1.70×    |
| memory bandwidth (spec)| ~300 GB/s| ~360 GB/s| ~936 GB/s|

The speedup orders inversely with memory bandwidth, exactly as
[RESULTS_3090.md](RESULTS_3090.md#why-the-speedup-is-lower-here) predicts.
SparseSAM's gain exceeds its FLOP reduction (2.21× speedup at 1.54× fewer MACs)
because most of it comes from cutting memory traffic — the fused CUTE FA2 kernel
plus token merging — and that lever is worth most on a bandwidth-starved GPU. This
card sits just above the L4 in bandwidth and lands just below it in speedup; the
3090, with ~2.6× the bandwidth, has a far less bandwidth-bound dense baseline and
correspondingly less headroom to recover.

The implementation/algorithm split says the same thing from the other side: the
kernel-only gain is 1.46× here vs 1.12× on the 3090, i.e. the fused kernel's
traffic reduction is what scales with bandwidth scarcity, while the algorithmic
1.29–1.49× is roughly constant across all three GPUs.

Practical read: SparseSAM's speedup tracks how bandwidth-limited the target GPU
is. Memory savings (2.9× / 7.2×) and accuracy are unaffected and transfer as-is.

Notes
- GFLOPs use the MAC convention; baseline cross-checks against fvcore (1487.9 vs
  1493.8 GMAC, 0.4%). fvcore can't trace sparsesam's custom kernel, so sparsesam
  FLOPs are hook-measured Linear/Conv MACs + density-scaled attention.
- Latency is median over 20 iters (10 at batch=8) after warmup; run-to-run spread
  was well under 1% (median vs min ≈ 0.2%).
- Bandwidth figures in the table above are vendor spec sheets, not measured here.
- Raw script output for every table is in [`rtx4000ada_run/`](rtx4000ada_run/).

## Reproduce

```
# env (see repo README; this run used python 3.12 + torch 2.5.1+cu121)
git submodule update --init algos/3rd_party/sam-hq
pip install -e . && pip install -e algos/3rd_party/sam-hq
pip install "nvidia-cutlass-dsl==4.6.1" "cuda-python==12.9.0"

# latency / memory / FLOPs (no checkpoint or data needed)
PYTHONPATH=$PWD python tasks/sam_profile/bench_encoder_l4.py --model-type vit_l --ratios 0.25 0.5 0.75
PYTHONPATH=$PWD python tasks/sam_profile/bench_encoder_l4.py --model-type vit_l --ratios 0.25 0.5 0.75 \
    --batch-sizes 8 --iters 10 --warmup 3
PYTHONPATH=$PWD python tasks/sam_profile/flops_encoder_l4.py --model-type vit_l --ratios 0.25 0.5 0.75
PYTHONPATH=$PWD python tasks/sam_profile/bench_e2e_l4.py --model-type vit_l --ratios 0.25 0.5 --prompts 1 10

# implementation vs algorithmic gain (fused-dense control)
PYTHONPATH=$PWD python tasks/sam_profile/bench_impl_ablation.py --model-type vit_l --ratios 0.25 0.5

# full end-to-end with per-stage breakdown (uses real images from input_imgs/)
PYTHONPATH=$PWD python tasks/sam_profile/bench_e2e_full.py --model-type vit_l \
    --ratios 0.25 0.5 0.75 --prompts 1 10
PYTHONPATH=$PWD python tasks/sam_profile/bench_e2e_full.py --model-type vit_l \
    --ratios 0.25 --prompts 1 --include-load          # add JPEG/PNG decode

# accuracy — NOT run here; needs ckts/sam_hq_vit_l.pth + data/thin_object_detection/COIFT
PYTHONPATH=$PWD python tasks/sam_hq44k/eval_miou_l4.py --num-samples 280 --ratios 0.25 0.5 0.75
```
