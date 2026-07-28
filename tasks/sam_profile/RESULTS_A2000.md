# SparseSAM benchmark on NVIDIA RTX A2000 12GB (SM86)

SAM-HQ **ViT-L**, GPU: NVIDIA RTX A2000 12GB (cc 8.6), driver 535.230.02
(CUDA 12.2), torch 2.5.1+cu121, cutlass-dsl 4.6.1. Same scripts and settings as
[RESULTS_3090.md](RESULTS_3090.md) / [RESULTS_L4.md](RESULTS_L4.md), re-run on a
12 GB workstation Ampere card.

Same SM86 family as the RTX 3090, so the SparseSAM FA2 CUTE kernel (written for
Ampere SM80) compiles and runs unchanged. **All device-independent numbers
reproduce exactly** — peak memory (766.5 MB) and GFLOPs are byte-identical to the
3090/L4 runs. Two A2000-specific effects:

1. **Speedup is *higher* than the 3090** (1.85× vs 1.70× at density 0.25). The
   A2000 is far more bandwidth-starved (~288 GB/s vs the 3090's ~936 GB/s), and
   SparseSAM's win is dominated by cutting memory traffic — so it recovers more on
   the slower-bandwidth card. This is the same trend the 3090 doc predicts.
2. **The dense baseline cannot run batch=8** — it OOMs on 12 GB. SparseSAM runs it
   in 1908 MB. Here sparsification doesn't just speed up a workload, it *enables*
   one the baseline can't fit at all.

## Headline (density ↔ speed ↔ memory ↔ FLOPs)

| density  | enc speedup | peak-mem saving | GFLOP saving |
|----------|-------------|-----------------|--------------|
| baseline | 1.00×       | 1.0×            | 1.0×         |
| 0.75     | 1.39×       | 2.9×            | 1.13×        |
| 0.50     | 1.57×       | 2.9×            | 1.30×        |
| 0.25     | 1.85×       | 2.9×            | 1.54×        |

(Encoder-only. Full pipeline including pre/post-processing is **1.42–1.81×** at
1 prompt/image — see [Full end-to-end](#full-end-to-end-including-prepost-processing-batch1).
Accuracy/mIoU not re-run here — it is device-independent and matches the 3090
table to ±0.0001; see [Accuracy](#accuracy).)

---

Details below. Latency / memory / GFLOPs use random-init weights (valid — these
don't depend on trained weights); input 1024×1024, fp16, batch=1 unless noted.

## Encoder: latency / memory / GFLOPs (batch=1)

| algo      | density | latency ms (median) | speedup | peak mem MB | mem saving | GMAC   | FLOP saving |
|-----------|---------|---------------------|---------|-------------|-----------|--------|-------------|
| baseline  | —       | 313.7               | 1.00×   | 2221.5      | 1.0×      | 1487.9 | 1.0×        |
| sparsesam | 0.25    | 169.4               | 1.85×   | 766.5       | 2.9×      | 968.8  | 1.54×       |
| sparsesam | 0.50    | 199.3               | 1.57×   | 766.5       | 2.9×      | 1141.8 | 1.30×       |
| sparsesam | 0.75    | 226.2               | 1.39×   | 766.5       | 2.9×      | 1314.8 | 1.13×       |

GMAC uses the MAC convention; baseline cross-checks against fvcore (1487.9 vs
1493.8, 0.4%). fvcore can't trace the custom sparsesam kernel, so sparsesam GMAC
is hook-measured Linear/Conv + density-scaled attention. These are identical to
the 3090/L4 runs.

## Batch=8 (same model/input)

| algo      | density | latency ms (median) | peak mem MB | note |
|-----------|---------|---------------------|-------------|------|
| baseline  | —       | **OOM**             | —           | dense attention exceeds 12 GB |
| sparsesam | 0.25    | 1376.7              | 1908.1      | fits |
| sparsesam | 0.50    | 1606.3              | 1908.1      | fits |
| sparsesam | 0.75    | 1836.7              | 1908.1      | fits |

The baseline OOMs at batch=8 on this 12 GB card (on the 24 GB 3090 it needs
13766 MB). SparseSAM's fused kernel never materializes the B×H×N×N attention
matrix, so it holds at 1908 MB — the same 1908.1 MB measured on the 3090,
allocator behaviour being device-independent.

## End-to-end vs encoder-only (batch=1)

Full SAM-HQ pipeline (image encoder + prompt encoder + mask decoder), random box
prompts. Encoder fp16; decoder fp32.

**1 prompt/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 327.0  | 10.9   | 338.0  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 175.1  | 11.1   | 186.2  | 1.87×       | 1.81×       |
| sparsesam 0.50 | 203.8  | 11.3   | 215.1  | 1.60×       | 1.57×       |

**10 prompts/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 328.2  | 60.8   | 389.1  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 175.9  | 62.8   | 238.6  | 1.87×       | 1.63×       |
| sparsesam 0.50 | 204.3  | 63.4   | 267.7  | 1.61×       | 1.45×       |

Same Amdahl story: the encoder dominates, so at 1 prompt e2e ≈ encoder speedup;
the unaccelerated decoder dilutes it at 10 prompts (1.87× → 1.63×).

## Full end-to-end, including pre/post-processing (batch=1)

Matches `SamPredictor.set_image` / `.predict` step for step (`bench_e2e_full.py`),
10 real images from `input_imgs/` at native resolution, ms per image, median of 10.

**1 prompt/image**
| algo           | preproc | encoder | prompt | decoder | postproc | d2h  | **e2e ms** | e2e speedup | img/s |
|----------------|---------|---------|--------|---------|----------|------|------------|-------------|-------|
| baseline       | 5.66    | 331.19  | 0.57   | 10.30   | 0.22     | 0.15 | **350.0**  | 1.00×       | 2.9   |
| sparsesam 0.25 | 5.59    | 176.46  | 0.57   | 10.57   | 0.21     | 0.15 | **193.8**  | 1.81×       | 5.2   |
| sparsesam 0.50 | 5.65    | 204.01  | 0.58   | 10.73   | 0.22     | 0.15 | **221.4**  | 1.58×       | 4.5   |
| sparsesam 0.75 | 5.69    | 229.45  | 0.59   | 10.93   | 0.22     | 0.15 | **247.0**  | 1.42×       | 4.0   |

Non-encoder floor: 18.84 ms/image (5.4% of e2e) → Amdahl ceiling 18.6×. The 1.89×
encoder speedup lands almost intact as 1.81× e2e.

**10 prompts/image**
| algo           | preproc | encoder | prompt | decoder | postproc | d2h  | **e2e ms** | e2e speedup | img/s |
|----------------|---------|---------|--------|---------|----------|------|------------|-------------|-------|
| baseline       | 5.72    | 335.87  | 0.58   | 60.29   | 0.98     | 0.85 | **403.8**  | 1.00×       | 2.5   |
| sparsesam 0.25 | 5.63    | 177.47  | 0.59   | 62.35   | 0.97     | 0.85 | **248.0**  | 1.63×       | 4.0   |
| sparsesam 0.50 | 5.70    | 203.89  | 0.58   | 62.95   | 0.97     | 0.86 | **274.8**  | 1.47×       | 3.6   |
| sparsesam 0.75 | 5.69    | 228.17  | 0.60   | 63.62   | 0.98     | 0.87 | **299.9**  | 1.35×       | 3.3   |

Non-encoder floor at 10 prompts: 67.91 ms/image (16.8%), ceiling 5.95×. As on the
other cards, preprocess (single-threaded PIL resize, ~5.7 ms) is untouched by
sparsification and identical across rows; the decoder is the growing fixed cost.

## Implementation gain vs algorithmic gain

`bench_impl_ablation.py` separates the fused-kernel win from the algorithm using a
**fused-dense control** (identical CUTE kernel at ratio=1.0: dense mask, full MLP).
ViT-L, batch=1, 1024², median of 20 iters:

| config              | what it is                                   | lat ms | peak MB | vs A  | vs B  |
|---------------------|----------------------------------------------|--------|---------|-------|-------|
| **A** baseline      | stock attention (manual QKᵀ + rel-pos + PV)  | 329.91 | 2221.5  | 1.00× | 0.79× |
| **B** fused-dense   | same CUTE kernel, dense, full MLP            | 259.40 | 766.5   | 1.27× | 1.00× |
| **C** attn-sparse 0.25 | A-shape sparse attention, full MLP        | 222.34 | 766.5   | 1.48× | 1.17× |
| **C** attn-sparse 0.50 | "                                         | 237.09 | 766.5   | 1.39× | 1.09× |
| **D** sparsesam 0.25 | + keep-token MLP (full method)              | 178.09 | 766.5   | 1.85× | 1.46× |
| **D** sparsesam 0.50 | "                                           | 206.97 | 766.5   | 1.59× | 1.25× |
| **E** tome 0.25     | baseline algorithm, stock attention          | 443.29 | 1992.6  | 0.74× | 0.59× |
| **E** tome 0.50     | "                                            | 444.03 | 1992.6  | 0.74× | 0.58× |

```
sparsesam 0.25:  1.85x total  =  1.27x implementation  x  1.46x algorithmic
sparsesam 0.50:  1.59x total  =  1.27x implementation  x  1.25x algorithmic
```

**The custom kernel is worth 1.27×; the algorithm is worth 1.46× at density 0.25.**
The implementation share is a bit larger here than on the 3090 (1.27× vs 1.12×)
because the A2000's weaker memory subsystem makes the fused kernel's traffic
reduction pay off more even before any sparsification. As on every card, ToMe is
*slower than not merging at all* (443 ms vs 330 ms) — its per-block bipartite
matching cost exceeds the MLP saving; near-identical latency at 0.25/0.50 confirms
matching overhead, not token count, dominates. Latency comparisons against ToMe
measure our kernel vs its reference implementation, not algorithm vs algorithm.

## Accuracy

Not re-run on this GPU — mIoU is device-independent and reproduces the
[3090 table](RESULTS_3090.md#accuracy--miou-coift-box-prompted-sam-hq-vit-l) to
within ±0.0001 (fp16 non-determinism). For reference, SAM-HQ ViT-L on 280 COIFT
images: baseline mIoU 0.9455; density 0.75 → 0.9444 (−0.12%); 0.50 → 0.9419
(−0.38%); 0.25 → 0.9296 (−1.67%). To reproduce here you need the checkpoint +
COIFT data (see Reproduce).

## Why the speedup is higher here (vs the 3090)

Everything device-independent is identical (memory, GMAC); only wall clock differs.
The A2000 is both slower *and* more bandwidth-starved than the 3090:

| | RTX 3090 (SM86) | RTX A2000 (SM86) | ratio |
|---|---|---|---|
| baseline enc, bs=1     | 100.1 ms | 313.7 ms | 3.1× slower |
| sparsesam 0.25, bs=1   | 58.9 ms  | 169.4 ms | 2.9× slower |
| resulting speedup      | 1.70×    | 1.85×    | — |

SparseSAM's gain comes mostly from cutting memory traffic (fused CUTE FA2 +
token merging), and that lever is worth most on a bandwidth-limited GPU. The A2000
has roughly a third of the 3090's memory bandwidth, so its dense baseline is more
traffic-bound and SparseSAM recovers more of it — 1.85× here vs 1.70× on the 3090
at the same 1.54× FLOP cut. Memory savings (2.9×) and accuracy are unaffected and
transfer as-is; on 12 GB the memory saving also unlocks batch=8, which the dense
baseline cannot run.

## Reproduce

```
# deps (this repo's .venv: torch 2.5.1+cu121, cutlass-dsl 4.6.1)
git submodule update --init algos/3rd_party/sam-hq
# core deps already in pyproject.toml + requirements.txt

# latency / memory / FLOPs (no checkpoint or data needed)
PYTHONPATH=$(pwd) python tasks/sam_profile/bench_encoder_l4.py --model-type vit_l --ratios 0.25 0.5 0.75
PYTHONPATH=$(pwd) python tasks/sam_profile/bench_encoder_l4.py --model-type vit_l --ratios 0.25 0.5 0.75 \
    --batch-sizes 8 --iters 10 --warmup 3
PYTHONPATH=$(pwd) python tasks/sam_profile/flops_encoder_l4.py  --model-type vit_l --ratios 0.25 0.5 0.75
PYTHONPATH=$(pwd) python tasks/sam_profile/bench_e2e_l4.py      --model-type vit_l --ratios 0.25 0.5 --prompts 1 10
PYTHONPATH=$(pwd) python tasks/sam_profile/bench_impl_ablation.py --model-type vit_l --ratios 0.25 0.5
PYTHONPATH=$(pwd) python tasks/sam_profile/bench_e2e_full.py    --model-type vit_l --ratios 0.25 0.5 0.75 --prompts 1 10

# accuracy — needs ckts/sam_hq_vit_l.pth + data/thin_object_detection/COIFT
PYTHONPATH=$(pwd) python tasks/sam_hq44k/eval_miou_l4.py --num-samples 280 --ratios 0.25 0.5 0.75
```

Raw logs for this run: `tasks/sam_profile/a2000_out/`.
