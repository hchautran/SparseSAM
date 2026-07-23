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
end-to-end is 1.3–1.6× depending on prompts/image.)

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

# accuracy — needs ckts/sam_hq_vit_l.pth + data/thin_object_detection/COIFT
PYTHONPATH=/SparseSAM python tasks/sam_hq44k/eval_miou_l4.py --num-samples 280 --ratios 0.25 0.5 0.75
```

Assets used for the accuracy run:
- checkpoint: `huggingface.co/lkeab/hq-sam` → `sam_hq_vit_l.pth` (1.25 GB) into `ckts/`
- COIFT: `thin_object_detection.zip` → extract `COIFT/` into `data/thin_object_detection/`
  (280 images + 280 masks)
