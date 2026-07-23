# SparseSAM benchmark on NVIDIA L4 (SM89)

SAM-HQ **ViT-L**, GPU: NVIDIA L4, CUDA 12.8, torch 2.8.0+cu128, cutlass-dsl 4.6.1.
The SparseSAM FA2 CUTE kernel was written for Ampere SM80 (A100) but runs correctly
on the L4 (Ada SM89) — reproducing the paper's ~2× speed / 2.8× memory / <1% IoU
headline on a different, smaller GPU.

## Headline (density ↔ speed ↔ memory ↔ accuracy)

| density  | enc speedup | peak-mem saving | GFLOP saving | mIoU (COIFT) | Δ mIoU  |
|----------|-------------|-----------------|--------------|--------------|---------|
| baseline | 1.00×       | 1.0×            | 1.0×         | 0.9455       | —       |
| 0.75     | 1.75×       | 2.9×            | 1.13×        | 0.9444       | −0.11%  |
| 0.50     | 1.96×       | 2.9×            | 1.30×        | 0.9419       | −0.38%  |
| 0.25     | 2.28×       | 2.9×            | 1.54×        | 0.9297       | −1.67%  |

(Memory saving grows to **7.2×** at batch=8 — see below. Speedup is encoder-only;
end-to-end is 1.8–2.2× depending on prompts/image.)

---

Details below. Latency / memory / GFLOPs use random-init weights (valid — these
don't depend on trained weights); input 1024×1024, fp16, batch=1 unless noted.

## Encoder: latency / memory / GFLOPs (batch=1)

| algo      | density | latency ms (median) | speedup | peak mem MB | mem saving | GMAC   | FLOP saving |
|-----------|---------|---------------------|---------|-------------|-----------|--------|-------------|
| baseline  | —       | 184.7               | 1.00×   | 2221.5      | 1.0×      | 1487.9 | 1.0×        |
| sparsesam | 0.25    | 80.9                | 2.28×   | 765.3       | 2.9×      | 968.8  | 1.54×       |
| sparsesam | 0.50    | 94.2                | 1.96×   | 765.3       | 2.9×      | 1141.8 | 1.30×       |
| sparsesam | 0.75    | 105.3               | 1.75×   | 765.3       | 2.9×      | 1314.8 | 1.13×       |

## Batch=8 (same model/input)

| algo      | density | latency ms (median) | speedup | peak mem MB | mem saving |
|-----------|---------|---------------------|---------|-------------|-----------|
| baseline  | —       | 1548.9              | 1.00×   | 13766.2     | 1.0×      |
| sparsesam | 0.25    | 740.4               | 2.09×   | 1908.1      | 7.2×      |
| sparsesam | 0.50    | 839.1               | 1.85×   | 1908.1      | 7.2×      |
| sparsesam | 0.75    | 944.8               | 1.64×   | 1908.1      | 7.2×      |

**Memory saving grows with batch:** baseline peak scales ~6× from batch 1→8
(2221→13766 MB) while sparsesam scales only ~2.5× (765→1908 MB), so the mem
advantage widens 2.9× → 7.2×. Latency speedup stays ~2×.

Notes
- Latency speedup exceeds FLOP reduction → gain is from the fused CUTE FA2 kernel +
  reduced memory traffic (token merge), not FLOPs alone.
- GFLOPs use the MAC convention; baseline cross-checks against fvcore (1487.9 vs
  1493.8 GMAC, 0.4%). fvcore can't trace sparsesam's custom kernel, so sparsesam
  FLOPs are hook-measured Linear/Conv MACs + density-scaled attention.

## End-to-end vs encoder-only (batch=1)

Full SAM-HQ pipeline (image encoder + prompt encoder + mask decoder), random box
prompts. Encoder runs fp16; decoder fp32 (this sam-hq HEAD is not half-clean).

**1 prompt/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 181.8  | 6.1    | 187.9  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 79.3   | 6.1    | 85.5   | 2.29×       | 2.20×       |
| sparsesam 0.50 | 91.4   | 6.7    | 98.1   | 1.99×       | 1.91×       |

**10 prompts/image**
| algo           | enc ms | dec ms | e2e ms | enc speedup | e2e speedup |
|----------------|--------|--------|--------|-------------|-------------|
| baseline       | 181.0  | 44.2   | 225.2  | 1.00×       | 1.00×       |
| sparsesam 0.25 | 78.5   | 44.5   | 123.0  | 2.31×       | 1.83×       |
| sparsesam 0.50 | 91.7   | 44.4   | 136.1  | 1.97×       | 1.65×       |

The encoder dominates, so with few prompts e2e ≈ encoder speedup (2.2×). The
decoder is unaccelerated fixed cost, so many prompts/image dilute the speedup
(Amdahl): 10 prompts → e2e 1.83×.

## Accuracy — mIoU (COIFT, box-prompted, SAM-HQ ViT-L)

280 COIFT images, box prompts from GT. Higher density = less sparsification.

| density  | mIoU   | Δ mIoU            | Boundary IoU |
|----------|--------|-------------------|--------------|
| baseline | 0.9455 | —                 | 0.8959       |
| 0.75     | 0.9444 | −0.0010 (−0.11%)  | 0.8946       |
| 0.50     | 0.9419 | −0.0036 (−0.38%)  | 0.8915       |
| 0.25     | 0.9297 | −0.0158 (−1.67%)  | 0.8720       |

Confirms the paper's "<1% IoU loss" at density ≥0.5. Clean accuracy/speed tradeoff:
density 0.25 → 2.28× encoder speedup at −1.67% mIoU; density 0.75 → essentially
lossless (−0.1%).

Reproduce: `PYTHONPATH=/SparseSAM python tasks/sam_hq44k/eval_miou_l4.py --num-samples 280 --ratios 0.25 0.5 0.75`
(needs `ckts/sam_hq_vit_l.pth` + `data/thin_object_detection/COIFT`).

Reproduce:
```
PYTHONPATH=/SparseSAM python tasks/sam_profile/bench_encoder_l4.py --model-type vit_l --ratios 0.25 0.5 0.75
PYTHONPATH=/SparseSAM python tasks/sam_profile/flops_encoder_l4.py --model-type vit_l --ratios 0.25 0.5 0.75
```
