# SAM COCO DINO — measured results

All numbers below were measured with this repo on MS-COCO validation using
detector-proposed boxes. Each run used the first **500** COCO images, batch
size **1**, and an fp16 SAM predictor, on a single **NVIDIA A100-SXM4-80GB**.

Speedup is `baseline_enc_ms / algo_enc_ms` (>1 = faster). `ΔmAP` is absolute vs
the dense baseline for the same backbone.

## Reading the memory columns

Memory is scoped to the **SAM image encoder**: the CUDA peak counter is reset
immediately before each encoder forward and read immediately after.

- **Peak GPU** — resident fp16 SAM weights plus the encoder's activation peak,
  as one absolute number. This is the same convention used in
  [`tasks/sam_hq44k/RESULTS.md`](../sam_hq44k/RESULTS.md), so the two files can
  be read side by side (see the caveat below). SAM weights are 598 MB for ViT-L
  and 181 MB for ViT-B; every algorithm here is parameter-free, so that term is
  constant within a backbone.
- **Enc act** — encoder activations alone, i.e. `Peak GPU` minus the weights.
  This is the part that actually responds to the algorithm.

The co-resident FocalNet-DINO detector is deliberately excluded from both. A
whole-process peak — which is what an earlier version of this file reported —
is dominated by the detector forward and by the mask decoder upsampling every
detected box to original resolution, both algorithm-independent. It read
~7565 MB for *every* ViT-L configuration regardless of sparsity. That number is
still recorded per run as `pipeline_peak_memory_mb` in the CSVs, along with
`encoder_peak_memory_mb` (encoder high-water including detector weights), but
neither is useful for comparing algorithms.

### Comparing against HQ-44K

`Peak GPU` here and `Peak GPU` in the HQ-44K results use the same definition but
**different batch sizes**: HQ-44K ran at batch 8, COCO runs at batch 1 because
the detector pipeline feeds `predictor.set_image()` one image at a time.
Activations scale with batch, weights do not, so the correct conversion is:

```
hq44k_peak  ≈  8 × (coco Enc act)  +  weights + per-batch resident tensors
```

Checked across all 13 shared configurations, that residual term is constant at
**1077 ± 62 MB** (598 MB of SAM weights, the rest being the feature maps, input
batch and full-resolution GT masks HQ-44K holds for 8 images). The two
benchmarks agree; only the batch size differs.

## ViT-L

Detector: **FocalNet-DINO** · SAM-HQ **vit_l**

### Dense baseline

| mAP | mAP50 | mAP75 | enc/img | Peak GPU | Enc act | dtype |
|---:|---:|---:|---:|---:|---:|---:|
| 0.532 | 0.787 | 0.578 | 57.5 ms | 2217 MB | 1619 MB | float16 |

### SparseSAM — Stripe-Sort attention (± Residual-Consistency MLP)

| Density | Variant | mAP (Δ) | Speedup | enc/img | Peak GPU | Enc act | dtype |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 30% | SparseSAM attention only | 0.526 (−0.006) | ×1.65 | 35.0 ms | 754 MB | 156 MB | float16 |
| 50% | SparseSAM attention only | 0.532 (+0.000) | ×1.59 | 36.3 ms | 754 MB | 156 MB | float16 |
| 70% | SparseSAM attention only | 0.532 (+0.000) | ×1.54 | 37.3 ms | 754 MB | 156 MB | float16 |
| 30% | SparseSAM full | 0.519 (−0.013) | ×1.96 | 29.3 ms | 754 MB | 156 MB | float16 |
| 50% | SparseSAM full | 0.531 (−0.001) | ×1.79 | 32.1 ms | 753 MB | 155 MB | float16 |
| 70% | SparseSAM full | 0.534 (+0.002) | ×1.68 | 34.3 ms | 754 MB | 156 MB | float16 |

### Piecewise Sparse Attention

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | Enc act | dtype |
|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.522 (−0.010) | ×0.70 | 82.5 ms | 3256 MB | 2658 MB | float16 |
| 50% | 0.531 (−0.001) | ×0.69 | 83.8 ms | 3256 MB | 2658 MB | float16 |
| 70% | 0.531 (−0.001) | ×0.68 | 84.1 ms | 3256 MB | 2658 MB | float16 |

### SpargeAttn — top-k sparse attention

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | Enc act | dtype |
|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.500 (−0.032) | ×1.01 | 57.1 ms | 2217 MB | 1619 MB | float16 |
| 50% | 0.529 (−0.003) | ×1.00 | 57.6 ms | 2217 MB | 1619 MB | float16 |
| 70% | 0.529 (−0.003) | ×0.99 | 58.0 ms | 2217 MB | 1619 MB | float16 |

### ToMe — bipartite soft matching

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | Enc act | dtype |
|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.465 (−0.067) | ×0.82 | 70.5 ms | 1970 MB | 1371 MB | float16 |
| 50% | 0.465 (−0.067) | ×0.82 | 70.4 ms | 1970 MB | 1371 MB | float16 |
| 70% | 0.527 (−0.005) | ×0.66 | 87.5 ms | 2178 MB | 1580 MB | float16 |

### GradToMe / StructSAM — gradient-aware matching

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | Enc act | dtype |
|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.372 (−0.160) | ×0.82 | 69.8 ms | 2018 MB | 1420 MB | float16 |
| 50% | 0.487 (−0.045) | ×0.73 | 79.1 ms | 2226 MB | 1627 MB | float16 |
| 70% | 0.526 (−0.006) | ×0.63 | 91.4 ms | 2433 MB | 1835 MB | float16 |

Raw run directories:
- `benchmark_results/sam_coco/vit_l_mlp`
- `benchmark_results/sam_coco/vit_l_attnonly`

## ViT-B

Detector: **FocalNet-DINO** · SAM-HQ **vit_b**

### Dense baseline

| mAP | mAP50 | mAP75 | enc/img | Peak GPU | Enc act | dtype |
|---:|---:|---:|---:|---:|---:|---:|
| 0.509 | 0.769 | 0.539 | 28.6 ms | 1395 MB | 1214 MB | float16 |

### SparseSAM — Stripe-Sort attention (± Residual-Consistency MLP)

| Density | Variant | mAP (Δ) | Speedup | enc/img | Peak GPU | Enc act | dtype |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 30% | SparseSAM attention only | 0.502 (−0.007) | ×1.89 | 15.1 ms | 298 MB | 117 MB | float16 |
| 50% | SparseSAM attention only | 0.507 (−0.002) | ×1.80 | 15.9 ms | 298 MB | 117 MB | float16 |
| 70% | SparseSAM attention only | 0.508 (−0.001) | ×1.70 | 16.8 ms | 298 MB | 117 MB | float16 |
| 30% | SparseSAM full | 0.498 (−0.011) | ×1.96 | 14.6 ms | 298 MB | 117 MB | float16 |
| 50% | SparseSAM full | 0.507 (−0.002) | ×1.98 | 14.5 ms | 298 MB | 117 MB | float16 |
| 70% | SparseSAM full | 0.508 (−0.001) | ×1.83 | 15.6 ms | 298 MB | 117 MB | float16 |

### Piecewise Sparse Attention

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | Enc act | dtype |
|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.498 (−0.011) | ×0.67 | 42.7 ms | 2174 MB | 1993 MB | float16 |
| 50% | 0.508 (−0.001) | ×0.66 | 43.3 ms | 2174 MB | 1993 MB | float16 |
| 70% | 0.508 (−0.001) | ×0.66 | 43.6 ms | 2174 MB | 1994 MB | float16 |

### SpargeAttn — top-k sparse attention

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | Enc act | dtype |
|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.491 (−0.018) | ×1.06 | 27.0 ms | 1395 MB | 1214 MB | float16 |
| 50% | 0.503 (−0.006) | ×1.05 | 27.3 ms | 1395 MB | 1214 MB | float16 |
| 70% | 0.504 (−0.005) | ×1.04 | 27.6 ms | 1395 MB | 1214 MB | float16 |

### ToMe — bipartite soft matching

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | Enc act | dtype |
|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.449 (−0.060) | ×0.74 | 38.5 ms | 1211 MB | 1030 MB | float16 |
| 50% | 0.450 (−0.059) | ×0.74 | 38.5 ms | 1211 MB | 1030 MB | float16 |
| 70% | 0.506 (−0.003) | ×0.59 | 48.6 ms | 1367 MB | 1186 MB | float16 |

### GradToMe / StructSAM — gradient-aware matching

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | Enc act | dtype |
|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.354 (−0.155) | ×0.74 | 38.5 ms | 1246 MB | 1065 MB | float16 |
| 50% | 0.460 (−0.049) | ×0.66 | 43.3 ms | 1402 MB | 1221 MB | float16 |
| 70% | 0.501 (−0.008) | ×0.56 | 50.9 ms | 1558 MB | 1377 MB | float16 |

Raw run directories:
- `benchmark_results/sam_coco/vit_b_mlp`
- `benchmark_results/sam_coco/vit_b_attnonly`

## Notes on the memory results

**SparseSAM's footprint is flat across densities** (156 MB activations on ViT-L,
117 MB on ViT-B at every density tested; 754 / 298 MB Peak GPU). Density buys
latency, not memory: the win comes
from the fused stripe-sort kernel never materializing the N×N attention matrix
at all, which is a fixed structural saving. Enabling the Residual-Consistency
MLP costs no additional activation memory.

**Token-merging methods scale with density but start from the dense cost.**
ToMe and GradToMe still run full attention before merging, so their activations
track the kept-token count and only fall below the dense baseline at low
density. GradToMe at 70% exceeds the dense baseline outright (1835 vs 1619 MB on
ViT-L) because the similarity and scatter buffers are added on top of, not in
place of, the attention matrix. The same inversion appears in the HQ-44K
results.

**SpargeAttn is memory-neutral.** Its top-k selection runs on top of the full
attention path, so activations sit at exactly the dense value (1619 MB ViT-L /
1214 MB ViT-B) at every density, matching its ~×1.0 speedup.

**Piecewise is the most expensive.** It materializes block-sparse masks in
addition to the attention matrix, costing ~1.6× the dense baseline on both
backbones while also running slower than dense.

**ToMe's 30% and 50% rows are identical**, on both backbones and in every
column. This is not a copy-paste error: ToMe's per-layer merge schedule
saturates, so requesting a density below ~50% merges the same number of tokens
as 50%. Treat those as one data point.

## Reproduce

```bash
MODEL_TYPE=vit_l DETECTOR=dino bash tasks/sam_coco/eval_coco.sh
MODEL_TYPE=vit_b DETECTOR=dino bash tasks/sam_coco/eval_coco.sh

# SparseSAM attention-only variant (no Residual-Consistency MLP)
MODEL_TYPE=vit_l DETECTOR=dino ALGOS=sparsesam MLP_MERGE=no bash tasks/sam_coco/eval_coco.sh
```
