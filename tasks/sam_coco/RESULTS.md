# SAM COCO DINO — measured results

All numbers below were measured with this repo on MS-COCO validation using detector-proposed boxes. Each run used the first **500** COCO images, batch size **1**, and an fp16 SAM predictor.

Speedup is `baseline_enc_ms / algo_enc_ms` (>1 = faster). `ΔmAP` is absolute vs the dense baseline for the same backbone.

## ViT-L

Detector: **FocalNet-DINO** · SAM-HQ **vit_l**

### Dense baseline

| mAP | mAP50 | mAP75 | enc/img | Peak GPU | dtype |
|---:|---:|---:|---:|---:|---:|
| 0.532 | 0.787 | 0.578 | 57.4 ms | 7565 MB | float16 |

### SparseSAM — Stripe-Sort attention (± Residual-Consistency MLP)

| Density | Variant | mAP (Δ) | Speedup | enc/img | Peak GPU | dtype |
|---:|:---|---:|---:|---:|---:|---:|
| 30% | SparseSAM attention only | 0.526 (−0.006) | ×1.65 | 34.7 ms | 7573 MB | float16 |
| 50% | SparseSAM attention only | 0.532 (+0.000) | ×1.59 | 36.0 ms | 7573 MB | float16 |
| 70% | SparseSAM attention only | 0.532 (+0.000) | ×1.55 | 37.1 ms | 7573 MB | float16 |
| 30% | SparseSAM full | 0.519 (−0.013) | ×1.95 | 29.4 ms | 7573 MB | float16 |
| 50% | SparseSAM full | 0.531 (−0.001) | ×1.79 | 32.1 ms | 7573 MB | float16 |
| 70% | SparseSAM full | 0.534 (+0.002) | ×1.68 | 34.2 ms | 7573 MB | float16 |

### Piecewise Sparse Attention

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | dtype |
|---:|---:|---:|---:|---:|---:|
| 30% | 0.522 (−0.010) | ×0.70 | 82.5 ms | 7566 MB | float16 |
| 50% | 0.531 (−0.001) | ×0.68 | 83.9 ms | 7566 MB | float16 |
| 70% | 0.531 (−0.001) | ×0.68 | 84.2 ms | 7566 MB | float16 |

### SpargeAttn — top-k sparse attention

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | dtype |
|---:|---:|---:|---:|---:|---:|
| 30% | 0.500 (−0.032) | ×1.00 | 57.5 ms | 7566 MB | float16 |
| 50% | 0.529 (−0.003) | ×0.99 | 58.0 ms | 7566 MB | float16 |
| 70% | 0.529 (−0.003) | ×0.98 | 58.3 ms | 7566 MB | float16 |

### ToMe — bipartite soft matching

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | dtype |
|---:|---:|---:|---:|---:|---:|
| 30% | 0.465 (−0.067) | ×0.82 | 70.4 ms | 7582 MB | float16 |
| 50% | 0.465 (−0.067) | ×0.81 | 70.7 ms | 7582 MB | float16 |
| 70% | 0.527 (−0.005) | ×0.65 | 87.8 ms | 7583 MB | float16 |

### GradToMe / StructSAM — gradient-aware matching

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | dtype |
|---:|---:|---:|---:|---:|---:|
| 30% | 0.372 (−0.160) | ×0.81 | 70.6 ms | 7567 MB | float16 |
| 50% | 0.487 (−0.045) | ×0.72 | 79.6 ms | 7566 MB | float16 |
| 70% | 0.526 (−0.006) | ×0.62 | 91.9 ms | 7566 MB | float16 |

Raw run directory:
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA_Quantization/chi/VLA_Quantization/sparsesam/SparseSAM/benchmark_results/sam_coco/dino_fp16_500_20260526_041810`

## ViT-B

Detector: **FocalNet-DINO** · SAM-HQ **vit_b**

### Dense baseline

| mAP | mAP50 | mAP75 | enc/img | Peak GPU | dtype |
|---:|---:|---:|---:|---:|---:|
| 0.509 | 0.769 | 0.539 | 29.6 ms | 6309 MB | float16 |

### SparseSAM — Stripe-Sort attention (± Residual-Consistency MLP)

| Density | Variant | mAP (Δ) | Speedup | enc/img | Peak GPU | dtype |
|---:|:---|---:|---:|---:|---:|---:|
| 30% | SparseSAM attention only | 0.502 (−0.007) | ×1.99 | 14.8 ms | 6316 MB | float16 |
| 50% | SparseSAM attention only | 0.507 (−0.002) | ×1.88 | 15.7 ms | 6316 MB | float16 |
| 70% | SparseSAM attention only | 0.508 (−0.001) | ×1.79 | 16.5 ms | 6316 MB | float16 |
| 30% | SparseSAM full | 0.498 (−0.011) | ×2.02 | 14.6 ms | 6316 MB | float16 |
| 50% | SparseSAM full | 0.507 (−0.002) | ×2.02 | 14.6 ms | 6317 MB | float16 |
| 70% | SparseSAM full | 0.508 (−0.001) | ×1.90 | 15.6 ms | 6316 MB | float16 |

### Piecewise Sparse Attention

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | dtype |
|---:|---:|---:|---:|---:|---:|
| 30% | 0.498 (−0.011) | ×0.69 | 42.7 ms | 6312 MB | float16 |
| 50% | 0.508 (−0.001) | ×0.69 | 43.1 ms | 6312 MB | float16 |
| 70% | 0.508 (−0.001) | ×0.68 | 43.4 ms | 6312 MB | float16 |

### SpargeAttn — top-k sparse attention

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | dtype |
|---:|---:|---:|---:|---:|---:|
| 30% | 0.491 (−0.018) | ×1.09 | 27.1 ms | 6309 MB | float16 |
| 50% | 0.503 (−0.006) | ×1.09 | 27.2 ms | 6309 MB | float16 |
| 70% | 0.504 (−0.005) | ×1.07 | 27.6 ms | 6309 MB | float16 |

### ToMe — bipartite soft matching

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | dtype |
|---:|---:|---:|---:|---:|---:|
| 30% | 0.451 (−0.058) | ×0.76 | 39.0 ms | 6325 MB | float16 |
| 50% | 0.449 (−0.060) | ×0.77 | 38.5 ms | 6325 MB | float16 |
| 70% | 0.506 (−0.003) | ×0.61 | 48.7 ms | 6323 MB | float16 |

### GradToMe / StructSAM — gradient-aware matching

| Density | mAP (Δ) | Speedup | enc/img | Peak GPU | dtype |
|---:|---:|---:|---:|---:|---:|
| 30% | 0.354 (−0.155) | ×0.77 | 38.7 ms | 6310 MB | float16 |
| 50% | 0.460 (−0.049) | ×0.68 | 43.2 ms | 6311 MB | float16 |
| 70% | 0.501 (−0.008) | ×0.58 | 50.7 ms | 6311 MB | float16 |

Raw run directory:
- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA_Quantization/chi/VLA_Quantization/sparsesam/SparseSAM/benchmark_results/sam_coco/dino_fp16_vit_b_500_20260526_065546`

## Reproduce

```bash
MODEL_TYPE=vit_l DETECTOR=dino bash tasks/sam_coco/eval_coco.sh
MODEL_TYPE=vit_b DETECTOR=dino bash tasks/sam_coco/eval_coco.sh
```
