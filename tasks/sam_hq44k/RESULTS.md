# SAM-HQ HQ-44K — measured results

All numbers measured in this repo (not copied from the paper) on
**NVIDIA A100X-20C (sm80)** · PyTorch 2.5.1 + CUDA 12.1 · SAM-HQ ViT-L ·
batch = 8 · full datasets (470 imgs DIS5K-VD, 500 imgs ThinObject5K-TE).

Speedup is `baseline_enc_ms / algo_enc_ms` (>1 = faster, **bold** = best
in column). ΔmIoU is absolute vs the dense baseline.

## Dense baseline

| Dataset           |   mIoU |  B-IoU | enc/img | Peak GPU |
|---                |   ---: |   ---: |    ---: |     ---: |
| DIS5K-VD          | 0.7863 | 0.7060 | 55.7 ms | 14130 MB |
| ThinObject5K-TE   | 0.8956 | 0.7962 | 52.9 ms | 14130 MB |

## SparseSAM — Stripe-Sort attention (± Residual-Consistency MLP)

Ablation: attention-only (Stripe-Sort) vs. full SparseSAM
(Stripe-Sort + Residual-Consistency MLP).

| Density | Variant            |    DIS5K-VD mIoU (Δ) |   Speedup |    ThinObj mIoU (Δ) |   Speedup | Peak GPU |
|---:     | :---               |                 ---: |      ---: |                ---: |      ---: |     ---: |
|     30% | attn only          |      0.7718 (−0.015) |     ×1.71 |     0.8907 (−0.005) |     ×1.66 |  2271 MB |
|     30% | attn + MLP (full)  |      0.7500 (−0.036) | **×2.04** |     0.8756 (−0.020) | **×2.02** |  2272 MB |
|     50% | attn only          |      0.7838 (−0.003) |     ×1.64 |     0.8941 (−0.001) |     ×1.59 |  2277 MB |
|     50% | attn + MLP (full)  |      0.7756 (−0.011) | **×1.89** |     0.8921 (−0.003) | **×1.85** |  2277 MB |
|     70% | attn only          |  **0.7847 (−0.002)** |     ×1.61 | **0.8965 (+0.001)** |     ×1.56 |  2282 MB |
|     70% | attn + MLP (full)  |      0.7819 (−0.004) | **×1.78** |     0.8960 (+0.000) | **×1.73** |  2282 MB |

Toggle via `mlp_merge=True/False` in `apply_sam(..., name="sparsesam", mlp_merge=...)`
or `--mlp-merge` / `--no-mlp-merge` on the CLI. The full Residual-Consistency
MLP path adds ~0.2× extra speedup at every density; quality cost is ≤0.004
mIoU at 70% density and grows to 0.022 mIoU on DIS5K-VD at 30%. Either
variant is the only algorithm here that simultaneously drops both latency
and memory — **~2× encoder speedup with ~84% less GPU memory** vs the
dense baseline.

## ToMe — bipartite soft matching (attn + MLP)

| Density |  DIS5K-VD mIoU (Δ) |       ThinObj mIoU (Δ) | Speedup (DIS / Thin) | Peak GPU |
|---:     |               ---: |                   ---: |                 ---: |     ---: |
|     30% |    0.6970 (−0.089) |        0.8428 (−0.053) |         ×0.79 / ×0.82 | 12129 MB |
|     50% |    0.6970 (−0.089) |        0.8428 (−0.053) |         ×0.80 / ×0.82 | 12129 MB |
|     70% |    0.7650 (−0.021) |        0.8906 (−0.005) |         ×0.49 / ×0.50 | 13795 MB |

Merge + unmerge overhead exceeds the savings → encoder runs slower than
dense; memory drops only modestly.

## GradToMe / StructSAM — gradient-aware bipartite matching

| Density |  DIS5K-VD mIoU (Δ) |       ThinObj mIoU (Δ) | Speedup (DIS / Thin) | Peak GPU |
|---:     |               ---: |                   ---: |                 ---: |     ---: |
|     30% |    0.5608 (−0.226) |        0.7138 (−0.182) |         ×0.72 / ×0.74 | 12366 MB |
|     50% |    0.6850 (−0.101) |        0.8268 (−0.069) |         ×0.65 / ×0.65 | 14033 MB |
|     70% |    0.7561 (−0.030) |        0.8830 (−0.013) |         ×0.56 / ×0.58 | 15694 MB |

Catastrophic quality drop at low density (especially thin structures);
also runs slower than dense, and at 70% density uses *more* GPU memory
than the dense baseline.

## SpargeAttn — top-k sparse attention (no token reduction)

| Density |  DIS5K-VD mIoU (Δ) |       ThinObj mIoU (Δ) | Speedup (DIS / Thin) | Peak GPU |
|---:     |               ---: |                   ---: |                 ---: |     ---: |
|     30% |    0.7350 (−0.051) |        0.8539 (−0.042) |         ×1.12 / ×1.13 | 14027 MB |
|     50% |    0.7709 (−0.015) |        0.8882 (−0.007) |         ×1.10 / ×1.13 | 14027 MB |
|     70% |    0.7713 (−0.015) |        0.8898 (−0.006) |         ×1.12 / ×1.11 | 14027 MB |

Quality stays close to dense and a modest ~1.1× speedup, but no token
reduction → memory is essentially unchanged from the baseline.

## Reproduce

```bash
# main sweep (full Residual-Consistency MLP path)
python tasks/sam_hq44k/eval_hq44k.py \
    --algos none sparsesam tome gradtome sparge \
    --ratios 0.30 0.50 0.70 \
    --batch-sizes 8 --num-samples 500 \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
    --dataset-idx 0 1 --no-wandb --no-plot --mlp-merge

# attn-only ablation (swap --mlp-merge for --no-mlp-merge)
python tasks/sam_hq44k/eval_hq44k.py \
    --algos sparsesam --ratios 0.30 0.50 0.70 \
    --batch-sizes 8 --num-samples 500 \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
    --dataset-idx 0 1 --no-wandb --no-plot --no-mlp-merge
```

Raw CSVs:
- main: [`benchmark_results/tome_benchmark_20260522_191155.csv`](../../benchmark_results/tome_benchmark_20260522_191155.csv)
- ablation: [`benchmark_results/tome_benchmark_20260522_191816.csv`](../../benchmark_results/tome_benchmark_20260522_191816.csv)
