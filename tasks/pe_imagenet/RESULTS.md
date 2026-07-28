# Perception Encoder ImageNet zero-shot — measured results

Full ImageNet-1k val (50 000 images) · **PE-Core-L14-336** · batch = 128 ·
fp16 · NVIDIA A100X-20C (sm80) · PyTorch 2.5.1 + CUDA 12.1.

Speedup is `baseline_time / algo_time` over the entire eval pass
(encoder + classifier head, not encoder-only). ΔTop-1 is absolute vs the
dense baseline.

## Dense baseline

| Top-1 | Top-5 | Wall (s) |
|  ---: |  ---: |     ---: |
| 0.835 | 0.974 |      246 |

## PE tuning note

Per the paper's appendix A.5, the Residual-Consistency MLP path "fails
catastrophically" on PE — PE's early MLP updates are spatially selective,
so applying the SAM-tuned token-merging MLP from block 0 hurts a lot.
Either run **attention-only** (`--no-mlp-merge`) or set
**`--partial-start-block 5`** to skip the early blocks. The
SAM-friendly defaults (`start_block=0, mlp_merge=True`) will under-perform
on PE.

## SparseSAM — Stripe-Sort attention (± Residual-Consistency MLP)

| Density | Variant                          | Top-1 (Δ)            | Top-5 | Wall (s) | Speedup   |
|---:     | :---                             | :---:                | :---: |     ---: | :---:     |
|     30% | attn only (`--no-mlp-merge`)     | **0.8145 (−0.021)**  | 0.967 |    191.1 | **×1.29** |
|     30% | attn + MLP, `start_block=5`      | 0.7489 (−0.086)      | 0.936 |    173.7 | **×1.42** |
|     50% | attn only                        | **0.8330 (−0.005)**  | 0.973 |    187.9 | **×1.31** |
|     70% | attn only                        | **0.8350 (+0.000)**  | 0.973 |    194.3 | **×1.27** |

The **attn-only** rows essentially match the dense baseline at r=0.7
(zero Top-1 drop) and stay within 0.005 at r=0.5, while still delivering
~1.3× wall speedup. The attn+MLP row at `start_block=5` trades more
quality for more speed and demonstrates the start-block knob.

## ToMe (bipartite soft matching, attn + MLP)

| Density | Top-1 (Δ)        | Top-5  | Wall (s) | Speedup |
|---:     | :---:            | :---:  |     ---: |   ---:  |
|     30% | 0.5451 (−0.290)  | 0.794  |    306.6 |  ×0.80  |
|     50% | 0.7434 (−0.091)  | 0.935  |    280.2 |  ×0.88  |
|     70% | 0.8069 (−0.028)  | 0.964  |    289.5 |  ×0.85  |

ToMe runs slower than dense at every density on PE-L14-336 — merge +
unmerge bookkeeping doesn't amortize at PE's relatively small token
count (576 tokens for 336/14).

## SpargeAttn (top-k sparse attention, no token reduction)

| Density | Top-1 (Δ)        | Top-5  | Wall (s) | Speedup |
|---:     | :---:            | :---:  |     ---: |   ---:  |
|     30% | 0.4764 (−0.358)  | 0.710  |    252.3 |  ×0.98  |
|     50% | 0.7237 (−0.111)  | 0.922  |    252.6 |  ×0.97  |
|     70% | 0.8022 (−0.033)  | 0.963  |    254.5 |  ×0.97  |

Stays at ~baseline wall time (×0.97); the sparse-attn kernel doesn't help
when token count is small, and quality drops faster than SparseSAM at
every density.

## Takeaways

- **SparseSAM (attn-only) is essentially lossless on PE** at r ≥ 0.5 while
  giving ~1.3× wall speedup. At r=0.3 it loses only 0.021 Top-1 — vs
  ToMe's 0.290 and SpargeAttn's 0.358 drops at the same density.
- **ToMe runs slower than dense** at every density (overhead doesn't
  amortize at 576-token PE).
- **SpargeAttn** ≈ baseline wall time; quality also drops faster than
  SparseSAM.

## Reproduce

```bash
# Main sweep (attn+MLP, start_block=0 — SAM-tuned defaults; suboptimal on PE)
python tasks/pe_imagenet/eval_pe_clip.py \
    --model PE-Core-L14-336 \
    --dataset imagenet1k --dataset-root ./data/imagenet \
    --batch-size 128 --dtype fp16 \
    --algorithm none sparsesam_partial tome_partial sparge \
    --ratio 0.30 0.50 0.70

# Attn-only (paper-recommended PE config)
python tasks/pe_imagenet/eval_pe_clip.py \
    --model PE-Core-L14-336 \
    --dataset imagenet1k --dataset-root ./data/imagenet \
    --batch-size 128 --dtype fp16 \
    --algorithm sparsesam_partial \
    --ratio 0.30 0.50 0.70 --no-mlp-merge

# attn+MLP with start_block=5 (single config)
python tasks/pe_imagenet/eval_pe_clip.py \
    --model PE-Core-L14-336 \
    --dataset imagenet1k --dataset-root ./data/imagenet \
    --batch-size 128 --dtype fp16 \
    --algorithm sparsesam_partial \
    --ratio 0.30 --partial-start-block 5
```

Raw CSV (main sweep):
[`benchmark_results/pe_clip_PE-Core-L14-336_20260522_211041.csv`](../../benchmark_results/pe_clip_PE-Core-L14-336_20260522_211041.csv).
Ablation CSVs live under `benchmark_results/` with the same prefix.
