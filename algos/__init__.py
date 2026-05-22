"""SparseSAM token-compression / attention algorithms.

The registry, public apply/remove API, and built-in algorithm registration
all live in `algos.registry`. This `__init__` is intentionally empty so
that importing a single submodule (e.g. `algos.sparsesam.patch.sam2_hiera`)
does NOT eagerly drag in PE / SigLIP / MViT-side patches — useful on hosts
where those frameworks (perception_models, transformers' siglip, timm
mvitv2) are not installed.

Public API (lazy):

    from algos.registry import (
        apply_pe, apply_sam, apply_siglip, apply_mvit,
        remove_all_pe, remove_all_sam, remove_all_siglip, remove_all_mvit,
        update_sam_ratio,
        REGISTRY, AlgoSpec, register, choices, spec_of,
        algo_choices, sam_algo_choices, siglip_algo_choices, mvit_algo_choices,
    )

Algorithm subpackages:
  • `algos.tome`       — Token Merging (bipartite soft matching).
  • `algos.gradtome`   — Gradient-aware bipartite matching on the spatial grid.
  • `algos.sparsesam`  — Z-group merge + block-sparse cute-kernel attention.
  • `algos.sparge`     — SpargeAttn drop-in sparse attention (no token merge).
  • `algos.kernels`    — fused cutlass-DSL CUDA kernels (FA2 + rel-pos / RoPE).

Adding a new algorithm: write the patch, then add a single
`register(AlgoSpec(...))` call in `algos.registry._register_builtins`.
See `docs/ADDING_ALGORITHMS.md`.
"""
