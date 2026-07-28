"""SparseSAM token-compression / attention algorithms.

The registry, public apply/remove API, and built-in algorithm registration
all live in `algos.registry`. This `__init__` is intentionally empty so that
importing a single submodule (e.g. `algos.sparsesam.sam`) does NOT eagerly
drag in PE-side patches — useful on hosts where `perception_models` is not
installed.

Public API (lazy):

    from algos.registry import (
        apply_pe, apply_sam,
        remove_all_pe, remove_all_sam,
        update_sam_ratio,
        REGISTRY, AlgoSpec, register, choices, spec_of,
        algo_choices, sam_algo_choices,
    )

Supported backbones: **SAM** (SAM-HQ ViT) and **PE** (Perception Encoder).

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
