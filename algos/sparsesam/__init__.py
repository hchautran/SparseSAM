"""SparseSAM — Z-group merge + block-sparse cute-kernel attention.

Patches: `pe_compress.py`, `pe_partial.py`, `sam.py`, `sam_random.py`.
Helpers: `merge.py` (tile_stride_matching), `z_utils.py` (z-order perms),
`hilbert_utils.py` (Hilbert-curve perms).

Public entry via `algos.registry.apply_pe / apply_sam`.
"""
