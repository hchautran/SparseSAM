"""Token Merging (ToMe) — bipartite soft matching.

Patches: `pe_compress.py`, `pe_partial.py`, `sam.py`, `siglip.py`.
Primitives: `merge.py` (bipartite_soft_matching, consecutive_soft_matching, merge_wavg).

Public entry via `algos.registry.apply_pe / apply_sam / apply_siglip`.
"""
