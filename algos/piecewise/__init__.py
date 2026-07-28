"""Piecewise Sparse Attention (PISA) integration.

The SAM patch is imported lazily by `algos.registry` so the project remains
usable when the optional `piecewise_attn` Triton package is not installed.
"""

