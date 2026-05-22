"""Fused CUDA kernels (cutlass DSL) used by sparsesam patches.

  • `flash_attn_rel_pos` — FA2 + SAM's decomposed (h_rel, w_rel) relative-position
    bias. Used by the SAM-HQ sparsesam patch.
  • `flash_attn_rope_fused` — FA2 + 2D-axial RoPE applied in-kernel. Used by
    the PE / SigLIP sparsesam_partial patches.

Both are imported lazily by their callers — importing this package does not
compile or instantiate the kernels.
"""
