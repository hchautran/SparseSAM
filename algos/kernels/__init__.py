"""Fused CUDA kernels (cutlass DSL) used by sparsesam patches.

  • `flash_attn.FlashAttentionForwardAmpereRelPos` — FA2 + SAM's decomposed
    (h_rel, w_rel) relative-position bias. Used by the SAM-HQ sparsesam patch.
    (Also exported as `FlashAttentionForwardAmpere` for back-compat.)
  • `flash_attn.FlashAttentionForwardAmpereRoPE` — FA2 + 2D-axial RoPE applied
    in-kernel. Used by the PE / SigLIP sparsesam_partial patches.

Both share a common FA2 scaffolding (`_FlashAttentionForwardAmpereBase`) and
differ only in their positional-encoding hooks. Imported lazily by their
callers — importing this package does not compile or instantiate the kernels.
"""
