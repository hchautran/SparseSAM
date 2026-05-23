"""PE patch base classes + cute-kernel infrastructure (consolidated from the
previous `_pe_stage.py` / `_pe_stage_sparse.py`).

Layout:
  • `cute_kernel.py` — fused FA2+RoPE cute kernel helpers, caches, masks.
  • `classes.py`     — PE attention/block subclasses (dense + block-sparse).
  • `install.py`     — `apply_*` / `remove_*` entry points + Transformer finders.

Each PE algorithm imports the public symbols it needs from this package:

    from ..pe_base import (
        SelfAttention, ResidualAttentionBlock,
        StageCompressPEBlock, FlashRopePEAttention,
        apply_stage_compress, apply_pe_flash_rope_patch,
    )
"""

from .classes import (
    SelfAttention, ResidualAttentionBlock,
    VisionTransformer, Transformer,
    FlashRopePEAttention, FlashRopeOnlyPEAttention,
    StageCompressPEBlock, StageCompressSparsePEBlock,
    SparseRopePEAttention,
)
from .cute_kernel import (
    flash_rope_attn, flash_rope_sparse_attn,
    _ensure_cute_deps, _get_kernel, _module_cached_cos_sin,
    _make_A_mask, _get_uniform_stride_perm, _ensure_block_mask,
)
from .install import (
    apply_stage_compress, apply_stage_compress_sparse,
    apply_pe_flash_rope_patch,
    remove_stage_compress, remove_stage_compress_sparse, remove_pe_flash_rope_patch,
    _find_vision_transformer, _vit_uses_cls_token,
)

__all__ = [
    # Stock + subclass classes
    "SelfAttention", "ResidualAttentionBlock", "VisionTransformer", "Transformer",
    "FlashRopePEAttention", "FlashRopeOnlyPEAttention",
    "StageCompressPEBlock", "StageCompressSparsePEBlock", "SparseRopePEAttention",
    # Cute-kernel surface (consumed by algo patches that build their own caches)
    "flash_rope_attn", "flash_rope_sparse_attn",
    "_ensure_cute_deps", "_get_kernel", "_module_cached_cos_sin",
    "_make_A_mask", "_get_uniform_stride_perm", "_ensure_block_mask",
    # Install / introspection
    "apply_stage_compress", "apply_stage_compress_sparse",
    "apply_pe_flash_rope_patch",
    "remove_stage_compress", "remove_stage_compress_sparse", "remove_pe_flash_rope_patch",
    "_find_vision_transformer", "_vit_uses_cls_token",
]
