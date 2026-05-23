"""PE patch base classes — Attention + Block subclasses for stage-compress
patches (dense and block-sparse variants) plus the standalone flash-rope-only
attention swap.

State is read from `self._tome_info`, set by `apply_stage_compress*` in
`install.py`. Same convention as SAM patches.
"""

from __future__ import annotations
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn


# Put perception_models on sys.path so `core.vision_encoder.pe` is importable
# without pip-installing it.
_here = os.path.dirname(__file__)
_pe_root = os.path.normpath(os.path.join(_here, "..", "3rd_party", "perception_models"))
if os.path.isdir(_pe_root) and _pe_root not in sys.path:
    sys.path.insert(0, _pe_root)

from core.vision_encoder.pe import (
    SelfAttention,
    ResidualAttentionBlock,
    VisionTransformer,
    Transformer,
)

from . import cute_kernel as _ck


@torch.no_grad()
def _build_stage_cache(self_attn: nn.Module, S: int, dtype: torch.dtype,
                       active_idx: Optional[torch.Tensor]) -> dict:
    """Pre-slice cos/sin to the active subset for dense cute-kernel reuse."""
    kernel, _m, _n = _ck._get_kernel(dtype, self_attn.head_dim)
    if kernel is None:
        return {}
    cos_full, sin_full = _ck._module_cached_cos_sin(self_attn, dtype)
    if active_idx is not None:
        cos = cos_full.index_select(0, active_idx).contiguous()
        sin = sin_full.index_select(0, active_idx).contiguous()
    else:
        cos = cos_full[:S].contiguous()
        sin = sin_full[:S].contiguous()
    return {"cos": cos, "sin": sin, "block_mask": None}


@torch.no_grad()
def _build_stage_cache_sparse(self_attn: nn.Module, S: int, dtype: torch.dtype,
                              active_idx: Optional[torch.Tensor],
                              sr: float, group_size: int, has_cls: bool,
                              device) -> dict:
    """Pre-slice cos/sin and build the perm/inv_perm tensors. Block mask is
    built lazily per-batch."""
    kernel, _m, n_blk = _ck._get_kernel(dtype, self_attn.head_dim)
    if kernel is None:
        return {}
    cos_full, sin_full = _ck._module_cached_cos_sin(self_attn, dtype)
    if active_idx is not None:
        cos = cos_full.index_select(0, active_idx)
        sin = sin_full.index_select(0, active_idx)
    else:
        cos = cos_full[:S]
        sin = sin_full[:S]
    perm, inv_perm = _ck._get_uniform_stride_perm(S, sr, group_size, n_blk, has_cls, device)
    cos = cos.index_select(0, perm).contiguous()
    sin = sin.index_select(0, perm).contiguous()
    return {
        "cos": cos, "sin": sin,
        "perm": perm, "inv_perm": inv_perm,
        "block_mask": None,
    }


def _sdpa_with_active_rope(self_attn, x, attn_mask, active_idx):
    """SDPA fallback: slice rope.freq to active_idx, run stock forward, restore."""
    orig_freq = self_attn.rope.freq
    self_attn.rope.freq = orig_freq.index_select(1, active_idx)
    try:
        return SelfAttention.forward(self_attn, x, attn_mask=attn_mask)
    finally:
        self_attn.rope.freq = orig_freq


_FALLBACK_WARNED: dict = {}


class FlashRopePEAttention(SelfAttention):
    """SelfAttention that respects `info['active_idx']` (so RoPE follows
    surviving tokens after compression) and optionally routes through the
    fused FA2+RoPE cute kernel when `info['use_flash_rope']`."""

    def forward(self, x, attn_mask=None):
        info       = self._tome_info
        active_idx = info.get("active_idx", None)
        use_flash  = info.get("use_flash_rope", False)

        if use_flash:
            cache = info.get("_stage_cache")
            cache_key = (id(active_idx), x.shape[1], x.dtype)
            if cache is None or cache.get("_key") != cache_key:
                cache = _build_stage_cache(self, x.shape[1], x.dtype, active_idx)
                cache["_key"] = cache_key
                info["_stage_cache"] = cache

            out = _ck.flash_rope_attn(
                self, x,
                cos=cache.get("cos"), sin=cache.get("sin"),
                block_mask=cache.get("block_mask"),
            )
            if out is not None:
                return out
            # Kernel couldn't be built — fall through to stock SDPA.

        if active_idx is None:
            return super().forward(x, attn_mask=attn_mask)
        return _sdpa_with_active_rope(self, x, attn_mask, active_idx)


class FlashRopeOnlyPEAttention(SelfAttention):
    """Pure cute-kernel attention swap; no compression awareness."""

    def forward(self, x, attn_mask=None):
        del attn_mask
        out = _ck.flash_rope_attn(self, x)
        if out is None:
            return super().forward(x)
        return out


class StageCompressPEBlock(ResidualAttentionBlock):
    """Base for stage-end blocks. Runs the original block forward, then calls
    `self.compress(x, active_idx, info) -> (x, new_active_idx)`."""

    def compress(self, x, active_idx, info):
        raise NotImplementedError

    def forward(self, x, attn_mask=None):
        x = super().forward(x, attn_mask=attn_mask)
        info: dict = self._tome_info
        if info.get("ratio", 1.0) >= 1.0:
            return x
        x, new_active = self.compress(x, info.get("active_idx", None), info)
        info["active_idx"] = new_active
        info["_stage_cache"] = None   # invalidate cos/sin cache
        return x


class SparseRopePEAttention(SelfAttention):
    """Sparse FA2+RoPE attention.

    Pre-compress (active_idx is None): stock SDPA, no perm, no kernel.
    Post-compress: block-sparse cute kernel; `x` is expected to be in
    permuted layout (the compress block did it once); `assume_permuted=True`
    skips per-call permutation. Falls back to SDPA if kernel can't build."""

    def forward(self, x, attn_mask=None):
        info       = self._tome_info
        active_idx = info.get("active_idx", None)

        if active_idx is None:
            return super().forward(x, attn_mask=attn_mask)

        kernel, _m, _n = _ck._get_kernel(x.dtype, self.head_dim)
        if kernel is None:
            key = (str(x.dtype), int(self.head_dim))
            if key not in _FALLBACK_WARNED:
                _FALLBACK_WARNED[key] = True
                print(f"[pe-sparse] cute kernel unavailable for dtype={x.dtype} "
                      f"head_dim={self.head_dim} — falling back to stock SDPA. "
                      f"To force a tile size, edit "
                      f"algos/pe_base/cute_kernel.py::_BLOCK_CANDIDATES.")
            return _sdpa_with_active_rope(self, x, attn_mask, active_idx)

        sr         = info.get("sparse_ratio", info.get("ratio", 1.0))
        group_size = info.get("group_size", 4)
        has_cls    = info.get("use_cls_token", False)

        cache = info.get("_stage_cache")
        cache_key = (id(active_idx), x.shape[1], x.dtype, sr)
        if cache is None or cache.get("_key") != cache_key:
            cache = _build_stage_cache_sparse(
                self, x.shape[1], x.dtype, active_idx, sr,
                group_size, has_cls, x.device,
            )
            cache["_key"] = cache_key
            info["_stage_cache"] = cache

        _ck._ensure_block_mask(cache, self, x, sr)

        permuted = bool(info.get("x_is_permuted"))
        out = _ck.flash_rope_sparse_attn(
            self, x,
            cos=cache.get("cos"), sin=cache.get("sin"),
            block_mask=cache.get("block_mask"),
            perm=cache.get("perm"), inv_perm=cache.get("inv_perm"),
            assume_permuted=permuted,
        )
        if out is not None:
            return out

        if permuted and cache.get("inv_perm") is not None:
            x = x.index_select(1, cache["inv_perm"])
        out = _sdpa_with_active_rope(self, x, attn_mask, active_idx)
        if permuted and cache.get("perm") is not None:
            out = out.index_select(1, cache["perm"])
        return out


class StageCompressSparsePEBlock(ResidualAttentionBlock):
    """Stage-end block for sparse-attn compression. Forward:
      1. Run original block (attn + MLP).
      2. Un-permute if currently in permuted layout.
      3. `self.compress` on natural-order x.
      4. Build next stage's cache (sliced + permuted cos/sin).
      5. Permute the compressed output once; `x_is_permuted=True` so
         downstream blocks run cute sparse with `assume_permuted=True`."""

    def compress(self, x, active_idx, info):
        raise NotImplementedError

    def forward(self, x, attn_mask=None):
        x = super().forward(x, attn_mask=attn_mask)
        info: dict = self._tome_info

        if info.get("x_is_permuted"):
            cache = info.get("_stage_cache")
            inv_perm = cache.get("inv_perm") if cache else None
            if inv_perm is not None:
                x = x.index_select(1, inv_perm)
            info["x_is_permuted"] = False

        if info.get("ratio", 1.0) >= 1.0:
            return x

        x, new_active = self.compress(x, info.get("active_idx", None), info)
        info["active_idx"] = new_active

        sr = info.get("sparse_ratio", info.get("ratio", 1.0))
        cache = _build_stage_cache_sparse(
            self.attn, x.shape[1], x.dtype, new_active, sr,
            info.get("group_size", 4),
            info.get("use_cls_token", False),
            x.device,
        )
        cache["_key"] = (id(new_active), x.shape[1], x.dtype, sr)
        info["_stage_cache"] = cache

        perm = cache.get("perm")
        if perm is not None:
            x = x.index_select(1, perm)
            info["x_is_permuted"] = True
        return x
