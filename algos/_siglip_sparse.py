"""Cute block-sparse attention for SigLIP (sibling of `_pe_stage_sparse.py`).

SigLIP has no RoPE — position info is added once at the input as a
learned embedding. The cute fused FA2+RoPE kernel always rotates q/k
with `q*cos + rotate_half(q)*sin`. To make the RoPE step a no-op we
pass identity tables (`cos=1, sin=0`) for every position. After
permutation those identity tables are still all 1s and 0s respectively,
so the kernel runs as plain block-sparse FA2.

The Q/K/V projection uses SigLIP's separate `q_proj / k_proj / v_proj`
(vs PE's fused `in_proj_weight`); output projection is `out_proj`.

`n_prefix=0` for SigLIP (no CLS, no register tokens) — every token is
a patch.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _pe_stage as _ps
from . import _pe_stage_sparse as _pss
from ._pe_stage_sparse import _ensure_block_mask
from ._siglip import SiglipAttention


@torch.no_grad()
def _identity_cos_sin(S: int, head_dim: int,
                      dtype: torch.dtype, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) tables of shape (S, head_dim) that make the
    cute kernel's RoPE step a no-op. `cos=1, sin=0` everywhere."""
    cos = torch.ones(S, head_dim, dtype=dtype, device=device).contiguous()
    sin = torch.zeros(S, head_dim, dtype=dtype, device=device).contiguous()
    return cos, sin


@torch.no_grad()
def build_siglip_sparse_cache(self_attn: SiglipAttention,
                              S: int, dtype: torch.dtype,
                              ratio: float, group_size: int, device) -> dict:
    """Pre-build identity cos/sin (permuted to kernel layout), perm/inv_perm,
    and the MLP keep/merge counts for one encoder forward.

    Returns `{}` if the cute kernel can't be built for this (dtype, head_dim)
    — caller falls through to baseline."""
    kernel, _m_blk, n_blk = _ps._get_kernel(dtype, self_attn.head_dim)
    if kernel is None:
        return {}

    perm, inv_perm = _pss._get_uniform_stride_perm(
        S, ratio, group_size, n_blk, has_cls=False, device=device,
    )
    # Identity cos/sin: invariant under permutation, but materialize the
    # full-S tables anyway so the kernel sees the right shape.
    cos, sin = _identity_cos_sin(S, self_attn.head_dim, dtype, device)
    cos = cos.index_select(0, perm).contiguous()
    sin = sin.index_select(0, perm).contiguous()

    # MLP keep/merge counts (mirrors PE's `_build_partial_cache`; n_prefix=0).
    if S > 0 and group_size > 0 and S % group_size == 0:
        n_groups = S // group_size
        K = max(1, round(ratio * S))
        if K >= n_groups:
            n_keep = max(0, (K - n_groups) // (group_size - 1))
            n_keep = min(n_keep, n_groups)
        else:
            n_keep = 0
        n_merge       = n_groups - n_keep
        cls_part_size = n_keep * group_size       # no prefix → keep section starts at 0
    else:
        n_merge       = 0
        cls_part_size = S

    return {
        "cos": cos, "sin": sin,
        "perm": perm, "inv_perm": inv_perm,
        "block_mask": None,
        "cls_part_size": cls_part_size,
        "n_merge":       n_merge,
        "gs":            group_size,
    }


def flash_block_sparse_attn_siglip(
    self_attn: SiglipAttention, x: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    block_mask: torch.Tensor,
    perm: Optional[torch.Tensor] = None,
    inv_perm: Optional[torch.Tensor] = None,
    assume_permuted: bool = False,
) -> Optional[torch.Tensor]:
    """SigLIP variant of the PE flash_rope_sparse_attn. Same kernel,
    same banded-diagonal + keep-bar mask. Q/K/V come from separate
    Linear layers; cos/sin are identity tables so RoPE is a no-op.

    Returns `None` if the cute kernel can't be built — caller should
    fall back to stock SDPA."""
    if x.dtype not in (torch.float16, torch.bfloat16):
        x = x.to(torch.float16)

    head_dim = self_attn.head_dim
    H        = self_attn.num_heads

    kernel, m_blk, n_blk = _ps._get_kernel(x.dtype, head_dim)
    if kernel is None or block_mask is None:
        return None

    B, S, _ = x.shape
    if assume_permuted or perm is None:
        x_in = x
    else:
        x_in = x.index_select(1, perm)

    q = self_attn.q_proj(x_in).view(B, S, H, head_dim)
    k = self_attn.k_proj(x_in).view(B, S, H, head_dim)
    v = self_attn.v_proj(x_in).view(B, S, H, head_dim)
    o = torch.empty_like(q)

    dtype_width = 16 if x.dtype in (torch.float16, torch.bfloat16) else 32
    def _cute_qkvo(t):
        return (_ps.from_dlpack(t, assumed_align=16)
                .mark_layout_dynamic(leading_dim=3)
                .mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(),
                                            divisibility=128 // dtype_width))
    q_c, k_c, v_c, o_c = _cute_qkvo(q), _cute_qkvo(k), _cute_qkvo(v), _cute_qkvo(o)
    cos_c = _ps.from_dlpack(cos, assumed_align=16)
    sin_c = _ps.from_dlpack(sin, assumed_align=16)
    mask_c = _ps.from_dlpack(block_mask, assumed_align=4)

    cu_stream = _ps.cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    scale = float(self_attn.scale)

    compiled = _ps._get_compiled(
        kernel, q_c, k_c, v_c, o_c, cos_c, sin_c, mask_c, scale, cu_stream,
        x.dtype, head_dim, B, S, H, m_blk, n_blk,
    )
    compiled(q_c, k_c, v_c, o_c, cos_c, sin_c, mask_c, scale, cu_stream)

    attn = o.view(B, S, H * head_dim)
    if not assume_permuted and inv_perm is not None:
        attn = attn.index_select(1, inv_perm)
    return self_attn.out_proj(attn)


__all__ = [
    "_identity_cos_sin",
    "build_siglip_sparse_cache",
    "flash_block_sparse_attn_siglip",
]
