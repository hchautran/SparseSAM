"""SpargeAttn patch for the Perception Encoder (PE) `SelfAttention`.

PE's attention is a clean `F.scaled_dot_product_attention` call with RoPE
applied to q/k beforehand. This patch class-swaps every `SelfAttention`
to `SpargePEAttention`, which keeps the RoPE step intact and routes the
SDPA call through `spas_sage2_attn_meansim_topk_cuda` instead.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from core.vision_encoder.pe import SelfAttention


def _sparge_kernel():
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
    return spas_sage2_attn_meansim_topk_cuda


class SpargePEAttention(SelfAttention):
    """PE SelfAttention with SDPA replaced by SpargeAttn (RoPE preserved)."""

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        del attn_mask  # SpargeAttn has no attn_mask param; PE never passes one.

        B, S, E = x.shape
        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        proj = (proj.unflatten(-1, (3, E))
                    .unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous())
        q, k, v = proj[0], proj[1], proj[2]

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if self.rope is not None:
            q, k = self.rope(q, k)

        topk = float(getattr(self, "_sparge_topk", 1.0))

        in_dtype = q.dtype
        if in_dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        attn = _sparge_kernel()(q, k, v, topk=topk, is_causal=False)
        if attn.dtype != in_dtype:
            attn = attn.to(in_dtype)

        attn = rearrange(attn, "b h s d -> b s (h d)")
        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)


def apply_pe_sparge_patch(model: nn.Module, ratio: float = 1.0,
                          verbose: bool = True) -> int:
    """Replace every PE SelfAttention with `SpargePEAttention`. No token
    compression. `ratio` is forwarded to SpargeAttn's `topk`."""
    topk = float(ratio if ratio is not None else 1.0)
    if not (0.0 < topk <= 1.0):
        raise ValueError(f"sparge ratio must be in (0, 1]; got {ratio!r}")

    # Eager import so we fail fast.
    _sparge_kernel()

    n = 0
    for mod in model.modules():
        if isinstance(mod, SelfAttention):
            if not isinstance(mod, SpargePEAttention):
                mod.__class__ = SpargePEAttention
            mod._sparge_topk = topk
            n += 1

    if verbose:
        print(f"[pe-sparge] patched {n} SelfAttention module(s)  topk={topk}")
    return n


def remove_pe_sparge_patch(model: nn.Module) -> int:
    """Back-compat wrapper. The real cleanup happens in `remove_all_pe`,
    which walks subclasses of SelfAttention and reverts the class pointer."""
    from ..registry import remove_all_pe
    return remove_all_pe(model)


__all__ = [
    "SpargePEAttention",
    "apply_pe_sparge_patch",
    "remove_pe_sparge_patch",
]
