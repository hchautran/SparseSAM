"""SpargeAttn drop-in for SigLIP.

Replaces SiglipAttention's SDPA with `spas_sage2_attn_meansim_topk_cuda`.
SigLIP has no RoPE so q/k go straight from projection into the kernel
without rotation. `start_block` controls which layers get patched
(everything below stays stock SDPA).

Note: SpargeAttn requires `head_dim ∈ {64, 128}`. siglip2-base/16 and
siglip2-large/16 have head_dim=64; so400m has head_dim=72 (unsupported).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from .._siglip import SiglipAttention, _find_siglip_vision_encoder


def _sparge_kernel():
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
    return spas_sage2_attn_meansim_topk_cuda


class SpargeSiglipAttention(SiglipAttention):
    """SiglipAttention with SDPA replaced by SpargeAttn."""

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        B, S, _ = hidden_states.shape
        H = self.num_heads

        q = rearrange(self.q_proj(hidden_states), "b s (h d) -> b h s d", h=H)
        k = rearrange(self.k_proj(hidden_states), "b s (h d) -> b h s d", h=H)
        v = rearrange(self.v_proj(hidden_states), "b s (h d) -> b h s d", h=H)

        topk = float(getattr(self, "_sparge_topk", 1.0))
        in_dtype = q.dtype
        if in_dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        attn = _sparge_kernel()(q, k, v, topk=topk, is_causal=False)
        if attn.dtype != in_dtype:
            attn = attn.to(in_dtype)

        attn = rearrange(attn, "b h s d -> b s (h d)").contiguous()
        return self.out_proj(attn), None


def apply_siglip_sparge_patch(model: nn.Module, ratio: float = 1.0,
                              start_block: int = 0,
                              verbose: bool = True) -> int:
    """Patch SiglipAttention modules from `start_block` onwards with
    `SpargeSiglipAttention`. `ratio` is forwarded as `topk` (1.0 = dense).
    """
    topk = float(ratio if ratio is not None else 1.0)
    if not (0.0 < topk <= 1.0):
        raise ValueError(f"sparge ratio must be in (0, 1]; got {ratio!r}")
    _sparge_kernel()  # eager import → fail fast if not installed

    encoder = _find_siglip_vision_encoder(model)
    if encoder is None:
        raise RuntimeError("Could not locate SiglipEncoder in `model`.")

    head_dim = encoder.layers[0].self_attn.head_dim
    if head_dim not in (64, 128):
        raise RuntimeError(
            f"SpargeAttn requires head_dim ∈ {{64, 128}}; SigLIP head_dim={head_dim}."
        )

    n_blocks = len(encoder.layers)
    sb = max(0, min(int(start_block), n_blocks))

    n = 0
    for idx in range(sb, n_blocks):
        attn = encoder.layers[idx].self_attn
        if not isinstance(attn, SpargeSiglipAttention):
            attn.__class__ = SpargeSiglipAttention
        attn._sparge_topk = topk
        n += 1

    if verbose:
        print(f"[siglip-sparge] L={n_blocks}  start_block={sb}  "
              f"patched_attn={n}  topk={topk}")
    return n


def remove_siglip_sparge_patch(model: nn.Module) -> int:
    from ..registry import remove_all_siglip
    return remove_all_siglip(model)


__all__ = [
    "SpargeSiglipAttention",
    "apply_siglip_sparge_patch",
    "remove_siglip_sparge_patch",
]
