"""GradToMe partial for SigLIP — port of `gradtome/pe_partial.py`.

Spatial-gradient-aware bipartite matching on the (H, W) patch grid →
merged-K/V SDPA → optional merge → MLP → unmerge. Falls back to chained
plain bipartite when ratio < 0.5 or grid isn't square.

SigLIP-specific: separate q/k/v/out_proj, no RoPE, n_prefix=0,
plain pre-norm layer (no LayerScale, no DropPath).
"""

from __future__ import annotations
import math
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .._siglip import (
    SiglipAttention, SiglipEncoderLayer,
    _find_siglip_vision_encoder,
)
from .merge import grad_bipartite_soft_matching, do_nothing
from ..tome.merge import bipartite_soft_matching


def _grad_match_single(metric: torch.Tensor, ratio: float, sx: int, sy: int):
    N = metric.shape[1]
    r_grid = int(math.isqrt(N))
    if r_grid * r_grid != N:
        return None
    r = math.floor(N - N * ratio)
    return grad_bipartite_soft_matching(
        metric=metric, H=r_grid, W=r_grid, sx=sx, sy=sy, r=r,
    )


def _chained_bipartite_match(metric: torch.Tensor, ratio: float):
    if ratio >= 0.5 - 1e-6:
        return bipartite_soft_matching(metric, ratio=ratio, class_token=False)
    n_passes = max(1, math.ceil(math.log(max(ratio, 1e-6), 0.5)))
    per_pass_ratio = max(ratio ** (1.0 / n_passes), 0.5 + 1e-6)
    merges, unmerges = [], []
    cur, achieved = metric, 1.0
    for i in range(n_passes):
        r_pass = (max(min(ratio / achieved, 1.0), 0.5 + 1e-6)
                  if i == n_passes - 1 else per_pass_ratio)
        m, u = bipartite_soft_matching(cur, ratio=r_pass, class_token=False)
        if m is do_nothing:
            break
        merges.append(m); unmerges.append(u)
        achieved *= r_pass
        cur, _ = m(cur, mode="mean")
    if not merges:
        return do_nothing, do_nothing

    def merge(x, mode="mean"):
        for m in merges:
            x, _ = m(x, mode=mode)
        return x, None

    def unmerge(x):
        for u in reversed(unmerges):
            x = u(x)
        return x

    return merge, unmerge


def _build_grad_match(metric: torch.Tensor, ratio: float, sx: int, sy: int):
    """Build (merge, unmerge) over the patch grid; falls back to chained
    bipartite when grid isn't square or ratio < 0.5."""
    N = metric.shape[1]
    r_grid = int(math.isqrt(N))
    if r_grid * r_grid != N or ratio < 0.5 - 1e-6:
        return _chained_bipartite_match(metric, ratio)
    sp = _grad_match_single(metric, ratio, sx, sy)
    if sp is None:
        return _chained_bipartite_match(metric, ratio)
    return sp


class GradTomeSiglipAttention(SiglipAttention):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        info  = self._tome_info
        ratio = info.get("ratio", 1.0)
        sx    = info.get("grad_sx", 2)
        sy    = info.get("grad_sy", 2)

        if ratio >= 1.0:
            info["_block_merge_fn"]   = do_nothing
            info["_block_unmerge_fn"] = do_nothing
            return super().forward(hidden_states, attention_mask=attention_mask, **kwargs)

        B, S, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, S, H, D).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, H, D).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, H, D).transpose(1, 2)

        k_flat = rearrange(k, "b h s d -> b s (h d)")
        v_flat = rearrange(v, "b h s d -> b s (h d)")

        metric = hidden_states.mean(0, keepdim=True)
        merge_fn, unmerge_fn = _build_grad_match(metric, ratio, sx, sy)
        info["_block_merge_fn"]   = merge_fn
        info["_block_unmerge_fn"] = unmerge_fn

        if merge_fn is do_nothing:
            return super().forward(hidden_states, attention_mask=attention_mask, **kwargs)

        k_merged_flat, _ = merge_fn(k_flat, mode="mean")
        v_merged_flat, _ = merge_fn(v_flat, mode="mean")
        k_merged = rearrange(k_merged_flat, "b s (h d) -> b h s d", h=H)
        v_merged = rearrange(v_merged_flat, "b s (h d) -> b h s d", h=H)

        attn = F.scaled_dot_product_attention(
            q, k_merged, v_merged, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=self.scale,
        )
        attn = rearrange(attn, "b h s d -> b s (h d)").contiguous()
        return self.out_proj(attn), None


class GradTomeSiglipLayer(SiglipEncoderLayer):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        info: dict = self._tome_info
        info["_block_merge_fn"] = info["_block_unmerge_fn"] = None

        residual = hidden_states
        h = self.layer_norm1(hidden_states)
        h, _ = self.self_attn(h, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + h

        merge_fn   = info.get("_block_merge_fn")
        unmerge_fn = info.get("_block_unmerge_fn")
        residual = hidden_states
        if (info.get("mlp_merge", True)
                and merge_fn is not None and merge_fn is not do_nothing):
            x_merged, _ = merge_fn(hidden_states, mode="mean")
            x_merged = self.mlp(self.layer_norm2(x_merged))
            hidden_states = unmerge_fn(x_merged) + residual
        else:
            hidden_states = residual + self.mlp(self.layer_norm2(hidden_states))

        info["_block_merge_fn"] = info["_block_unmerge_fn"] = None
        return hidden_states


def apply_siglip_gradtome_partial_patch(model: nn.Module,
                                        ratio: float = 0.7,
                                        grad_sx: int = 2,
                                        grad_sy: int = 2,
                                        start_block: int = 0,
                                        mlp_merge: bool = True,
                                        verbose: bool = True) -> int:
    encoder = _find_siglip_vision_encoder(model)
    if encoder is None:
        raise RuntimeError("Could not locate SiglipEncoder in `model`.")

    n_blocks = len(encoder.layers)
    sb = max(0, min(int(start_block), n_blocks))

    info = {
        "ratio":     float(ratio),
        "grad_sx":   int(grad_sx),
        "grad_sy":   int(grad_sy),
        "n_prefix":  0,
        "mlp_merge": bool(mlp_merge),
    }
    model._tome_info = info

    n_attn = 0
    for idx in range(sb, n_blocks):
        layer = encoder.layers[idx]
        attn  = layer.self_attn
        if not isinstance(attn, GradTomeSiglipAttention):
            attn.__class__ = GradTomeSiglipAttention
        attn._tome_info = info
        n_attn += 1
        if not isinstance(layer, GradTomeSiglipLayer):
            layer.__class__ = GradTomeSiglipLayer
        layer._tome_info = info

    if verbose:
        print(f"[siglip-gradtome-partial] L={n_blocks}  start_block={sb}  "
              f"patched_layers={n_blocks - sb}  patched_attn={n_attn}  "
              f"ratio={info['ratio']}  sx={grad_sx} sy={grad_sy}  "
              f"mlp_merge={info['mlp_merge']}")
    return n_blocks - sb


def remove_siglip_gradtome_partial_patch(model: nn.Module) -> int:
    from ..registry import remove_all_siglip
    return remove_all_siglip(model)


def get_classes() -> Tuple[type, type]:
    return GradTomeSiglipLayer, GradTomeSiglipAttention


__all__ = [
    "apply_siglip_gradtome_partial_patch",
    "remove_siglip_gradtome_partial_patch",
    "get_classes",
]
