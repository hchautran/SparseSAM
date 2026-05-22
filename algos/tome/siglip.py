"""ToMe partial for SigLIP — port of `tome/pe_partial.py`.

Per layer: bipartite-match `x`, run SDPA(Q_full, K_merged, V_merged),
then optionally merge → MLP → unmerge. Token count preserved at the
layer boundary.

SigLIP-specific adaptations vs PE:
  * Separate `q_proj / k_proj / v_proj / out_proj` (vs PE's fused
    `in_proj_weight`).
  * No RoPE — drop the `self.rope(q, k)` step.
  * `n_prefix = 0` — no CLS token, no register tokens.
  * Layer attribute names: `layer_norm1 / self_attn / layer_norm2 /
    mlp` (no LayerScale, no DropPath in plain SigLIP blocks).
"""

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .._siglip import (
    SiglipAttention, SiglipEncoderLayer,
    _find_siglip_vision_encoder,
)
from .merge import bipartite_soft_matching, do_nothing


def _chained_bipartite_match(metric: torch.Tensor, ratio: float):
    """Bipartite that supports `ratio < 0.5` via chained passes."""
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


class TomeSiglipAttention(SiglipAttention):
    """Full Q + ToMe-merged K/V SDPA. Stashes (merge, unmerge) on
    `_tome_info` so the layer forward can reuse them on the MLP path."""

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        info  = self._tome_info
        ratio = info.get("ratio", 1.0)

        if ratio >= 1.0:
            info["_block_merge_fn"]   = do_nothing
            info["_block_unmerge_fn"] = do_nothing
            return super().forward(hidden_states, attention_mask=attention_mask, **kwargs)

        B, S, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, S, H, D).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, H, D).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, H, D).transpose(1, 2)
        # No RoPE for SigLIP — position info lives in the input embedding only.

        k_flat = rearrange(k, "b h s d -> b s (h d)")
        v_flat = rearrange(v, "b h s d -> b s (h d)")

        metric = hidden_states.mean(0, keepdim=True)
        merge_fn, unmerge_fn = _chained_bipartite_match(metric, ratio=ratio)
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


class TomeSiglipLayer(SiglipEncoderLayer):
    """Layer forward: attn (which stashes the match) + optional
    merge/MLP/unmerge — same shape as `TomePEPartialBlock`."""

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


def apply_siglip_tome_partial_patch(model: nn.Module,
                                    ratio: float = 0.7,
                                    start_block: int = 0,
                                    mlp_merge: bool = True,
                                    verbose: bool = True) -> int:
    encoder = _find_siglip_vision_encoder(model)
    if encoder is None:
        raise RuntimeError("Could not locate SiglipEncoder in `model`.")

    n_blocks = len(encoder.layers)
    sb = max(0, min(int(start_block), n_blocks))

    info = {"ratio": float(ratio), "n_prefix": 0, "mlp_merge": bool(mlp_merge)}
    model._tome_info = info

    n_attn = 0
    for idx in range(sb, n_blocks):
        layer = encoder.layers[idx]
        attn  = layer.self_attn
        if not isinstance(attn, TomeSiglipAttention):
            attn.__class__ = TomeSiglipAttention
        attn._tome_info = info
        n_attn += 1
        if not isinstance(layer, TomeSiglipLayer):
            layer.__class__ = TomeSiglipLayer
        layer._tome_info = info

    if verbose:
        print(f"[siglip-tome-partial] L={n_blocks}  start_block={sb}  "
              f"patched_layers={n_blocks - sb}  patched_attn={n_attn}  "
              f"ratio={info['ratio']}  mlp_merge={info['mlp_merge']}")
    return n_blocks - sb


def remove_siglip_tome_partial_patch(model: nn.Module) -> int:
    from ..registry import remove_all_siglip
    return remove_all_siglip(model)


def get_classes() -> Tuple[type, type]:
    return TomeSiglipLayer, TomeSiglipAttention


__all__ = [
    "apply_siglip_tome_partial_patch",
    "remove_siglip_tome_partial_patch",
    "get_classes",
]
