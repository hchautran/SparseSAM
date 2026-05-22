"""Sparsesam patch for SAM3's ViT trunk (HuggingFace
`transformers.models.sam3.modeling_sam3.Sam3ViTModel`).

Implements two compression paths:

  1. **Partial-MLP** — same group-rank scheme as SAM-HQ
     ([../sam.py:393-410]). At local layers AFTER the first global
     populates the cache, MLP runs on top-`round(r·N)` tokens by
     avg-rank only; dropped tokens skip MLP (residual passthrough).

  2. **Sparse attention** — same keep-bar + diagonal-band mask as
     SAM-HQ's cute kernel, ported to RoPE-aware SDPA. Per-layer:
       * Compute per-head perm via `tile_stride_matching` on pre-RoPE k.
       * Average across heads → single shared perm per batch.
       * Permute q, k, v, cos, sin by that perm.
       * Apply RoPE on permuted q, k (post-permute the RoPE tables).
       * SDPA with a block-granularity sparse mask
         (everyone attends to top-keep_n keys + diagonal).
       * Unpermute the output.
     RoPE is applied on permuted q/k with permuted cos/sin tables, so
     RoPE positions map correctly through the permutation.

SAM3's vision backbone is a standard ViT with the same windowed/global
pattern as SAM-HQ ViT — just bigger:
  * 32 layers (vs SAM-HQ ViT-L's 24).
  * Globals at [7, 15, 23, 31] (every 8th layer).
  * 72×72 = 5184 token grid (vs 64×64 = 4096).
  * Single resolution (no q-pool, unlike SAM2 Hiera).
  * Separate `q_proj`/`k_proj`/`v_proj` (vs SAM-HQ's combined `qkv`).
  * RoPE inside attention (vs SAM-HQ's additive rel-pos bias).

Usage:
    from algos.sparsesam.patch.sam3_vit import apply_patch
    apply_patch(model.detector_model.vision_encoder.backbone, ratio=0.5)
"""

from __future__ import annotations

import os
import sys
import types
from typing import List, Optional

import torch
import torch.nn.functional as F


# ── flex_attention (PyTorch ≥ 2.5) — pure-Python sparse attention API.
# Mirrors the role of the cute kernel in PE's pe_partial.py but without
# DLPack wrapping. flex_attention MUST run inside torch.compile to hit
# its fused kernel; in eager mode it falls back to slow Python.
_FLEX_AVAILABLE = False
_FLEX_ERROR: Optional[str] = None
_FLEX_COMPILED = None
try:
    from torch.nn.attention.flex_attention import (  # type: ignore[import]
        flex_attention as _flex_attention_eager, create_block_mask,
    )
    # Compile once at import. dynamic=True so torch.compile reuses the
    # same compiled kernel across batch sizes / Sq variants without
    # recompiling. Dropping max-autotune for faster compile (still
    # uses the fused flex kernel).
    _FLEX_COMPILED = torch.compile(_flex_attention_eager, dynamic=True)
    _FLEX_AVAILABLE = True
except Exception as _e:
    _FLEX_ERROR = repr(_e)
    _flex_attention_eager = None
    create_block_mask = None  # type: ignore[assignment]


# Default block sizes for the flex_attention block mask. flex_attention
# operates on 128×128 blocks by default (BLOCK_M, BLOCK_N).
_M_BLOCK  = 64
_N_BLOCK  = 64


# ─────────────────────────────────────────────────────────────────────────
# Sparse-attention helpers (flex_attention + RoPE-aware static perm)
# ─────────────────────────────────────────────────────────────────────────

def _build_keep_bar_block_mask(B: int, H: int, Sq: int,
                               keep_n_tokens: int,
                               diag_block_size: int,
                               device) -> "torch.nn.attention.flex_attention.BlockMask":
    """flex_attention BlockMask in **permuted** space:
    every query attends to top-`keep_n_tokens` keys (keep-bar) +
    its own diagonal block of width `diag_block_size`.

    In permuted space (kept tokens are at indices [0, keep_n)), this
    is equivalent to PE's `_make_A_mask`: keep-bar prefix + diagonal
    band. Same pattern for every (b, h) so we build a single block-mask
    and let flex_attention broadcast.
    """
    keep_n = int(keep_n_tokens)
    bs = int(diag_block_size)

    def keep_bar_diag(b, h, q_idx, kv_idx):
        in_keep_bar = kv_idx < keep_n
        on_diag     = (q_idx // bs) == (kv_idx // bs)
        return in_keep_bar | on_diag

    # B=H=None → mask is broadcast across batch + heads (cheaper).
    return create_block_mask(
        keep_bar_diag, B=None, H=None,
        Q_LEN=Sq, KV_LEN=Sq, device=device,
    )


def _maybe_extract_rope(position_embeddings, Sq: int, head_dim: int):
    """Reshape SAM3's cos/sin to (Sq, D) fp32.

    `Sam3ViTRotaryEmbedding.forward` returns shape (1, 1, Sq, D); we
    just reshape to (Sq, D) for `apply_rotary_pos_emb_2d` after we've
    pre-permuted by `index_select`.
    """
    cos, sin = position_embeddings
    if cos.dim() == 4:
        cos = cos.reshape(-1, head_dim)
        sin = sin.reshape(-1, head_dim)
    elif cos.dim() == 3:
        cos = cos.reshape(-1, head_dim)
        sin = sin.reshape(-1, head_dim)
    if cos.shape[0] != Sq:
        raise RuntimeError(
            f"cos/sin shape ({cos.shape[0]},) doesn't match Sq={Sq}."
        )
    return cos.float().contiguous(), sin.float().contiguous()


# ─────────────────────────────────────────────────────────────────────────
# Patched forwards
# ─────────────────────────────────────────────────────────────────────────

def _patched_layer_forward(self, hidden_states, **kwargs):
    """`Sam3ViTLayer.forward` with the grouped-perm partial-MLP."""
    from transformers.models.sam3.modeling_sam3 import (
        window_partition, window_unpartition,
    )

    info  = self._tome_info
    ratio = info.get("ratio", 1.0)
    gs    = info.get("group_size", 4)

    residual      = hidden_states                            # B, H, W, C
    hidden_states = self.layer_norm1(hidden_states)
    H_full, W_full = hidden_states.shape[1], hidden_states.shape[2]

    # ── Cache the perm at GLOBAL blocks only (mirrors SAM-HQ
    # [../sam.py:322-327]: globals see the full grid, so their k is
    # naturally at full resolution; locals see windowed input and
    # would key the cache on the wrong dims). Local blocks BEFORE the
    # first global thus run full MLP — same behaviour as SAM-HQ.
    #
    # SAM3 globals: layers [7, 15, 23, 31]. So layers 0–6 run full
    # MLP; layers 8–14, 16–22, 24–30 (21 of 32) run partial-MLP.
    if (info.get("mlp_merge", False) and ratio < 1.0
            and self.window_size == 0
            and (H_full * W_full) % gs == 0
            and (H_full, W_full, ratio) not in info.get("perm_cache", {})):
        N_in = H_full * W_full
        attn = self.attention
        nh   = attn.num_attention_heads
        with torch.no_grad():
            k_full = attn.k_proj(hidden_states)              # (B, H, W, C)
            k_full = k_full.reshape(hidden_states.shape[0], N_in, nh, -1)
            k_flat = (k_full.permute(0, 2, 1, 3)
                            .reshape(hidden_states.shape[0] * nh, N_in, -1)
                            .float())
            from ..sam import tile_stride_matching as _tsm
            _p, _i, inv_group = _tsm(
                k_flat, H_full, W_full, ratio=ratio, group_size=gs,
            )
            del k_full, k_flat
        info.setdefault("perm_cache", {})[(H_full, W_full, ratio)] = inv_group

    # ── stock attention path (mirrors Sam3ViTLayer.forward) ──
    pad_hw = None
    if self.window_size > 0:
        height, width = hidden_states.shape[1], hidden_states.shape[2]
        hidden_states, pad_hw = window_partition(hidden_states, self.window_size)

    position_embeddings = self.rotary_emb()
    hidden_states, _    = self.attention(
        hidden_states, position_embeddings, **kwargs,
    )

    if self.window_size > 0:
        hidden_states = window_unpartition(
            hidden_states, self.window_size, pad_hw, (height, width),
        )

    hidden_states = residual + hidden_states                 # post-attn residual
    post_attn     = hidden_states
    normed        = self.layer_norm2(hidden_states)

    # ── partial-MLP path ──
    mlp_merge  = info.get("mlp_merge", True)
    H_post, W_post, C = post_attn.shape[1], post_attn.shape[2], post_attn.shape[3]
    N = H_post * W_post

    cached: Optional[torch.Tensor] = None
    if (mlp_merge and ratio < 1.0 and self.window_size > 0 and N % gs == 0):
        cached = info.get("perm_cache", {}).get((H_post, W_post, ratio))

    if cached is not None:
        B = post_attn.shape[0]
        nh = self.attention.num_attention_heads
        normed_seq    = normed.reshape(B, N, C)
        post_attn_seq = post_attn.reshape(B, N, C)
        keep_n = max(1, round(ratio * N))
        avg_rank = cached.view(B, nh, -1).float().mean(dim=1)
        top_idx  = avg_rank.topk(keep_n, dim=1, largest=False).indices
        idx_e    = top_idx.unsqueeze(-1).expand(-1, -1, C)
        x_norm_kept  = normed_seq.gather(1, idx_e)
        mlp_out_kept = self.dropout(self.mlp(x_norm_kept))
        # Residual-add the kept MLP outputs at the kept positions.
        out_seq = post_attn_seq.scatter_add(1, idx_e, mlp_out_kept)
        return out_seq.reshape(B, H_post, W_post, C)

    # Full-MLP fallback (globals, mlp_merge=False, or pre-cache blocks).
    return post_attn + self.dropout(self.mlp(normed))


def _patched_attention_forward(self, hidden_states, position_embeddings, **kwargs):
    """`Sam3ViTRoPEAttention.forward` with `flex_attention`-based sparse
    attention. Mirrors PE's `pe_partial.py` pattern:

      1. Build a uniform-stride perm once per (Sq, ratio); pre-permute
         cos/sin tables.
      2. Build a static keep-bar + diagonal `BlockMask` in permuted
         space (every query attends to top-`round(r·Sq)` keys plus
         its own diagonal block).
      3. Per-call: permute x by perm → q/k/v projections → apply RoPE
         in PyTorch with permuted cos/sin → flex_attention(q, k, v,
         block_mask) → inverse permute → o_proj.

    Falls back to the stock attention when flex_attention unavailable,
    sparse_attn disabled, ratio=1.0, or Sq misaligned with m/n_block.
    """
    from transformers.models.sam3.modeling_sam3 import apply_rotary_pos_emb_2d

    info = getattr(self, "_tome_info", None)
    sparse_on = (
        _FLEX_AVAILABLE
        and info is not None
        and info.get("sparse_attn", False)
        and info.get("ratio", 1.0) < 1.0
    )

    B, height, width, _ = hidden_states.shape
    Sq = height * width
    nh = self.num_attention_heads
    d  = self.head_dim

    m_blk = info.get("m_block", _M_BLOCK) if info else _M_BLOCK
    n_blk = info.get("n_block", _N_BLOCK) if info else _N_BLOCK
    if not sparse_on or Sq % m_blk != 0 or Sq % n_blk != 0:
        return self._tome_orig_attention_forward(
            hidden_states, position_embeddings, **kwargs,
        )

    ratio = info["ratio"]
    gs    = info.get("group_size", 4)

    # ── Static cache (perm + permuted cos/sin + block mask) per
    # (Sq, ratio, n_blk). Mirrors PE's _build_partial_cache.
    C = hidden_states.shape[3]
    cache_key  = (Sq, ratio, n_blk)
    attn_cache = info.setdefault("attn_keep_cache", {})
    cached = attn_cache.get(cache_key)
    if cached is None:
        from ..._pe_stage_sparse import _get_uniform_stride_perm
        perm, inv_perm = _get_uniform_stride_perm(
            Sq, ratio, gs, n_blk, has_cls=False, device=hidden_states.device,
        )
        cos_t, sin_t = _maybe_extract_rope(position_embeddings, Sq, d)
        cos_p = cos_t.index_select(0, perm).contiguous()
        sin_p = sin_t.index_select(0, perm).contiguous()
        keep_n = max(1, round(ratio * Sq))
        block_mask = _build_keep_bar_block_mask(
            B=B, H=nh, Sq=Sq, keep_n_tokens=keep_n,
            diag_block_size=n_blk, device=hidden_states.device,
        )
        cached = {
            "perm":       perm,
            "inv_perm":   inv_perm,
            "cos_p":      cos_p,
            "sin_p":      sin_p,
            "block_mask": block_mask,
        }
        attn_cache[cache_key] = cached

    perm       = cached["perm"]
    inv_perm   = cached["inv_perm"]
    cos_p      = cached["cos_p"]
    sin_p      = cached["sin_p"]
    block_mask = cached["block_mask"]

    # ── permute hidden_states by perm.
    x_flat = hidden_states.reshape(B, Sq, C)
    x_perm = x_flat.index_select(1, perm)

    # ── q/k/v projections (in permuted space).
    new_shape = (B, Sq, nh, d)
    q = self.q_proj(x_perm).view(*new_shape).transpose(1, 2)         # (B, H, S, D)
    k = self.k_proj(x_perm).view(*new_shape).transpose(1, 2)
    v = self.v_proj(x_perm).view(*new_shape).transpose(1, 2)

    # ── RoPE with PRE-PERMUTED cos/sin (so each token receives its
    # original-position rotation regardless of where it ended up after
    # permutation). Reshape cos_p/sin_p to (1, 1, S, D) for broadcast.
    cos_b = cos_p.view(1, 1, Sq, d).to(q.dtype)
    sin_b = sin_p.view(1, 1, Sq, d).to(q.dtype)
    q, k = apply_rotary_pos_emb_2d(q, k, cos=cos_b, sin=sin_b)

    # ── flex_attention with the static keep-bar + diag block mask.
    # MUST run via torch.compile (in eager mode it's a slow fallback).
    out = _FLEX_COMPILED(q, k, v, block_mask=block_mask, scale=self.scaling)

    # ── (B, H, S, D) → (B, S, H*D) → inverse-permute → o_proj.
    attn = out.transpose(1, 2).contiguous().view(B, Sq, nh * d)
    attn = attn.index_select(1, inv_perm)
    return self.o_proj(attn.view(B, height, width, -1)), None


# ─────────────────────────────────────────────────────────────────────────
# Patch / unpatch
# ─────────────────────────────────────────────────────────────────────────

def _reset_perm_cache(info):
    def _hook(_module, _inputs):
        info["perm_cache"] = {}
    return _hook


def apply_patch(vit, ratio: float = 0.5, group_size: int = 4,
                prune_mlp: bool = True, sparse_attn: bool = False,
                m_block: int = _M_BLOCK, n_block: int = _N_BLOCK, **_):
    """Patch a SAM3 ViT for sparsesam partial-MLP and/or sparse attention.

    Args:
        vit:          `transformers.models.sam3.modeling_sam3.Sam3ViTModel`
                      instance — the backbone of the SAM3 vision encoder.
                      Locate it via e.g.
                      `model.detector_model.vision_encoder.backbone`.
        ratio:        keep ratio in (0, 1].
        group_size:   Z-curve group size (default 4 = 2×2 spatial quad).
        prune_mlp:    enable partial-MLP path (default True).
        sparse_attn:  enable `flex_attention`-based sparse attention.
                      Mirrors PE's `pe_partial.py` pattern: uniform-
                      stride perm + pre-permuted cos/sin + keep-bar +
                      diagonal `BlockMask`. Requires PyTorch 2.5+.
        m_block,
        n_block:      block sizes for the flex_attention BlockMask.
                      The diagonal-band's block width = n_block.
    """
    from transformers.models.sam3.modeling_sam3 import (
        Sam3ViTLayer, Sam3ViTModel, Sam3ViTRoPEAttention,
    )

    assert 0 < ratio <= 1.0, "ratio must be in (0, 1]"

    if not isinstance(vit, Sam3ViTModel):
        raise TypeError(
            f"apply_patch expects a Sam3ViTModel, got {type(vit).__name__}. "
            f"Pass `model.detector_model.vision_encoder.backbone` (or equivalent)."
        )

    if sparse_attn and not _FLEX_AVAILABLE:
        print(
            f"[sparsesam-sam3] sparse_attn requested but "
            f"torch.nn.attention.flex_attention unavailable: "
            f"{_FLEX_ERROR}. Falling back to dense attn (needs torch ≥ 2.5)."
        )
        sparse_attn = False

    info = {
        "ratio":            ratio,
        "group_size":       group_size,
        "mlp_merge":        bool(prune_mlp),
        "sparse_attn":      bool(sparse_attn),
        "m_block":          m_block,
        "n_block":          n_block,
        "perm_cache":       {},                         # used by partial-MLP path
        "attn_keep_cache":  {},                         # used by sparse-attn path
    }
    vit._tome_info = info

    vit._tome_hook_handles: List = []
    h = vit.register_forward_pre_hook(_reset_perm_cache(info))
    vit._tome_hook_handles.append(h)

    n_patched = 0
    for layer in vit.layers:
        if not isinstance(layer, Sam3ViTLayer):
            continue
        layer._tome_info = info
        if not getattr(layer, "_tome_orig_forward", None):
            layer._tome_orig_forward = layer.forward
        layer.forward = types.MethodType(_patched_layer_forward, layer)

        # Patch attention for the sparse-attn path. Even if sparse_attn
        # is False today, attaching the wrapper is cheap — the wrapper
        # short-circuits to the original when the flag is off.
        attn = layer.attention
        if isinstance(attn, Sam3ViTRoPEAttention):
            attn._tome_info = info
            if not getattr(attn, "_tome_orig_attention_forward", None):
                attn._tome_orig_attention_forward = attn.forward
            attn.forward = types.MethodType(_patched_attention_forward, attn)
        n_patched += 1

    n_globals = sum(1 for layer in vit.layers
                    if isinstance(layer, Sam3ViTLayer) and layer.window_size == 0)
    n_local   = n_patched - n_globals
    print(
        f"[sparsesam-sam3] patched  ratio={ratio}  gs={group_size}  "
        f"prune_mlp={prune_mlp}  sparse_attn={sparse_attn}  "
        f"layers={n_patched} (globals={n_globals}, locals={n_local})"
    )


def remove_patch(vit):
    """Restore original forwards. Inverse of `apply_patch`."""
    for h in getattr(vit, "_tome_hook_handles", []):
        h.remove()
    if hasattr(vit, "_tome_hook_handles"):
        del vit._tome_hook_handles
    if hasattr(vit, "_tome_info"):
        del vit._tome_info

    for layer in vit.layers:
        if hasattr(layer, "_tome_orig_forward"):
            layer.forward = layer._tome_orig_forward
            del layer._tome_orig_forward
        if hasattr(layer, "_tome_info"):
            del layer._tome_info
        attn = getattr(layer, "attention", None)
        if attn is not None:
            if hasattr(attn, "_tome_orig_attention_forward"):
                attn.forward = attn._tome_orig_attention_forward
                del attn._tome_orig_attention_forward
            if hasattr(attn, "_tome_info"):
                del attn._tome_info
