"""SparseSAM partial for SigLIP — port of `sparsesam/pe_partial.py`.

Per-layer flow (matches PE-side):
  1. First layer permutes `x` once (uniform-stride z-order keep groups
     followed by interleaved merge groups). `info["x_is_permuted"]`
     flips True; downstream layers reuse the layout.
  2. Attention runs the **cute fused FA2 + RoPE block-sparse kernel**
     against the permuted layout. SigLIP has no RoPE, so cos/sin are
     identity tables (`cos=1, sin=0`) — kernel's RoPE step is a no-op.
  3. (Optional) Block forward does ToMe-style merge → MLP → unmerge on
     the residual stream — same broadcast pattern as
     `SparsesamPEPartialBlock`.
  4. Encoder post-hook un-permutes the final hidden states before the
     pooling head and `post_layernorm` see them.

SigLIP-specific adaptations:
  * Separate `q_proj / k_proj / v_proj / out_proj` (vs PE's fused
    `in_proj_weight` / `out_proj`).
  * No RoPE — handled via identity cos/sin in `build_siglip_sparse_cache`.
  * `n_prefix = 0` (no CLS, no register tokens).
  * SigLIP layer has no LayerScale and no DropPath — just
    `norm1 / self_attn / norm2 / mlp` with plain residuals.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .._siglip import (
    SiglipAttention, SiglipEncoderLayer,
    _find_siglip_vision_encoder,
)
from .._siglip_sparse import (
    build_siglip_sparse_cache, flash_block_sparse_attn_siglip,
)
from .. import _pe_stage as _ps
from .._pe_stage_sparse import _ensure_block_mask


def _kernel_dtype(self_attn: SiglipAttention) -> torch.dtype:
    """Pick the dtype for kernel lookup. Under autocast `x.dtype` may be
    fp32 (LayerNorm policy upcasts), but the kernel only supports
    fp16/bf16. Use the projection-weight dtype."""
    w_dtype = self_attn.q_proj.weight.dtype
    if w_dtype in (torch.float16, torch.bfloat16):
        return w_dtype
    return torch.float16


# ── Subclasses ───────────────────────────────────────────────────────────

class SparsesamSiglipAttention(SiglipAttention):
    """Block-sparse cute-kernel attention for SigLIP. Expects permuted
    layout (the layer forward did this once at encoder entry); falls
    back to stock SDPA when the kernel can't be built."""

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        info = self._tome_info
        sr   = info.get("sparse_ratio", info.get("ratio", 1.0))

        if sr >= 1.0:
            return super().forward(hidden_states,
                                   attention_mask=attention_mask, **kwargs)

        kdtype = _kernel_dtype(self)
        kernel, _m_blk, _n_blk = _ps._get_kernel(kdtype, self.head_dim)
        if kernel is None:
            return super().forward(hidden_states,
                                   attention_mask=attention_mask, **kwargs)

        cache = info.get("_perm_cache")
        cache_key = (hidden_states.shape[1], kdtype, sr)
        if cache is None or cache.get("_key") != cache_key:
            cache = build_siglip_sparse_cache(
                self, hidden_states.shape[1], kdtype,
                ratio=sr, group_size=info.get("group_size", 4),
                device=hidden_states.device,
            )
            cache["_key"] = cache_key
            info["_perm_cache"] = cache

        if not cache:
            return super().forward(hidden_states,
                                   attention_mask=attention_mask, **kwargs)

        _ensure_block_mask(cache, self, hidden_states, sr, dtype=kdtype)

        permuted = bool(info.get("x_is_permuted"))
        out = flash_block_sparse_attn_siglip(
            self, hidden_states,
            cos=cache.get("cos"), sin=cache.get("sin"),
            block_mask=cache.get("block_mask"),
            perm=cache.get("perm"), inv_perm=cache.get("inv_perm"),
            assume_permuted=permuted,
        )
        if out is not None:
            return out, None

        # Kernel call failed — un-permute and fall through to stock SDPA.
        if permuted and cache.get("inv_perm") is not None:
            x = hidden_states.index_select(1, cache["inv_perm"])
        else:
            x = hidden_states
        out = super().forward(x, attention_mask=attention_mask, **kwargs)
        if isinstance(out, tuple):
            attn_out, attn_w = out[0], out[1] if len(out) > 1 else None
        else:
            attn_out, attn_w = out, None
        if permuted and cache.get("perm") is not None:
            attn_out = attn_out.index_select(1, cache["perm"])
        return attn_out, attn_w


class SparsesamSiglipLayer(SiglipEncoderLayer):
    """Layer forward: permute once on first call, cute sparse-attn,
    optional ToMe-style merge/MLP/unmerge. Mirrors
    `SparsesamPEPartialBlock` but uses SigLIP's plain pre-norm
    structure (no LayerScale, no DropPath)."""

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        info: dict = self._tome_info
        ratio = info.get("ratio", 1.0)
        sr    = info.get("sparse_ratio", ratio)

        # First layer in the encoder forward: build cache and permute x once.
        if not info.get("x_is_permuted") and sr < 1.0:
            kdtype = _kernel_dtype(self.self_attn)
            kernel, _, _ = _ps._get_kernel(kdtype, self.self_attn.head_dim)
            if kernel is not None:
                cache_key = (hidden_states.shape[1], kdtype, sr)
                cache = info.get("_perm_cache")
                if cache is None or cache.get("_key") != cache_key:
                    cache = build_siglip_sparse_cache(
                        self.self_attn, hidden_states.shape[1], kdtype,
                        ratio=sr, group_size=info.get("group_size", 4),
                        device=hidden_states.device,
                    )
                    cache["_key"] = cache_key
                    info["_perm_cache"] = cache
                perm = cache.get("perm") if cache else None
                if perm is not None:
                    hidden_states = hidden_states.index_select(1, perm)
                    info["x_is_permuted"] = True

        # Attention with residual.
        residual = hidden_states
        h = self.layer_norm1(hidden_states)
        h, _ = self.self_attn(h, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + h

        # MLP with optional merge → MLP → unmerge (PE broadcast pattern).
        cache = info.get("_perm_cache")
        n_merge   = (cache.get("n_merge", 0) if cache else 0)
        mlp_merge = info.get("mlp_merge", True)

        residual = hidden_states
        if (mlp_merge and info.get("x_is_permuted")
                and n_merge > 0 and ratio < 1.0):
            B, S, C = hidden_states.shape
            cls_part_size = cache["cls_part_size"]
            gs            = cache["gs"]

            keep_part     = hidden_states[:, :cls_part_size, :]
            merge_section = hidden_states[:, cls_part_size:, :]
            # merge_section is laid out interleaved (gs, n_merge): rows
            # are member-index, columns are group-index. Pick the first
            # member of each merge group as the representative. With a
            # fixed perm across layers (no per-layer keep-set update),
            # deterministic first-pick beats mean / norm-weighted /
            # max-norm pick — the MLP processes a consistent raster
            # position every layer and broadcasts to its group-mates.
            # Mean and weighted-avg variants produce out-of-distribution
            # inputs; max-norm-pick varies by layer and breaks
            # cross-layer consistency.
            merge_view    = merge_section.reshape(B, gs, n_merge, C)
            merge_repr    = merge_view[:, 0, :, :]

            reduced = torch.cat([keep_part, merge_repr], dim=1)
            reduced = self.mlp(self.layer_norm2(reduced))

            keep_out          = reduced[:, :cls_part_size, :]
            merge_repr_out    = reduced[:, cls_part_size:, :]
            merge_section_out = (merge_repr_out
                                  .unsqueeze(1)
                                  .expand(B, gs, n_merge, C)
                                  .reshape(B, gs * n_merge, C))

            hidden_states = torch.cat([keep_out, merge_section_out], dim=1) + residual
        else:
            hidden_states = residual + self.mlp(self.layer_norm2(hidden_states))

        return hidden_states


# ── Per-forward state hooks ──────────────────────────────────────────────

def _reset_state_hook(info):
    def _hook(_module, _inputs):
        info["x_is_permuted"] = False
    return _hook


def _encoder_unpermute_post_hook(info):
    """Un-permute the encoder output before the pooling head sees it."""
    def _hook(_module, _inputs, output):
        if not info.get("x_is_permuted"):
            return output
        cache = info.get("_perm_cache")
        inv_perm = cache.get("inv_perm") if cache else None
        if inv_perm is None:
            return output
        from transformers.modeling_outputs import BaseModelOutput
        if isinstance(output, BaseModelOutput):
            output.last_hidden_state = output.last_hidden_state.index_select(1, inv_perm)
        elif isinstance(output, torch.Tensor):
            output = output.index_select(1, inv_perm)
        info["x_is_permuted"] = False
        return output
    return _hook


# ── Apply / Remove ───────────────────────────────────────────────────────

def apply_siglip_sparsesam_partial_patch(model: nn.Module,
                                         ratio: float = 0.5,
                                         group_size: int = 4,
                                         sparse_ratio: Optional[float] = None,
                                         start_block: int = 5,
                                         mlp_merge: bool = False,
                                         verbose: bool = True) -> int:
    """Cute block-sparse attention + (optional) ToMe-style
    merge/MLP/unmerge for SigLIP.

    Defaults selected via the autoresearch sweep
    (`tasks/siglip_imagenet/improve_sparsesam.py`, 100+ iterations on
    siglip2-base-patch16-512 / COCO 5K retrieval):

      * `start_block=5`   — preserve early layers, sparsify last 7 of 12
      * `mlp_merge=False` — broadcast-merge MLP destroys SigLIP quality
                            on fixed-perm setups (verified empirically:
                            attn-only > broadcast-first > mean-merge >
                            scatter ≪ random)
      * `ratio=0.5`       — middle of the quality/compression tradeoff
      * `group_size=4`    — saturated above; 4–32 give within 0.001 sum

    Combined with module-level `_DIAG_BAND_WIDTH=3` and
    `_KEEP_BAR_SCALE=2.0` (in `_pe_stage_sparse.py`), this config raises
    i2t R@1 from 65.16% (band=1, kbs=1.0) to 67.98% on COCO 5K.
    """
    encoder = _find_siglip_vision_encoder(model)
    if encoder is None:
        raise RuntimeError("Could not locate SiglipEncoder in `model`.")

    _ps._ensure_cute_deps()
    if _ps.FlashAttentionForwardAmpereRoPE is None:
        raise RuntimeError(
            f"[siglip-sparsesam-partial] cute kernel not importable: "
            f"{_ps._KERNEL_IMPORT_ERROR!r}"
        )

    # Pre-flight: cute kernel must support this head_dim.
    head_dim = encoder.layers[0].self_attn.head_dim
    w_dtype = encoder.layers[0].self_attn.q_proj.weight.dtype
    kdtype = w_dtype if w_dtype in (torch.float16, torch.bfloat16) else torch.float16
    kernel, _, _ = _ps._get_kernel(kdtype, head_dim)
    if kernel is None:
        raise RuntimeError(
            f"[siglip-sparsesam-partial] cute kernel cannot be built for "
            f"(dtype={kdtype}, head_dim={head_dim}). Try base-* checkpoints "
            f"with head_dim=64."
        )

    n_blocks = len(encoder.layers)
    sb = max(0, min(int(start_block), n_blocks))

    info = {
        "ratio":         float(ratio),
        "sparse_ratio":  float(sparse_ratio if sparse_ratio is not None else ratio),
        "group_size":    int(group_size),
        "n_prefix":      0,
        "mlp_merge":     bool(mlp_merge),
        "x_is_permuted": False,
        "_perm_cache":   None,
    }
    model._tome_info = info

    if not hasattr(encoder, "_siglip_partial_pre_hook"):
        encoder._siglip_partial_pre_hook = encoder.register_forward_pre_hook(
            _reset_state_hook(info)
        )
    if not hasattr(encoder, "_siglip_partial_post_hook"):
        encoder._siglip_partial_post_hook = encoder.register_forward_hook(
            _encoder_unpermute_post_hook(info)
        )

    n_attn = 0
    for idx in range(sb, n_blocks):
        layer = encoder.layers[idx]
        attn  = layer.self_attn
        if not isinstance(attn, SparsesamSiglipAttention):
            attn.__class__ = SparsesamSiglipAttention
        attn._tome_info = info
        n_attn += 1
        if not isinstance(layer, SparsesamSiglipLayer):
            layer.__class__ = SparsesamSiglipLayer
        layer._tome_info = info

    if verbose:
        print(f"[siglip-sparsesam-partial] L={n_blocks}  start_block={sb}  "
              f"patched_layers={n_blocks - sb}  patched_attn={n_attn}  "
              f"head_dim={head_dim}  ratio={info['ratio']}  "
              f"sparse_ratio={info['sparse_ratio']}  group_size={info['group_size']}  "
              f"mlp_merge={info['mlp_merge']}  "
              f"(layers 0..{sb - 1}: stock SDPA + full MLP; "
              f"layers {sb}..{n_blocks - 1}: cute sparse-attn + "
              + ("ToMe-style merge/MLP/unmerge"
                 if info['mlp_merge']
                 else "full MLP (permuted)") + ")")
    return n_blocks - sb


def remove_siglip_sparsesam_partial_patch(model: nn.Module) -> int:
    encoder = _find_siglip_vision_encoder(model)
    if encoder is not None:
        for hook_attr in ("_siglip_partial_pre_hook", "_siglip_partial_post_hook"):
            handle = getattr(encoder, hook_attr, None)
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass
                delattr(encoder, hook_attr)
    from ..registry import remove_all_siglip
    return remove_all_siglip(model)


def get_classes() -> Tuple[type, type]:
    return SparsesamSiglipLayer, SparsesamSiglipAttention


__all__ = [
    "apply_siglip_sparsesam_partial_patch",
    "remove_siglip_sparsesam_partial_patch",
    "get_classes",
]
