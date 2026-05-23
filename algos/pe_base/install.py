"""apply_* / remove_* entry points for PE patches + helpers to locate
the PE VisionTransformer / Transformer modules."""

from __future__ import annotations
from typing import List, Optional

import torch.nn as nn

from . import cute_kernel as _ck
from .classes import (
    SelfAttention, ResidualAttentionBlock,
    VisionTransformer, Transformer,
    FlashRopeOnlyPEAttention,
)


def _stage_end_indices(n_blocks: int, num_stages: int) -> List[int]:
    """Indices of blocks AFTER which compression happens. Length = num_stages-1."""
    if num_stages <= 1 or n_blocks < num_stages:
        return []
    chunk = n_blocks // num_stages
    return [chunk * (i + 1) - 1 for i in range(num_stages - 1)]


def _find_vision_transformer(model: nn.Module):
    for m in model.modules():
        if isinstance(m, VisionTransformer):
            for sub in m.modules():
                if isinstance(sub, Transformer):
                    return sub
    return None


def _vit_uses_cls_token(model: nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, VisionTransformer):
            return bool(getattr(m, "use_cls_token", False))
    return False


def _reset_active_idx_hook(info):
    def _hook(_module, _inputs):
        info["active_idx"] = None
        info["_stage_cache"] = None
    return _hook


def _reset_sparse_state_hook(info):
    def _hook(_module, _inputs):
        info["active_idx"] = None
        info["_stage_cache"] = None
        info["x_is_permuted"] = False
    return _hook


def _transformer_unpermute_post_hook(info):
    """If the last stage left x in permuted layout, restore natural order
    before the Transformer returns."""
    def _hook(_module, _inputs, output):
        if not info.get("x_is_permuted"):
            return output
        cache = info.get("_stage_cache")
        inv_perm = cache.get("inv_perm") if cache else None
        if inv_perm is not None:
            output = output.index_select(1, inv_perm)
        info["x_is_permuted"] = False
        return output
    return _hook


def apply_stage_compress(model: nn.Module,
                         compress_block_class: type,
                         attn_class: type,
                         info: dict,
                         num_stages: int,
                         use_flash_rope: bool,
                         compress_at_blocks: Optional[List[int]] = None,
                         verbose_tag: str = "pe-stage") -> int:
    """Install `compress_block_class` at stage-end indices and `attn_class`
    on every SelfAttention with rope. State lives on `model._tome_info`.

    `info` must contain at least `ratio`, `num_stages`, `use_flash_rope`,
    plus algo-specific fields. `compress_at_blocks` overrides `num_stages`.

    Returns the number of compression points installed.
    """
    transformer = _find_vision_transformer(model)
    if transformer is None:
        raise RuntimeError("Could not locate the PE vision Transformer in `model`.")

    if use_flash_rope:
        _ck._ensure_cute_deps()
        if _ck.FlashAttentionForwardAmpereRoPE is None and verbose_tag:
            print(f"[{verbose_tag}] use_flash_rope=True but cute kernel not "
                  f"importable ({_ck._KERNEL_IMPORT_ERROR!r}); will fall back "
                  f"at runtime to stock SDPA.")

    info.setdefault("use_cls_token", _vit_uses_cls_token(model))
    info["active_idx"] = None
    info["_stage_cache"] = None
    model._tome_info = info

    if not hasattr(transformer, "_pe_compress_pre_hook"):
        transformer._pe_compress_pre_hook = transformer.register_forward_pre_hook(
            _reset_active_idx_hook(info)
        )

    n_attn = 0
    for mod in model.modules():
        if isinstance(mod, SelfAttention) and mod.rope is not None:
            if not isinstance(mod, attn_class):
                mod.__class__ = attn_class
            mod._tome_info = info
            n_attn += 1

    n_blocks = len(transformer.resblocks)
    stage_ends = (sorted({int(i) for i in compress_at_blocks if 0 <= int(i) < n_blocks})
                  if compress_at_blocks is not None
                  else _stage_end_indices(n_blocks, num_stages))

    for idx in stage_ends:
        blk = transformer.resblocks[idx]
        if not isinstance(blk, compress_block_class):
            blk.__class__ = compress_block_class
        blk._tome_info = info

    if verbose_tag:
        mode = "explicit" if compress_at_blocks is not None else f"num_stages={num_stages}"
        print(f"[{verbose_tag}] L={n_blocks}  {mode}  "
              f"compress_after_blocks={stage_ends}  ratio={info.get('ratio')}  "
              f"use_cls_token={info['use_cls_token']}  patched_attn={n_attn}  "
              f"use_flash_rope={info.get('use_flash_rope', False)}")
    return len(stage_ends)


def apply_stage_compress_sparse(model: nn.Module,
                                compress_block_class: type,
                                attn_class: type,
                                info: dict,
                                num_stages: int,
                                compress_at_blocks: Optional[list] = None,
                                verbose_tag: str = "pe-stage-sparse") -> int:
    """Install block-sparse stage-compression patch. Requires the cute kernel
    to be importable; raises if not. `info` must contain `ratio`, `num_stages`,
    `group_size`, plus algo-specific fields. Optional `sparse_ratio` (defaults
    to `ratio`) sets the keep-bar width in the cute mask."""
    transformer = _find_vision_transformer(model)
    if transformer is None:
        raise RuntimeError("Could not locate the PE vision Transformer in `model`.")

    _ck._ensure_cute_deps()
    if _ck.FlashAttentionForwardAmpereRoPE is None:
        raise RuntimeError(
            f"[{verbose_tag}] cute kernel not importable: "
            f"{_ck._KERNEL_IMPORT_ERROR!r} — block-sparse patch requires the cute kernel."
        )

    info.setdefault("use_cls_token", _vit_uses_cls_token(model))
    info.setdefault("sparse_ratio", float(info.get("ratio", 1.0)))
    info["active_idx"] = None
    info["_stage_cache"] = None
    info["x_is_permuted"] = False
    model._tome_info = info

    if not hasattr(transformer, "_pe_compress_pre_hook"):
        transformer._pe_compress_pre_hook = transformer.register_forward_pre_hook(
            _reset_sparse_state_hook(info)
        )
    if not hasattr(transformer, "_pe_compress_post_hook"):
        transformer._pe_compress_post_hook = transformer.register_forward_hook(
            _transformer_unpermute_post_hook(info)
        )

    n_attn = 0
    for mod in model.modules():
        if isinstance(mod, SelfAttention) and mod.rope is not None:
            if not isinstance(mod, attn_class):
                mod.__class__ = attn_class
            mod._tome_info = info
            n_attn += 1

    n_blocks = len(transformer.resblocks)
    stage_ends = (sorted({int(i) for i in compress_at_blocks if 0 <= int(i) < n_blocks})
                  if compress_at_blocks is not None
                  else _stage_end_indices(n_blocks, num_stages))

    for idx in stage_ends:
        blk = transformer.resblocks[idx]
        if not isinstance(blk, compress_block_class):
            blk.__class__ = compress_block_class
        blk._tome_info = info

    if verbose_tag:
        mode = "explicit" if compress_at_blocks is not None else f"num_stages={num_stages}"
        print(f"[{verbose_tag}] L={n_blocks}  {mode}  "
              f"compress_after_blocks={stage_ends}  ratio={info.get('ratio')}  "
              f"sparse_ratio={info.get('sparse_ratio')}  "
              f"use_cls_token={info['use_cls_token']}  patched_attn={n_attn}  "
              f"(pre-compress: SDPA, post-compress: cute sparse)")
    return len(stage_ends)


def apply_pe_flash_rope_patch(model: nn.Module, verbose: bool = True) -> int:
    """Replace every PE SelfAttention with `FlashRopeOnlyPEAttention` — pure
    cute-kernel swap, no token compression. Falls back at runtime if the
    kernel can't be built for this (dtype, head_dim)."""
    _ck._ensure_cute_deps()
    if _ck.FlashAttentionForwardAmpereRoPE is None:
        msg = (f"[pe-flash-rope] cute kernel not importable: "
               f"{_ck._KERNEL_IMPORT_ERROR!r} — patch not applied")
        if verbose:
            print(msg)
        raise RuntimeError(msg)

    n = 0
    for mod in model.modules():
        if isinstance(mod, SelfAttention) and mod.rope is not None:
            if not isinstance(mod, FlashRopeOnlyPEAttention):
                mod.__class__ = FlashRopeOnlyPEAttention
            n += 1

    if verbose:
        print(f"[pe-flash-rope] patched {n} SelfAttention module(s)")
    return n


# Thin back-compat shims — registry's `remove_all_pe` does the actual work.
def remove_stage_compress(model: nn.Module) -> int:
    from ..registry import remove_all_pe
    return remove_all_pe(model)


def remove_stage_compress_sparse(model: nn.Module) -> int:
    from ..registry import remove_all_pe
    return remove_all_pe(model)


def remove_pe_flash_rope_patch(model: nn.Module) -> int:
    from ..registry import remove_all_pe
    return remove_all_pe(model)
