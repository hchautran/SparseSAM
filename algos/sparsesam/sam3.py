"""SparseSAM monkey patch for the vendored Meta SAM3 ViT encoder.

Targets `sam3.model.vitdet.Block` (the per-layer transformer block used by
`sam3.model.vitdet.ViT`, which is the shared visual backbone of SAM3's
detector / tracker). The strategy is the MLP-merge variant of SparseSAM:

  * Attention runs *unchanged* on the full token set. SAM3 uses 2-D RoPE with
    `freqs_cis` precomputed for the full 72x72 grid, so reducing tokens
    before attn would invalidate position encoding. Keeping attn full is the
    safe path.
  * The MLP is executed only on `round(ratio * N)` "keep" tokens picked at
    uniform stride over a Z-order traversal of the (H, W) grid. The
    remaining tokens skip the MLP update entirely (residual passes through).

Speedup comes from MLP cost being roughly half of the per-block compute. The
patch is applied to every block (both windowed and global); for windowed
blocks the MLP still runs on the full unpartitioned (B, H, W, C) tensor, so
the savings apply uniformly.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Optional

import torch
import torch.nn as nn

from .z_utils import get_z_order

_here = os.path.dirname(__file__)
_sam3_root = os.path.normpath(os.path.join(_here, "..", "3rd_party", "sam3"))
if _sam3_root not in sys.path:
    sys.path.insert(0, _sam3_root)

from sam3.model.vitdet import (
    Attention,
    Block,
    window_partition,
    window_unpartition,
)


_KEEP_PERM_CACHE: dict = {}


def _get_keep_indices(H: int, W: int, ratio: float, device) -> torch.Tensor:
    """Return raster-order indices of the kept tokens.

    Tokens are first traversed in Z (Morton) order; we then sample
    `round(ratio*N)` indices at uniform stride from that traversal. The
    result is a 1-D LongTensor of original raster positions (no batch dim)
    that can be used directly with `gather`/`scatter` over a (B, H*W, C)
    sequence."""
    N = H * W
    keep_n = max(1, round(ratio * N))
    keep_n = min(keep_n, N)
    key = (H, W, keep_n, str(device))
    cached = _KEEP_PERM_CACHE.get(key)
    if cached is not None:
        return cached

    z_perm = get_z_order(H, W, device=device)
    stride_pos = torch.round(
        torch.arange(keep_n, device=device, dtype=torch.float32) * (N / keep_n)
    ).long().clamp_(0, N - 1)
    keep_raster = z_perm.index_select(0, stride_pos).contiguous()
    _KEEP_PERM_CACHE[key] = keep_raster
    return keep_raster


class ToMeSAM3Block(Block):
    """Block forward with MLP-merge: full attn, partial MLP on keep set."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        info = self._tome_info
        ratio = float(info["ratio"])
        mlp_merge = bool(info.get("mlp_merge", True))

        # ── attention (unchanged from stock Block.forward) ────────────────
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H_w, W_w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.ls1(self.attn(x))
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H_w, W_w))
        x = shortcut + self.dropout(self.drop_path(x))

        # ── MLP (optionally on keep-set only) ─────────────────────────────
        do_partial = mlp_merge and ratio < 1.0 and x.ndim == 4
        if do_partial:
            B, H, W, C = x.shape
            N = H * W
            x_seq = x.reshape(B, N, C)
            keep_idx = _get_keep_indices(H, W, ratio, x.device)
            idx_e = keep_idx.view(1, -1, 1).expand(B, -1, C)

            x_kept = x_seq.gather(1, idx_e)
            x_kept = x_kept + self.dropout(
                self.drop_path(self.ls2(self.mlp(self.norm2(x_kept))))
            )
            x_seq = x_seq.scatter(1, idx_e, x_kept)
            return x_seq.reshape(B, H, W, C)

        return x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))


def apply_patch(
    encoder: nn.Module,
    ratio: float = 0.9,
    mlp_merge: bool = True,
    **_,
) -> nn.Module:
    """Patch every `Block` in a SAM3 ViT encoder with `ToMeSAM3Block`.

    `encoder` may be either:
      * a `sam3.model.vitdet.ViT` directly, or
      * any container holding one in its module tree (e.g. the full
        `Sam3Image` model or its `backbone` / vision-neck wrapper).

    `ratio` in (0, 1]: fraction of tokens that receive an MLP update.
    `mlp_merge=False` falls back to the stock full MLP (ratio is ignored).
    """
    assert 0.0 < ratio <= 1.0, f"ratio must be in (0, 1], got {ratio}"

    info = {"ratio": float(ratio), "mlp_merge": bool(mlp_merge)}
    encoder.tome_info = info

    n_patched = 0
    for module in encoder.modules():
        if isinstance(module, Block) and not isinstance(module, ToMeSAM3Block):
            module.__class__ = ToMeSAM3Block
            module._tome_info = info
            n_patched += 1

    if n_patched == 0:
        raise RuntimeError(
            "apply_patch(sam3): no sam3.model.vitdet.Block found in encoder; "
            "pass the SAM3 ViT (or a model containing it)."
        )

    n_global = n_windowed = 0
    for m in encoder.modules():
        if isinstance(m, ToMeSAM3Block):
            if m.window_size == 0:
                n_global += 1
            else:
                n_windowed += 1
    print(
        f"[ToMe-SAM3] patched  ratio={ratio}  mlp_merge={mlp_merge}  "
        f"blocks={n_patched} (global={n_global} windowed={n_windowed})  "
        f"strategy={'partial-MLP keep-set' if mlp_merge else 'full-MLP'}"
    )
    return encoder


def remove_patch(encoder: nn.Module) -> int:
    """Revert every patched block to stock `Block`. Idempotent."""
    n = 0
    for module in encoder.modules():
        if type(module) is ToMeSAM3Block:
            module.__class__ = Block
            module.__dict__.pop("_tome_info", None)
            n += 1
    encoder.__dict__.pop("tome_info", None)
    return n


__all__ = ["apply_patch", "remove_patch", "ToMeSAM3Block"]
