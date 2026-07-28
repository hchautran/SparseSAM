"""Piecewise Sparse Attention patch for the SAM-HQ ViT image encoder.

This mirrors `algos.sparge.sam`: every SAM `Attention` module is class-swapped
to a drop-in sparse-attention implementation. The decomposed relative-position
bias is preserved by calling `piecewise_sparse_attention_pos`.
"""

from __future__ import annotations

import os
import sys

import torch

from segment_anything.modeling.image_encoder import (
    Attention,
    ImageEncoderViT,
    add_decomposed_rel_pos,
)


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_THIRD_PARTY = os.path.join(_REPO, "algos", "3rd_party", "piecewise-sparse-attention")
if os.path.isdir(_THIRD_PARTY) and _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)


def _piecewise_kernel():
    """Lazy import so unrelated evals do not require the optional Triton dep."""
    from piecewise_attn import piecewise_sparse_attention_pos

    return piecewise_sparse_attention_pos


class PiecewiseSAMAttention(Attention):
    """SAM-HQ Attention using PISA with decomposed relative-position bias."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        N = H * W

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.use_rel_pos:
            q_flat = q.reshape(B * self.num_heads, N, q.size(-1))
            zero_attn = q.new_zeros((B * self.num_heads, N, N))
            pos = add_decomposed_rel_pos(
                zero_attn,
                q_flat,
                self.rel_pos_h,
                self.rel_pos_w,
                (H, W),
                (H, W),
            ).reshape(B, self.num_heads, N, N)
        else:
            pos = q.new_zeros((B, self.num_heads, N, N))

        density = float(getattr(self, "_piecewise_density", 1.0))
        block_size = int(getattr(self, "_piecewise_block_size", 64))

        in_dtype = q.dtype
        if in_dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
            pos = pos.to(torch.float16)

        out = _piecewise_kernel()(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            pos.contiguous(),
            density=density,
            block_size=block_size,
            scale=self.scale,
        )
        if out.dtype != in_dtype:
            out = out.to(in_dtype)

        out = out.transpose(1, 2).reshape(B, H, W, -1)
        return self.proj(out)


def apply_patch(
    encoder: ImageEncoderViT,
    algo: str = "piecewise",
    ratio: float = 1.0,
    margin: float = 0.5,
    piecewise_block_size: int = 64,
    **_: object,
) -> ImageEncoderViT:
    """Patch every SAM-HQ Attention module to use PISA.

    Args:
        encoder: SAM-HQ `ImageEncoderViT`.
        algo: accepted for registry compatibility.
        ratio: PISA density. Lower values compute fewer exact blocks.
        margin: accepted for registry compatibility.
        piecewise_block_size: block size passed to the Triton kernel.
    """
    del algo, margin

    density = float(ratio if ratio is not None else 1.0)
    if not (0.0 < density <= 1.0):
        raise ValueError(f"piecewise ratio/density must be in (0, 1]; got {ratio!r}")
    if int(piecewise_block_size) <= 0:
        raise ValueError(f"piecewise_block_size must be positive; got {piecewise_block_size!r}")

    _piecewise_kernel()

    n_attn = 0
    n_with_rel_pos = 0
    for module in encoder.modules():
        if isinstance(module, Attention) and not isinstance(module, PiecewiseSAMAttention):
            module.__class__ = PiecewiseSAMAttention
            module._piecewise_density = density
            module._piecewise_block_size = int(piecewise_block_size)
            if getattr(module, "use_rel_pos", False):
                n_with_rel_pos += 1
            n_attn += 1

    encoder.tome_info = {
        "algo": "piecewise",
        "ratio": density,
        "piecewise_block_size": int(piecewise_block_size),
    }
    print(
        f"[piecewise-sam] patched {n_attn} Attention module(s)  "
        f"density={density}  block_size={int(piecewise_block_size)}  "
        f"rel_pos_kept={n_with_rel_pos}"
    )
    return encoder

