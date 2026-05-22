"""SpargeAttn patch for the SAM-HQ ViT image encoder.

Class-swaps every `Attention` module to `SpargeSAMAttention`, which routes
q/k/v through `spas_sage2_attn_meansim_topk_cuda_pos` (top-k attention-mass
sparsification) instead of the dense `(q @ k.T) → softmax → @v` path.

The decomposed relative-position bias *is* preserved: SpargeAttn's `_pos`
API takes a per-head L×L additive bias matrix (the same shape SAM's
`add_decomposed_rel_pos` produces), and the sm80/sm86/sm87 kernel adds
it pre-softmax inside the cute kernel.

Constraints (inherited from SpargeAttn):
- head_dim ∈ {64, 128}. SAM-HQ ViT-B/L are 64 ✓; ViT-H is 80 ✗ (will error
  out at apply time — use a different algo or a different SAM variant).
- sequence length ≥ 128. SAM windows (14² = 196) and global grids (≥ 64²)
  both satisfy this.
- Additive `pos` bias is honored on Ampere (sm80–87) only. On Ada/Hopper
  (sm89/sm90) the kernel silently drops `pos`; the patch warns at apply.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from segment_anything.modeling.image_encoder import (
    Attention, ImageEncoderViT, add_decomposed_rel_pos,
)


def _sparge_kernel():
    """Lazy import so SAM eval scripts that don't use sparge don't pay for
    the spas_sage_attn import (and don't fail if it isn't installed)."""
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda_pos
    return spas_sage2_attn_meansim_topk_cuda_pos


class SpargeSAMAttention(Attention):
    """SAM-HQ Attention with the dense softmax replaced by SpargeAttn.

    Uses the `_pos` kernel variant so the decomposed rel-pos bias is
    folded into the attention matrix inside the kernel — no quality
    regression vs. the baseline (apart from the top-k sparsification
    selected via `_sparge_topk`).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        N = H * W
        # qkv: (B, N, 3, num_heads, head_dim) → (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, num_heads, N, head_dim)

        # Build the rel-pos bias using the same code path as the baseline.
        # `add_decomposed_rel_pos` expects q in (B*num_heads, N, D) — we pass
        # an unscaled q (matches baseline: SAM scales q AFTER the bias-shape
        # einsum). Output bias shape: (B*num_heads, N, N).
        if self.use_rel_pos:
            q_flat = q.reshape(B * self.num_heads, N, q.size(-1))
            zero_attn = q.new_zeros((B * self.num_heads, N, N))
            pos = add_decomposed_rel_pos(
                zero_attn, q_flat, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W),
            )  # (B*num_heads, N, N)
        else:
            pos = None

        topk = float(getattr(self, "_sparge_topk", 1.0))

        in_dtype = q.dtype
        if in_dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
            if pos is not None:
                pos = pos.to(torch.float16)

        out = _sparge_kernel()(
            q, k, v, pos=pos, module_scale=self.scale,
            topk=topk, is_causal=False, scale=self.scale,
        )
        if out.dtype != in_dtype:
            out = out.to(in_dtype)

        # (B, num_heads, N, head_dim) → (B, H, W, num_heads*head_dim)
        out = out.transpose(1, 2).reshape(B, H, W, -1)
        return self.proj(out)


def _detect_arch_supports_pos() -> bool:
    """`spas_sage2_attn_meansim_topk_cuda_pos` only honors `pos` on
    sm80/sm86/sm87 (Ampere). Ada/Hopper paths silently drop it."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    cc = f"sm{major}{minor}"
    return cc in ("sm80", "sm86", "sm87")


def apply_patch(
    encoder: ImageEncoderViT,
    algo: str = "sparge",
    ratio: float = 1.0,
    margin: float = 0.5,  # accepted for registry compatibility, unused
    **_: object,
) -> ImageEncoderViT:
    """Patch every Attention module in `encoder` to use SpargeAttn.

    Args:
        encoder: SAM-HQ `ImageEncoderViT`.
        algo: ignored — present for registry compatibility.
        ratio: SpargeAttn `topk` (fraction of attention mass to keep).
            1.0 ≈ dense; lower = faster + sparser. Mirrors `--tome-ratio`.
        margin: accepted but ignored (sparge has no merge margin).
    """
    del algo, margin

    topk = float(ratio if ratio is not None else 1.0)
    if not (0.0 < topk <= 1.0):
        raise ValueError(f"sparge ratio must be in (0, 1]; got {ratio!r}")

    # Eager import so we fail fast at patch time, not deep in the forward.
    _sparge_kernel()

    # Head-dim guard — SpargeAttn requires {64, 128}; SAM-HQ ViT-H (head_dim=80)
    # cannot use this kernel without padding (not implemented here).
    for module in encoder.modules():
        if isinstance(module, Attention):
            head_dim = module.qkv.in_features // module.num_heads
            if head_dim not in (64, 128):
                raise RuntimeError(
                    f"[sparge-sam] head_dim={head_dim} not in {{64, 128}} — "
                    f"SpargeAttn doesn't support this. Use ViT-B/ViT-L "
                    f"(head_dim=64) or pick a different algo."
                )
            break

    pos_supported = _detect_arch_supports_pos()

    n_attn = 0
    n_with_rel_pos = 0
    for module in encoder.modules():
        if isinstance(module, Attention) and not isinstance(module, SpargeSAMAttention):
            module.__class__ = SpargeSAMAttention
            module._sparge_topk = topk
            if getattr(module, "use_rel_pos", False):
                n_with_rel_pos += 1
            n_attn += 1

    if n_with_rel_pos and not pos_supported:
        import warnings
        warnings.warn(
            f"[sparge-sam] this GPU's compute capability does not support "
            f"SpargeAttn's `pos` argument — the decomposed rel-pos bias "
            f"will be silently dropped on {n_with_rel_pos}/{n_attn} "
            f"attention modules. Expect a small mIoU regression.",
            stacklevel=2,
        )

    encoder.tome_info = {"algo": "sparge", "ratio": topk}
    print(f"[sparge-sam] patched {n_attn} Attention module(s)  "
          f"topk={topk}  rel_pos_kept={n_with_rel_pos}  "
          f"pos_arch_supported={pos_supported}")
    return encoder
