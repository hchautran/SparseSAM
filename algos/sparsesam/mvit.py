"""SparseSAM partial for `timm` MViTv2.

MViTv2 differs from PE / SigLIP in three ways that constrain how
sparsesam ports over:

  * **Pooling attention.** Each block's `MultiScaleAttention` runs
    `pool_q / pool_k / pool_v` (kernel 3×3) before SDPA. Q's spatial
    size shrinks at stage transitions (`stride_q=(2,2)` on the first
    block of stages 1/2/3); K/V are pooled too. The cute FA2 kernel
    used by PE/SigLIP sparsesam can't be dropped in here without major
    surgery, so this patch leaves attention untouched and only sparsifies
    the MLP path.

  * **Hierarchical token count.** Stage spatial grids are 56² → 28² →
    14² → 7². Stage 3 (7²=49) isn't divisible by `group_size=4`, so the
    patch is a no-op there. Stages 0–2 (56²/28²/14²) all divide cleanly
    and get the broadcast merge → MLP → unmerge treatment.

  * **No CLS token.** `n_prefix=0` everywhere; every token is a patch.

The MLP path uses the same broadcast-from-first-member pattern as
SigLIP (verified empirically optimal vs mean / norm-weighted /
max-norm / SAM-scatter when the perm is fixed across layers): after
the attention residual, permute via uniform-stride z-order; pick the
first member of each merge group as the representative; run MLP on
[keep groups + representatives]; broadcast each representative's MLP
output to all `gs-1` other group members; un-permute back to natural
order before returning.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from timm.models.mvitv2 import MultiScaleBlock, MultiScaleAttention
from .._pe_stage_sparse import _get_uniform_stride_perm


# Per-(stage, ratio) perm cache shared across blocks within a stage.
_MVIT_PERM_CACHE: dict = {}


def _get_mvit_stage_perm(H: int, W: int, ratio: float, group_size: int,
                         device) -> dict:
    """Build (or fetch) the uniform-stride z-order perm for an MViTv2
    stage's spatial grid. Returns dict with perm, inv_perm, n_merge,
    cls_part_size, gs."""
    cache_key = (H, W, float(ratio), int(group_size), str(device))
    cached = _MVIT_PERM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    S = H * W
    if S <= 0 or S % group_size != 0:
        return None

    # n_block doesn't matter here (we don't use the cute kernel for MViTv2).
    perm, inv_perm = _get_uniform_stride_perm(
        S=S, ratio=ratio, group_size=group_size, n_block=64,
        has_cls=False, device=device,
    )

    # MLP keep/merge counts (mirrors PE / SigLIP cache).
    n_groups = S // group_size
    K = max(1, round(ratio * S))
    if K >= n_groups:
        n_keep = max(0, (K - n_groups) // (group_size - 1))
        n_keep = min(n_keep, n_groups)
    else:
        n_keep = 0
    n_merge       = n_groups - n_keep
    cls_part_size = n_keep * group_size

    if n_merge == 0:
        return None

    cache = {
        "perm": perm, "inv_perm": inv_perm,
        "n_merge": n_merge,
        "cls_part_size": cls_part_size,
        "gs": int(group_size),
    }
    _MVIT_PERM_CACHE[cache_key] = cache
    return cache


# ── Subclass ──────────────────────────────────────────────────────────

class SparsesamMvitBlock(MultiScaleBlock):
    """MViTv2 block with sparsesam-style broadcast merge → MLP →
    unmerge on the residual stream. Attention is unchanged."""

    def forward(self, x: torch.Tensor, feat_size: List[int]):
        info: dict = self._tome_info
        ratio     = info.get("ratio", 1.0)
        group_size = info.get("group_size", 4)
        mlp_merge  = info.get("mlp_merge", True)

        # ── Attention residual (stock MViTv2) ──────────────────────────
        x_norm     = self.norm1(x)
        x_shortcut = x if self.shortcut_proj_attn is None else self.shortcut_proj_attn(x_norm)
        x_shortcut = self._shortcut_pool(x_shortcut, feat_size)
        x_attn, feat_size_new = self.attn(x_norm, feat_size)
        x = x_shortcut + self.drop_path1(x_attn)

        # ── MLP path: optionally compress via broadcast first-pick ─────
        x_norm = self.norm2(x)
        x_shortcut_mlp = x if self.shortcut_proj_mlp is None else self.shortcut_proj_mlp(x_norm)

        cache = None
        if mlp_merge and ratio < 1.0:
            H, W = int(feat_size_new[0]), int(feat_size_new[1])
            if H == W and H * W == x_norm.shape[1]:
                cache = _get_mvit_stage_perm(H, W, ratio, group_size, x_norm.device)

        if cache is not None:
            perm     = cache["perm"]
            inv_perm = cache["inv_perm"]
            n_merge       = cache["n_merge"]
            cls_part_size = cache["cls_part_size"]
            gs            = cache["gs"]

            # Permute the MLP input into [keep | merge-interleaved] layout.
            x_perm = x_norm.index_select(1, perm)

            B, S, C = x_perm.shape
            keep_part     = x_perm[:, :cls_part_size, :]
            merge_section = x_perm[:, cls_part_size:, :]
            # First-pick: take member 0 of each merge group as the representative.
            merge_view    = merge_section.reshape(B, gs, n_merge, C)
            merge_repr    = merge_view[:, 0, :, :]

            reduced = torch.cat([keep_part, merge_repr], dim=1)
            mlp_out = self.mlp(reduced)

            keep_out          = mlp_out[:, :cls_part_size, :]
            merge_repr_out    = mlp_out[:, cls_part_size:, :]
            merge_section_out = (merge_repr_out
                                  .unsqueeze(1)
                                  .expand(B, gs, n_merge, C)
                                  .reshape(B, gs * n_merge, C))

            mlp_full_perm = torch.cat([keep_out, merge_section_out], dim=1)
            mlp_full      = mlp_full_perm.index_select(1, inv_perm)

            x = x_shortcut_mlp + self.drop_path2(mlp_full)
        else:
            x = x_shortcut_mlp + self.drop_path2(self.mlp(x_norm))

        return x, feat_size_new


# ── Apply / Remove ────────────────────────────────────────────────────

def apply_mvit_sparsesam_partial_patch(model: nn.Module,
                                       ratio: float = 0.7,
                                       group_size: int = 4,
                                       start_block: int = 0,
                                       mlp_merge: bool = True,
                                       verbose: bool = True) -> int:
    """Patch every `MultiScaleBlock` (from `start_block` onwards in the
    flattened block list) to do broadcast merge → MLP → unmerge.

    `start_block` indexes the *flattened* block sequence (across stages)
    so `start_block=5` skips stages 0–1 and the first 2 blocks of stage 2.
    """
    info = {
        "ratio":      float(ratio),
        "group_size": int(group_size),
        "mlp_merge":  bool(mlp_merge),
    }
    model._tome_info = info

    # Flatten blocks in stage order.
    flat: List[MultiScaleBlock] = []
    for stage in model.stages:
        for blk in stage.blocks:
            flat.append(blk)

    sb = max(0, min(int(start_block), len(flat)))
    n_patched = 0
    skipped_unsupported: List[Tuple[int, int]] = []
    for idx in range(sb, len(flat)):
        blk = flat[idx]
        if not isinstance(blk, SparsesamMvitBlock):
            blk.__class__ = SparsesamMvitBlock
        blk._tome_info = info
        n_patched += 1

    if verbose:
        # Per-stage summary.
        stage_summary = []
        for si, stage in enumerate(model.stages):
            for bi, blk in enumerate(stage.blocks):
                if isinstance(blk, SparsesamMvitBlock):
                    stage_summary.append(f"s{si}b{bi}")
        print(f"[mvit-sparsesam-partial] flat_blocks={len(flat)}  "
              f"start_block={sb}  patched={n_patched} ({','.join(stage_summary)})  "
              f"ratio={info['ratio']}  group_size={info['group_size']}  "
              f"mlp_merge={info['mlp_merge']}  "
              f"(blocks where H≠W or H·W not divisible by group_size become "
              f"stock at runtime — expect stage 3 (7×7=49) to skip)")
    return n_patched


def remove_mvit_sparsesam_partial_patch(model: nn.Module) -> int:
    n = 0
    for module in model.modules():
        cls = type(module)
        if cls is not MultiScaleBlock and issubclass(cls, MultiScaleBlock):
            module.__class__ = MultiScaleBlock
            n += 1
        if hasattr(module, "_tome_info"):
            try:
                del module._tome_info
            except AttributeError:
                pass
    if hasattr(model, "_tome_info"):
        try:
            del model._tome_info
        except AttributeError:
            pass
    _MVIT_PERM_CACHE.clear()
    return n


def get_classes() -> Tuple[type, type]:
    return SparsesamMvitBlock, MultiScaleAttention


__all__ = [
    "SparsesamMvitBlock",
    "apply_mvit_sparsesam_partial_patch",
    "remove_mvit_sparsesam_partial_patch",
    "get_classes",
]
