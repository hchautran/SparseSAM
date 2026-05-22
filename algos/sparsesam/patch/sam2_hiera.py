"""Sparsesam patch for SAM2's Hiera trunk.

Mirrors algos/sparsesam/sam.py for SAM-HQ ViT, ported to:
  * Hiera's `MultiScaleBlock` + `MultiScaleAttention` (vs SAM-HQ's
    `Block` + `Attention`).
  * Stage 2 only — that's where 16 of 24 blocks live, all at N=4096
    on the 64×64 grid. Stages 0/1/3 are skipped (too few blocks at
    the wrong resolution to benefit from gs=4 Z-grouping).
  * No FA2 cute kernel — Hiera uses plain `F.scaled_dot_product_attention`.
    Our partial-MLP path is independent of attention, so this is
    actually simpler than the SAM-HQ patch.

The grouped-perm partial-MLP from
[../sam.py:393-410] ports verbatim, using the `inv_perm_1d_group`
output of `tile_stride_matching`.

Usage:
    from algos.sparsesam.patch.sam2_hiera import apply_patch
    apply_patch(model.image_encoder.trunk, ratio=0.5, prune_mlp=True)
"""

from __future__ import annotations

import types
from typing import Optional, List

import torch
import torch.nn.functional as F

from ..sam import tile_stride_matching


# ─────────────────────────────────────────────────────────────────────────
# Patched forwards (bound onto each block / attention via types.MethodType)
# ─────────────────────────────────────────────────────────────────────────

def _patched_attn_forward(self, x):
    """`MultiScaleAttention.forward` with a side-effect: at the global
    blocks (window_size==0, no windowing → k is full-resolution),
    compute and cache the `inv_perm_1d_group` for partial-MLP keep
    selection. For SAM2 Hiera, this is blocks 23/33/43 of stage 2.

    Earlier cache population (at local stage-2 blocks like block 9)
    is handled in `_patched_block_forward` via a separate
    full-resolution qkv projection on the block's pre-window input —
    necessary because local blocks see windowed input here, so their
    `k` is at window resolution, not the full grid.
    """
    from sam2.modeling.backbones.hieradet import do_pool

    info = getattr(self, "_tome_info", None)

    B, H, W, _ = x.shape
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
    q, k, v = torch.unbind(qkv, 2)                      # each (B, H*W, nh, d)

    # Cache only at global blocks (window_size==0 → x is the full grid,
    # so k is full-resolution). Local blocks populate the cache from
    # the block forward, which has access to pre-window input.
    # Restrict to the target stage's resolution (target_N) so we don't
    # touch stage 0/1/3 features that the FpnNeck depends on.
    target_N = info.get("target_N") if info is not None else None
    should_cache = (
        info is not None
        and getattr(self, "_is_global", False)
        and info.get("mlp_merge", False)
        and info.get("ratio", 1.0) < 1.0
        and (target_N is None or H * W == target_N)
    )
    if should_cache:
        ratio = info["ratio"]
        gs    = info.get("group_size", 4)
        N     = H * W
        if N % gs == 0:
            perm_cache = info.setdefault("perm_cache", {})
            cache_key  = (H, W, ratio)
            if cache_key not in perm_cache:
                # k → (B*nh, N, d) so tile_stride_matching gets per-head
                # features, mirroring SAM-HQ ([../sam.py:325]).
                nh = self.num_heads
                k_flat = (k.permute(0, 2, 1, 3)            # (B, nh, N, d)
                            .reshape(B * nh, N, -1)
                            .float())
                _perm_1d, _inv_perm_1d, inv_perm_1d_group = tile_stride_matching(
                    k_flat, H, W, ratio=ratio, group_size=gs,
                )
                # Store only what the MLP path needs.
                perm_cache[cache_key] = inv_perm_1d_group

    # ── stock attention math (mirrors hieradet.MultiScaleAttention.forward) ──
    if self.q_pool:
        q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
        H, W = q.shape[1:3]
        q = q.reshape(B, H * W, self.num_heads, -1)

    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    )
    out = out.transpose(1, 2)
    out = out.reshape(B, H, W, -1)
    return self.proj(out)


def _patched_block_forward(self, x):
    """`MultiScaleBlock.forward` with grouped-perm partial-MLP at
    *local* blocks of the target stage. Reads the per-head grouped
    inverse-perm from `info["perm_cache"]` (populated by the most
    recent global block at this resolution), takes top-keep_n by
    avg-rank, runs MLP only on those tokens, scatters the residual
    back. Dropped tokens skip MLP (production sparsesam semantics)."""
    from sam2.modeling.backbones.hieradet import (
        do_pool, window_partition, window_unpartition,
    )

    info  = self._tome_info
    ratio = info.get("ratio", 1.0)

    # ── stock block prelude: norm1, optional skip-proj, window partition ──
    shortcut = x                                          # B, H, W, C
    x = self.norm1(x)
    if self.dim != self.dim_out:
        shortcut = do_pool(self.proj(x), self.pool)

    H_full, W_full = x.shape[1], x.shape[2]

    # ── EARLY-CACHE — populate the partial-MLP perm at the FIRST
    # non-q-pool block of the TARGET STAGE so local blocks BEFORE the
    # first global can use partial-MLP. We restrict to the target
    # stage (default: stage 2 at 64×64 for SAM2 Hiera) because
    # applying partial-MLP at stage 0/1 (256×256, 128×128) destroys
    # low-level features the FpnNeck depends on, and the extra qkv
    # projection at those resolutions costs ~120 MB.
    #
    # Costs one extra qkv projection at the populating block. We need
    # to project on the full-resolution input to get full-resolution
    # k — local blocks' attention sees windowed input, so we can't
    # recover full-resolution k from the patched attention.
    gs = info.get("group_size", 4)
    target_N = info.get("target_N")
    if (info.get("mlp_merge", False) and ratio < 1.0
            and self.q_stride is None
            and target_N is not None
            and (H_full * W_full) == target_N
            and (H_full * W_full) % gs == 0
            and (H_full, W_full, ratio) not in info.get("perm_cache", {})):
        N_in = H_full * W_full
        attn = self.attn
        nh = attn.num_heads
        with torch.no_grad():
            qkv_full = F.linear(x, attn.qkv.weight, attn.qkv.bias)        # (B, H, W, 3·dim_out)
            qkv_full = qkv_full.reshape(x.shape[0], N_in, 3, nh, -1)
            k_full   = qkv_full[:, :, 1]                                  # (B, N, nh, d)
            k_flat   = (k_full.permute(0, 2, 1, 3)
                              .reshape(x.shape[0] * nh, N_in, -1)
                              .float())
            from ..sam import tile_stride_matching as _tsm
            _p, _i, inv_group = _tsm(
                k_flat, H_full, W_full, ratio=ratio, group_size=gs,
            )
            del qkv_full, k_full, k_flat                                  # free 120MB-class allocations
        info.setdefault("perm_cache", {})[(H_full, W_full, ratio)] = inv_group

    window_size = self.window_size
    pad_hw: Optional[tuple] = None
    if window_size > 0:
        x, pad_hw = window_partition(x, window_size)

    # Attention (also caches the grouped perm if this is a global block).
    x = self.attn(x)

    # Adjust window/H/W after potential q-pool.
    if self.q_stride:
        window_size = self.window_size // self.q_stride[0]
        H_full, W_full = shortcut.shape[1:3]
        pad_h = (window_size - H_full % window_size) % window_size
        pad_w = (window_size - W_full % window_size) % window_size
        pad_hw = (H_full + pad_h, W_full + pad_w)

    if self.window_size > 0:
        x = window_unpartition(x, window_size, pad_hw, (H_full, W_full))

    x = shortcut + self.drop_path(x)                     # post-attn residual

    # ── partial-MLP path ──
    # Eligible iff:
    #   (a) `mlp_merge` enabled and ratio < 1.0
    #   (b) this is a LOCAL block (window_size > 0) — globals do full MLP
    #   (c) a perm has been cached at this resolution by a prior global
    #   (d) N is divisible by gs
    mlp_merge = info.get("mlp_merge", True)
    gs        = info.get("group_size", 4)
    H_post, W_post, C = x.shape[1], x.shape[2], x.shape[3]
    N = H_post * W_post

    cached: Optional[torch.Tensor] = None
    target_N = info.get("target_N")
    if (mlp_merge and ratio < 1.0 and self.window_size > 0
            and N % gs == 0
            and (target_N is None or N == target_N)):
        cached = info.get("perm_cache", {}).get((H_post, W_post, ratio))

    if cached is not None:
        inv_perm_1d_group = cached
        B = x.shape[0]
        nh = self.attn.num_heads
        x_seq = x.reshape(B, N, C)
        keep_n = max(1, round(ratio * N))
        # Avg head rank across heads → group-coherent keep selection
        # (mirrors [../sam.py:402]).
        avg_rank = inv_perm_1d_group.view(B, nh, -1).float().mean(dim=1)
        top_idx  = avg_rank.topk(keep_n, dim=1, largest=False).indices
        idx_e    = top_idx.unsqueeze(-1).expand(-1, -1, C)
        x_kept   = x_seq.gather(1, idx_e)
        x_kept   = x_kept + self.drop_path(self.mlp(self.norm2(x_kept)))
        x_seq    = x_seq.scatter(1, idx_e, x_kept)
        return x_seq.reshape(B, H_post, W_post, C)

    # Full-MLP fallback (globals, q-pool blocks, blocks before any
    # global has run, or `mlp_merge=False`).
    return x + self.drop_path(self.mlp(self.norm2(x)))


# ─────────────────────────────────────────────────────────────────────────
# Patch / unpatch
# ─────────────────────────────────────────────────────────────────────────

def _reset_perm_cache(info):
    """Forward-pre hook: clear `perm_cache` at the start of each
    encoder forward so per-image perms don't leak across batches."""
    def _hook(_module, _inputs):
        info["perm_cache"] = {}
    return _hook


def apply_patch(trunk, ratio: float = 0.5, group_size: int = 4,
                prune_mlp: bool = True, use_cute: bool = False,
                target_N: Optional[int] = 4096, start_block: int = 0, **_):
    """Patch a Hiera trunk for sparsesam partial-MLP.

    Args:
        trunk:        `sam2.modeling.backbones.hieradet.Hiera` instance.
        ratio:        keep ratio in (0, 1]; fraction of MLP forwards.
        group_size:   Z-curve group size (default 4 = 2×2 spatial quad).
        prune_mlp:    if `False`, this is effectively a no-op
                      (`mlp_merge=False` skips the keep path everywhere).
                      Hiera uses SDPA which is already efficient, so we
                      don't implement a separate sparse-attention path.
        target_N:     only apply partial-MLP + early-cache at blocks
                      whose post-attn token count equals `target_N`.
                      Default 4096 = stage 2 of SAM2 Hiera at 1024×1024
                      input (a 64×64 grid). Set `None` to apply at all
                      resolutions (will likely hurt accuracy — stage 0/1
                      features feed the FpnNeck and aren't safe to drop).
        start_block:  only patch blocks with index >= `start_block`;
                      earlier blocks are left stock. Useful for ablations
                      like "patch only blocks after the first global"
                      (set to first_global_idx + 1). Default 0 = patch all.
        use_cute:     accepted for API compatibility with `profile_sam2.py`;
                      ignored — Hiera doesn't use FA2 cute kernels.
    """
    from sam2.modeling.backbones.hieradet import (
        MultiScaleBlock, MultiScaleAttention,
    )

    assert 0 < ratio <= 1.0, "ratio must be in (0, 1]"

    info = {
        "ratio":       ratio,
        "group_size":  group_size,
        "mlp_merge":   bool(prune_mlp),
        "perm_cache":  {},
        "target_N":    target_N,
    }
    trunk._tome_info = info

    # Reset cache at the start of every forward so different images
    # don't share perms.
    trunk._tome_hook_handles: List = []
    h = trunk.register_forward_pre_hook(_reset_perm_cache(info))
    trunk._tome_hook_handles.append(h)

    globals_set = set(trunk.global_att_blocks or ())
    n_patched_blk = 0
    n_skipped_blk = 0
    patched_globals = []
    for i, block in enumerate(trunk.blocks):
        if not isinstance(block, MultiScaleBlock):
            continue
        if i < start_block:
            n_skipped_blk += 1
            continue
        block._tome_info = info
        if not getattr(block, "_tome_orig_forward", None):
            block._tome_orig_forward = block.forward
        block.forward = types.MethodType(_patched_block_forward, block)

        attn = block.attn
        if isinstance(attn, MultiScaleAttention):
            attn._tome_info  = info
            attn._is_global  = (i in globals_set)
            if not getattr(attn, "_tome_orig_forward", None):
                attn._tome_orig_forward = attn.forward
            attn.forward = types.MethodType(_patched_attn_forward, attn)
        n_patched_blk += 1
        if i in globals_set:
            patched_globals.append(i)

    n_globals = len(patched_globals)
    n_local   = n_patched_blk - n_globals
    print(
        f"[sparsesam-sam2] patched  ratio={ratio}  gs={group_size}  "
        f"prune_mlp={prune_mlp}  start_block={start_block}  "
        f"blocks={n_patched_blk} (skipped={n_skipped_blk}, "
        f"globals={n_globals}, locals={n_local}; "
        f"patched_globals={patched_globals}; "
        f"all_global_att_blocks={sorted(globals_set)})"
    )


def remove_patch(trunk):
    """Restore original forwards. Inverse of `apply_patch`."""
    for h in getattr(trunk, "_tome_hook_handles", []):
        h.remove()
    if hasattr(trunk, "_tome_hook_handles"):
        del trunk._tome_hook_handles
    if hasattr(trunk, "_tome_info"):
        del trunk._tome_info

    for block in trunk.blocks:
        if hasattr(block, "_tome_orig_forward"):
            block.forward = block._tome_orig_forward
            del block._tome_orig_forward
        if hasattr(block, "_tome_info"):
            del block._tome_info

        attn = block.attn
        if hasattr(attn, "_tome_orig_forward"):
            attn.forward = attn._tome_orig_forward
            del attn._tome_orig_forward
        for k in ("_tome_info", "_is_global"):
            if hasattr(attn, k):
                delattr(attn, k)
