"""Cute (cutlass DSL) FA2 + RoPE kernel infrastructure for PE patches.

Provides the dense `flash_rope_attn` (full attention) and the block-sparse
`flash_rope_sparse_attn` (banded + keep-bar), plus the shared kernel cache,
compiled-shape cache, mask cache, perm cache, and `_make_A_mask` /
`_get_uniform_stride_perm` helpers used by both stage-compression patches.

Globals (`FlashAttentionForwardAmpereRoPE`, `from_dlpack`, `cuda_driver`,
`cutlass`) are populated lazily by `_ensure_cute_deps()` so this module is
cheap to import on CUDA-less hosts and from PE patches that may fall back
to stock SDPA.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_KERNEL_IMPORT_ERROR: Optional[Exception] = None
FlashAttentionForwardAmpereRoPE = None
from_dlpack = None
cuda_driver = None
cutlass = None


def _ensure_cute_deps():
    global FlashAttentionForwardAmpereRoPE, from_dlpack, cuda_driver, cutlass
    global _KERNEL_IMPORT_ERROR
    if FlashAttentionForwardAmpereRoPE is not None:
        return
    try:
        import cutlass as _cutlass
        from cutlass.cute.runtime import from_dlpack as _from_dlpack
        import cuda.bindings.driver as _cuda_driver
        from ..kernels.flash_attn import FlashAttentionForwardAmpereRoPE as _Kernel
    except Exception as e:
        _KERNEL_IMPORT_ERROR = e
        return
    cutlass = _cutlass
    from_dlpack = _from_dlpack
    cuda_driver = _cuda_driver
    FlashAttentionForwardAmpereRoPE = _Kernel


_KERNEL_CACHE: dict = {}      # (dtype, head_dim) -> (kernel, m_blk, n_blk)
_COMPILED_CACHE: dict = {}    # (dtype, head_dim, B, S, H, m_blk, n_blk) -> compiled
_MASK_CACHE: dict = {}        # (B, H, M_, N_, device) -> dense ones mask
_PERM_CACHE: dict = {}        # (S, ratio, gs, n_blk, has_cls, device) -> (perm, inv)
_SPARSE_MASK_CACHE: dict = {} # block-sparse mask cache

# 64x64 first: PE has small grids (≤32×32 patches), so small tiles give the
# block-sparse mask more granularity.
_BLOCK_CANDIDATES: Tuple[Tuple[int, int, int], ...] = (
    ( 64,  64, 128),
    ( 64, 128, 128),
    (128,  64, 128),
    (128, 128, 128),
)

# Defaults chosen by autoresearch sweep; only widen the kept attention region
# and matter most when num_n is small (queries see few keys).
_DIAG_BAND_WIDTH = 3
_KEEP_BAR_SCALE = 2.0


def _torch_dtype_to_cutlass(dtype: torch.dtype):
    return {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }.get(dtype, None)


def _get_kernel(dtype: torch.dtype, head_dim: int):
    key = (dtype, head_dim)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    cl_dtype = _torch_dtype_to_cutlass(dtype)
    if cl_dtype is None:
        print(f"[pe-cute] _get_kernel: no cutlass dtype for {dtype} — kernel unavailable.")
        _KERNEL_CACHE[key] = (None, None, None)
        return _KERNEL_CACHE[key]

    for m_blk, n_blk, n_thr in _BLOCK_CANDIDATES:
        if FlashAttentionForwardAmpereRoPE.can_implement(cl_dtype, head_dim, m_blk, n_blk, n_thr):
            kernel = FlashAttentionForwardAmpereRoPE(
                head_dim=head_dim, m_block_size=m_blk,
                n_block_size=n_blk, num_threads=n_thr,
            )
            _KERNEL_CACHE[key] = (kernel, m_blk, n_blk)
            return _KERNEL_CACHE[key]

    # No candidate worked — log SMEM/threading constraints for diagnosis.
    try:
        from cutlass.cute._mlir_helpers import _utils as _cute_utils  # type: ignore
        smem_cap = _cute_utils.get_smem_capacity_in_bytes("sm_80")
    except Exception:
        try:
            import cutlass.cute as _cute
            smem_cap = _cute.utils.get_smem_capacity_in_bytes("sm_80")
        except Exception:
            smem_cap = "?"
    head_dim_padded = (head_dim + 31) // 32 * 32
    print(f"[pe-cute] _get_kernel: NO candidate satisfies can_implement for "
          f"(dtype={dtype}, head_dim={head_dim}, head_dim_padded={head_dim_padded}). "
          f"SM80 SMEM budget = {smem_cap} bytes.")
    for m_blk, n_blk, n_thr in _BLOCK_CANDIDATES:
        smem = (m_blk * head_dim_padded * 2
                + n_blk * head_dim_padded * 2 * 2
                + m_blk * head_dim_padded * 2 * 2
                + n_blk * head_dim_padded * 2 * 2)
        thread_ok = (m_blk * 2) % n_thr == 0
        print(f"  candidate m={m_blk} n={n_blk} threads={n_thr}: "
              f"smem={smem}B  (m*2)%threads=={(m_blk*2)%n_thr} (thread_ok={thread_ok})")
    _KERNEL_CACHE[key] = (None, None, None)
    return _KERNEL_CACHE[key]


def _get_compiled(kernel, q, k, v, o, cos, sin, mask, scale, stream,
                  dtype, head_dim, B, S, H, m_blk, n_blk):
    key = (dtype, head_dim, B, S, H, m_blk, n_blk)
    fn = _COMPILED_CACHE.get(key)
    if fn is None:
        import cutlass.cute as cute
        fn = cute.compile(kernel, q, k, v, o, cos, sin, mask, scale, stream)
        _COMPILED_CACHE[key] = fn
    return fn


@torch.no_grad()
def _build_cos_sin(rope, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    freq = rope.freq
    if freq is None:
        raise RuntimeError(
            "Rope2D.freq is None — call rope.update_grid(...) first "
            "(usually done inside VisionTransformer.forward_features)."
        )
    cos = freq.cos().squeeze(0).contiguous().to(dtype)
    sin = freq.sin().squeeze(0).contiguous().to(dtype)
    return cos, sin


@torch.no_grad()
def _module_cached_cos_sin(self_attn: nn.Module,
                           dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-SelfAttention cache for full-grid cos/sin; invalidates on
    `rope.freq.data_ptr()` or dtype change."""
    freq = self_attn.rope.freq
    if freq is None:
        raise RuntimeError("Rope2D.freq is None — call rope.update_grid first.")
    key = (freq.data_ptr(), dtype, freq.shape[1])
    if getattr(self_attn, "_flash_cos_sin_key", None) == key:
        return self_attn._flash_cos, self_attn._flash_sin
    cos, sin = _build_cos_sin(self_attn.rope, dtype)
    self_attn._flash_cos = cos
    self_attn._flash_sin = sin
    self_attn._flash_cos_sin_key = key
    return cos, sin


def flash_rope_attn(self_attn: nn.Module, x: torch.Tensor,
                    cos: Optional[torch.Tensor] = None,
                    sin: Optional[torch.Tensor] = None,
                    block_mask: Optional[torch.Tensor] = None,
                    ) -> Optional[torch.Tensor]:
    """Run fused FA2 + RoPE cute kernel (dense). Returns None if the kernel
    can't be built for this (dtype, head_dim) — callers should fall back."""
    if x.dtype not in (torch.float16, torch.bfloat16):
        x = x.to(torch.float16)

    head_dim = self_attn.head_dim
    H        = self_attn.num_heads
    E        = self_attn.embed_dim

    kernel, m_blk, n_blk = _get_kernel(x.dtype, head_dim)
    if kernel is None:
        return None

    B, S, _ = x.shape

    proj = F.linear(x, self_attn.in_proj_weight, self_attn.in_proj_bias)
    proj = (proj.unflatten(-1, (3, E))
                 .unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous())
    q, k, v = proj[0], proj[1], proj[2]

    q = q.view(B, S, H, head_dim)
    k = k.view(B, S, H, head_dim)
    v = v.view(B, S, H, head_dim)
    o = torch.empty_like(q)

    if cos is None or sin is None:
        cos, sin = _module_cached_cos_sin(self_attn, x.dtype)
        cos = cos[:S].contiguous()
        sin = sin[:S].contiguous()

    if block_mask is None:
        M_ = (S + m_blk - 1) // m_blk
        N_ = (S + n_blk - 1) // n_blk
        mask_key = (B, H, M_, N_, str(x.device))
        block_mask = _MASK_CACHE.get(mask_key)
        if block_mask is None:
            block_mask = torch.ones((B, H, M_, N_), dtype=torch.int32, device=x.device)
            _MASK_CACHE[mask_key] = block_mask

    dtype_width = 16 if x.dtype in (torch.float16, torch.bfloat16) else 32
    def _cute_qkvo(t):
        return (from_dlpack(t, assumed_align=16)
                .mark_layout_dynamic(leading_dim=3)
                .mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(),
                                            divisibility=128 // dtype_width))
    q_c, k_c, v_c, o_c = _cute_qkvo(q), _cute_qkvo(k), _cute_qkvo(v), _cute_qkvo(o)
    cos_c = from_dlpack(cos, assumed_align=16)
    sin_c = from_dlpack(sin, assumed_align=16)
    mask_c = from_dlpack(block_mask, assumed_align=4)

    cu_stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    scale = float(self_attn.scale)

    compiled = _get_compiled(
        kernel, q_c, k_c, v_c, o_c, cos_c, sin_c, mask_c, scale, cu_stream,
        x.dtype, head_dim, B, S, H, m_blk, n_blk,
    )
    compiled(q_c, k_c, v_c, o_c, cos_c, sin_c, mask_c, scale, cu_stream)

    attn = o.view(B, S, H * head_dim)
    return F.linear(attn, self_attn.out_proj.weight, self_attn.out_proj.bias)


@torch.no_grad()
def _make_A_mask(B: int, H: int, T: int, ratio: float,
                 m_block: int, n_block: int,
                 band_width: int = _DIAG_BAND_WIDTH,
                 keep_bar_scale: float = _KEEP_BAR_SCALE,
                 device="cuda") -> torch.Tensor:
    """Banded-diagonal (width `band_width`) + vertical keep-bar A-shape mask."""
    num_m = math.ceil(T / m_block)
    num_n = math.ceil(T / n_block)
    t = torch.zeros(B, H, num_m, num_n, dtype=torch.int32, device=device)

    half = band_width // 2
    for k in range(-half, band_width - half):
        if k == 0:
            i = torch.arange(min(num_m, num_n), device=device)
            t[:, :, i, i] = 1
        else:
            i = torch.arange(max(0, -k), min(num_m, num_n - k), device=device)
            t[:, :, i, i + k] = 1

    n_keep_cols = int(ratio * num_n * keep_bar_scale)
    if n_keep_cols > 0:
        t[:, :, :, :max(n_keep_cols - band_width + 1, 0)] = 1
    return t


@torch.no_grad()
def _get_uniform_stride_perm(S: int, ratio: float, group_size: int,
                             n_block: int, has_cls: bool,
                             device) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_key = (S, ratio, group_size, n_block, has_cls, str(device))
    cached = _PERM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cls_off = 1 if has_cls else 0
    N = S - cls_off
    if N <= 0 or N % group_size != 0:
        perm = torch.arange(S, device=device)
        _PERM_CACHE[cache_key] = (perm, perm.clone())
        return _PERM_CACHE[cache_key]

    n_groups = N // group_size
    gs = group_size
    # MLP-budget-driven split: pick n_keep so the partial-MLP path spends
    # round(ratio·N) total MLP forwards, matching the SAM partial-MLP semantics:
    #     n_keep·gs + n_merge·1 = round(ratio·N)
    #     n_keep + n_merge      = n_groups
    # ⇒ n_keep = (round(ratio·N) − n_groups) / (gs − 1)
    # Valid when r ≥ 1/gs; below that, n_keep=0 (every group becomes a merge
    # group; only the first round(ratio·N) representatives run through MLP).
    K = max(1, round(ratio * N))
    if K >= n_groups:
        n_keep = max(0, (K - n_groups) // (gs - 1))
        n_keep = min(n_keep, n_groups)
    else:
        n_keep = 0
    n_merge = n_groups - n_keep

    # Group patch tokens by Z-order so each group of `gs` tokens is a
    # spatially-coherent z-curve block (2×2 for gs=4 on a square grid).
    from ..sparsesam.z_utils import get_z_order
    H_grid = int(math.isqrt(N))
    if H_grid * H_grid != N:
        raise ValueError(
            f"_get_uniform_stride_perm requires a square patch grid: "
            f"N={N} is not a perfect square (sqrt={math.sqrt(N):.3f})."
        )
    z_perm = get_z_order(H_grid, H_grid, device=device)
    group_raster = z_perm.view(n_groups, gs)

    if 0 < n_keep < n_groups:
        keep_idx = torch.round(
            torch.arange(n_keep, device=device, dtype=torch.float32)
            * (n_groups / n_keep)
        ).long().clamp_(0, n_groups - 1)
        m = torch.ones(n_groups, dtype=torch.bool, device=device)
        m[keep_idx] = False
        merge_idx = torch.nonzero(m, as_tuple=False).squeeze(1)
    elif n_keep == 0:
        keep_idx = torch.empty(0, dtype=torch.long, device=device)
        merge_idx = torch.arange(n_groups, device=device)
    else:
        keep_idx = torch.arange(n_groups, device=device)
        merge_idx = torch.empty(0, dtype=torch.long, device=device)

    keep_raster = group_raster[keep_idx].reshape(n_keep * gs)
    if n_merge > 0:
        merge_raster = group_raster[merge_idx].permute(1, 0).reshape(n_merge * gs)
        perm_sp = torch.cat([keep_raster, merge_raster], dim=0)
    else:
        perm_sp = keep_raster

    if cls_off:
        cls_perm = torch.arange(cls_off, device=device, dtype=perm_sp.dtype)
        perm = torch.cat([cls_perm, perm_sp + cls_off], dim=0)
    else:
        perm = perm_sp
    inv = torch.argsort(perm, dim=0)
    _PERM_CACHE[cache_key] = (perm, inv)
    return _PERM_CACHE[cache_key]


def flash_rope_sparse_attn(self_attn: nn.Module, x: torch.Tensor,
                           cos: torch.Tensor, sin: torch.Tensor,
                           block_mask: torch.Tensor,
                           perm: Optional[torch.Tensor] = None,
                           inv_perm: Optional[torch.Tensor] = None,
                           assume_permuted: bool = False,
                           ) -> Optional[torch.Tensor]:
    """Block-sparse FA2+RoPE. `cos`/`sin` must already be in kernel layout.
    `assume_permuted=True` skips per-call permutation (caller already did it
    upstream). Returns None if the cute kernel can't be built."""
    if x.dtype not in (torch.float16, torch.bfloat16):
        x = x.to(torch.float16)

    head_dim = self_attn.head_dim
    H        = self_attn.num_heads
    E        = self_attn.embed_dim

    kernel, m_blk, n_blk = _get_kernel(x.dtype, head_dim)
    if kernel is None or block_mask is None:
        return None

    B, S, _ = x.shape

    if assume_permuted or perm is None:
        x_in = x
    else:
        x_in = x.index_select(1, perm)

    proj = F.linear(x_in, self_attn.in_proj_weight, self_attn.in_proj_bias)
    proj = (proj.unflatten(-1, (3, E))
                 .unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous())
    q, k, v = proj[0], proj[1], proj[2]

    q = q.view(B, S, H, head_dim)
    k = k.view(B, S, H, head_dim)
    v = v.view(B, S, H, head_dim)
    o = torch.empty_like(q)

    dtype_width = 16 if x.dtype in (torch.float16, torch.bfloat16) else 32
    def _cute_qkvo(t):
        return (from_dlpack(t, assumed_align=16)
                .mark_layout_dynamic(leading_dim=3)
                .mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(),
                                            divisibility=128 // dtype_width))
    q_c, k_c, v_c, o_c = _cute_qkvo(q), _cute_qkvo(k), _cute_qkvo(v), _cute_qkvo(o)
    cos_c = from_dlpack(cos, assumed_align=16)
    sin_c = from_dlpack(sin, assumed_align=16)
    mask_c = from_dlpack(block_mask, assumed_align=4)

    cu_stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    scale = float(self_attn.scale)

    compiled = _get_compiled(
        kernel, q_c, k_c, v_c, o_c, cos_c, sin_c, mask_c, scale, cu_stream,
        x.dtype, head_dim, B, S, H, m_blk, n_blk,
    )
    compiled(q_c, k_c, v_c, o_c, cos_c, sin_c, mask_c, scale, cu_stream)

    attn = o.view(B, S, H * head_dim)
    if not assume_permuted and inv_perm is not None:
        attn = attn.index_select(1, inv_perm)
    return F.linear(attn, self_attn.out_proj.weight, self_attn.out_proj.bias)


def _ensure_block_mask(cache: dict, self_attn, x: torch.Tensor, sr: float,
                       dtype: Optional[torch.dtype] = None):
    """Build (or fetch) the cute block-sparse mask. Pass `dtype=weight_dtype`
    when running under autocast (x.dtype may be fp32 from LN upcast while
    the kernel is fp16/bf16)."""
    if cache.get("block_mask") is not None:
        return cache["block_mask"]
    kdtype = dtype if dtype is not None else x.dtype
    kernel, m_blk, n_blk = _get_kernel(kdtype, self_attn.head_dim)
    if kernel is None:
        return None
    B, S, _ = x.shape
    H = self_attn.num_heads
    mask_key = (B, H, S, sr, m_blk, n_blk,
                _DIAG_BAND_WIDTH, _KEEP_BAR_SCALE, str(x.device))
    block_mask = _SPARSE_MASK_CACHE.get(mask_key)
    if block_mask is None:
        block_mask = _make_A_mask(
            B, H, S, sr, m_blk, n_blk,
            band_width=_DIAG_BAND_WIDTH, keep_bar_scale=_KEEP_BAR_SCALE,
            device=x.device,
        )
        _SPARSE_MASK_CACHE[mask_key] = block_mask
    cache["block_mask"] = block_mask
    return block_mask
