import argparse
import math
from types import SimpleNamespace
from typing import Type, Callable

import cuda.bindings.driver as cuda
import cutlass.cute.testing as testing
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
import cutlass.pipeline as pipeline
import cutlass.utils as utils

"""
Flash Attention v2 forward pass for NVIDIA Ampere SM80 using the CUTE DSL.

Tensors: Q/K/V/O are (B, S, N, H) — batch, sequence, heads, head_dim.

Algorithm per CTA (one m-block, one (batch, head) pair):
  1. Load Q and the first K tile from GMEM→SMEM via CpAsync.
  2. For each n-block (right-to-left):
       a. S = Q * K^T  (tensor-core MMA, register pipeline).
       b. Apply seqlen padding mask on the first n-block.
       c. Online softmax update: rescale acc_O, accumulate row_max/row_sum.
       d. O += P * V  (tensor-core MMA, register pipeline).
       e. Prefetch next K tile.
  3. Normalize O by row_sum; store to GMEM.

Two positional-encoding variants share the same scaffolding:
  • FlashAttentionForwardAmpereRelPos — SAM's decomposed (h_rel, w_rel) bias
    added to S = Q·K^T per n-block. Also exposes a structured-mask launcher
    (`call_structured`) that skips the mask tensor for diagonal / A-shape masks.
  • FlashAttentionForwardAmpereRoPE  — 2D-axial RoPE applied in-place on
    Q (in prologue) and K (per n-block) before the S = Q·K^T MMA.

Constraints:
  - Only fp16 / bf16 supported.
  - Contiguous dim of each tensor must be ≥ 16 B aligned (head_dim % 8 == 0).
  - log-sum-exp (for training backward) is not computed.
  - m_block_size * 2 must be divisible by num_threads.
"""


class _FlashAttentionForwardAmpereBase:
    """Shared FA2 scaffolding. Subclasses inject positional encoding via:

        _pos_prepare_K(...)     — called at the top of each n-block, before S=Q·K^T.
                                  RoPE: rotate sK in place (skip on first, prologue did it).
                                  RelPos: no-op.
        _pos_modify_scores(...) — called between S=Q·K^T and softmax.
                                  RelPos: add (rRelH + rRelW) * inv_scale to acc_S.
                                  RoPE: no-op.

    Subclasses are responsible for their own `__call__` / `@cute.kernel kernel`
    (prologue + per-CTA orchestration), since the prologue shape and SMEM
    requirements differ. The shared parts live as `@cute.jit` helpers on this
    base class.
    """

    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
    ):
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    # ── shared launch-descriptor builder ───────────────────────────────────
    def _make_launch_descriptors(self):
        """Common SMEM layouts, GMEM copy atoms, and tiled-MMA for both variants.

        Returns (sQ_layout, sKV_layout, gmem_tiled_copy_QKV, gmem_tiled_copy_O,
                 tiled_mma).  Requires self._dtype to be set.
        """
        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._m_block_size, self._head_dim_padded), (0, 1)
        )
        sKV_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._n_block_size, self._head_dim_padded), (0, 1)
        )

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width

        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        vQKV_layout = cute.make_layout((1, async_copy_elems))

        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(atom_async_copy, tQKV_layout, vQKV_layout)
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tQKV_layout, vQKV_layout)

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )
        return sQ_layout, sKV_layout, gmem_tiled_copy_QKV, gmem_tiled_copy_O, tiled_mma

    # ── per-n-block compute (shared) ────────────────────────────────────────
    @cute.jit
    def compute_one_n_block(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        pos_state: SimpleNamespace,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
        no_mask: cutlass.Constexpr = False,
    ):
        """Process one n-block: optional K-prep, S=Q·K^T, optional score modify,
        online softmax, O+=P·V, prefetch next K."""
        acc_S = cute.make_rmem_tensor(
            mma_params.thr_mma.partition_shape_C((self._m_block_size, self._n_block_size)),
            cutlass.Float32,
        )

        # Drain previous K prefetch
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        if cutlass.const_expr(no_mask):
            block_enabled = True
        else:
            block_enabled = basic_params.m_block_mask[
                basic_params.batch_size, basic_params.num_head,
                basic_params.m_block, basic_params.n_block,
            ]

        # Hook: positional-encoding K prep (e.g., RoPE rotation). No-op for RelPos.
        self._pos_prepare_K(
            basic_params, pos_state, block_enabled,
            is_first_n_block=is_first_n_block,
        )

        # V load. For enabled blocks: full n_block_size rows.
        if block_enabled:
            if is_first_n_block:
                for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tVsV.shape[1])):
                    if cute.elem_less(
                        gmem_copy_params.tKVcKV[0, n, 0][1],
                        basic_params.mK.layout.shape[1],
                    ):
                        cute.copy(
                            gmem_copy_params.gmem_tiled_copy_QKV,
                            gmem_copy_params.tVgV[None, n, None, basic_params.n_block],
                            gmem_copy_params.tVsV[None, n, None],
                            pred=gmem_copy_params.tKVpKV[None, n, None],
                        )
                    else:
                        gmem_copy_params.tVsV[None, n, None].fill(0.0)
            else:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tVgV[None, None, None, basic_params.n_block],
                    gmem_copy_params.tVsV,
                    pred=gmem_copy_params.tKVpKV,
                )

        cute.arch.cp_async_commit_group()

        # S = Q * K^T  (register-pipelined).
        if block_enabled:
            acc_S.fill(0.0)
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, 0],
                smem_copy_params.tSrQ_copy_view[None, None, 0],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, 0],
                smem_copy_params.tSrK_copy_view[None, None, 0],
            )
            for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
                k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
                cute.copy(
                    smem_copy_params.smem_tiled_copy_Q,
                    smem_copy_params.tSsQ[None, None, k_next],
                    smem_copy_params.tSrQ_copy_view[None, None, k_next],
                )
                cute.copy(
                    smem_copy_params.smem_tiled_copy_K,
                    smem_copy_params.tSsK[None, None, k_next],
                    smem_copy_params.tSrK_copy_view[None, None, k_next],
                )
                cute.gemm(
                    mma_params.tiled_mma, acc_S,
                    mma_params.tSrQ[None, None, k],
                    mma_params.tSrK[None, None, k],
                    acc_S,
                )

        # Wait for V to arrive before reading it in the O GEMM
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # Prefetch K for the next n-block.
        if basic_params.n_block > 0:
            if cutlass.const_expr(no_mask):
                next_block_enabled = True
            else:
                next_block_enabled = basic_params.m_block_mask[
                    basic_params.batch_size, basic_params.num_head,
                    basic_params.m_block, basic_params.n_block - 1,
                ]
            if next_block_enabled:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tKgK[None, None, None, basic_params.n_block - 1],
                    gmem_copy_params.tKsK,
                    pred=gmem_copy_params.tKVpKV,
                )
            cute.arch.cp_async_commit_group()

        if block_enabled:
            # Hook: positional-encoding score modify (e.g., rel-pos bias). No-op for RoPE.
            self._pos_modify_scores(
                basic_params, mma_params, pos_state, acc_S,
                n_tile_size=self._n_block_size,
                n_tile_coord=basic_params.n_block,
            )
            self.softmax_rescale_O(
                basic_params, mma_params, softmax_params,
                acc_S, is_first_n_block, in_mask_steps,
                n_tile_size=self._n_block_size,
                n_tile_coord=basic_params.n_block,
            )
            # O += P * V
            rP = cute.make_fragment_like(acc_S, self._dtype)
            rP.store(acc_S.load().to(self._dtype))
            rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
            rP_mma_view = cute.make_layout(
                (
                    (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                    rP_layout_divided.shape[1],
                    rP_layout_divided.shape[2][1],
                ),
                stride=(
                    (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                    rP_layout_divided.stride[1],
                    rP_layout_divided.stride[2][1],
                ),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, 0],
                smem_copy_params.tOrVt_copy_view[None, None, 0],
            )
            for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                k_next = (k + 1) % cute.size(tOrS.shape[2])
                cute.copy(
                    smem_copy_params.smem_tiled_copy_V,
                    smem_copy_params.tOsVt[None, None, k_next],
                    smem_copy_params.tOrVt_copy_view[None, None, k_next],
                )
                cute.gemm(
                    mma_params.tiled_mma, mma_params.acc_O,
                    tOrS[None, None, k],
                    mma_params.tOrVt[None, None, k],
                    mma_params.acc_O,
                )

    # Default positional-encoding hooks — no-ops. Subclasses override.
    @cute.jit
    def _pos_prepare_K(
        self,
        basic_params: SimpleNamespace,
        pos_state: SimpleNamespace,
        block_enabled,
        is_first_n_block: cutlass.Constexpr,
    ):
        pass

    @cute.jit
    def _pos_modify_scores(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        pos_state: SimpleNamespace,
        acc_S: cute.Tensor,
        n_tile_size: cutlass.Constexpr,
        n_tile_coord: cutlass.Int32,
    ):
        pass

    @cute.jit
    def softmax_rescale_O(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        acc_S: cute.Tensor,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
        n_tile_size: cutlass.Constexpr,
        n_tile_coord: cutlass.Int32,
    ):
        """Apply online softmax to acc_S and rescale acc_O.

        Uses exp2(x * log2e - max * log2e) to fuse scale into the exponent.
        Rescales acc_O by exp(prev_max - cur_max) to maintain the running sum invariant.
        """
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)

        row_max_prev = None
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(softmax_params.row_max, cutlass.Float32)
            cute.basic_copy(softmax_params.row_max, row_max_prev)

        tScS_mn = None
        if cutlass.const_expr(in_mask_steps):
            mcS = cute.make_identity_tensor((
                basic_params.mQ.shape[0], basic_params.mQ.shape[1],
                basic_params.mQ.shape[2], basic_params.mK.shape[1],
            ))
            cS = cute.local_tile(
                mcS[basic_params.batch_size, None, basic_params.num_head, None],
                (self._m_block_size, n_tile_size),
                (basic_params.m_block, n_tile_coord),
            )
            tScS_mn = self._make_acc_tensor_mn_view(mma_params.thr_mma.partition_C(cS))

        for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
            if cutlass.const_expr(in_mask_steps):
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    if cute.elem_less(basic_params.mK.shape[1], tScS_mn[0, c][3] + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf

            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur_row = self._threadquad_reduce_max(
                acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            )

            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            else:
                # Clamp to 0 when all entries are -inf to avoid exp2(NaN)
                row_max_cur_row = (
                    0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
                )

            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_params.softmax_scale_log2
                - row_max_cur_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)

            if cutlass.const_expr(not is_first_n_block):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * softmax_params.softmax_scale_log2
                    - row_max_cur_row * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp

            softmax_params.row_max[r] = row_max_cur_row
            softmax_params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize_softmax(self, acc_O: cute.Tensor, row_sum: cute.Tensor):
        """Divide each output row by its softmax normalizer. Zero rows → zero output."""
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            scale = 1.0 if is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    @cute.jit
    def epilogue_store_O(
        self,
        mO: cute.Tensor,
        acc_O: cute.Tensor,
        row_sum: cute.Tensor,
        sQ: cute.Tensor,
        sO_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        gmem_tiled_copy_O: cute.TiledCopy,
        tidx,
        batch_size,
        num_head,
        m_block,
    ):
        """Normalize O, cast to output dtype, smem stage, then predicated GMEM store.

        Reuses the sQ SMEM buffer for sO since the layouts are identical.
        """
        self.normalize_softmax(acc_O, row_sum)
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))

        sO = cute.make_tensor(sQ.iterator, sO_layout)

        smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype)
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        gO = cute.local_tile(
            mO[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)

        # Wait for all SMEM stores before vectorized SMEM→RMEM→GMEM copy
        self.cta_sync_barrier.arrive_and_wait()
        cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

        mcO = cute.make_identity_tensor(mO.layout.shape)
        cO = cute.local_tile(
            mcO[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        tOcO = gmem_thr_copy_O.partition_D(cO)
        tOpO = cute.make_rmem_tensor(
            cute.make_layout(
                (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                stride=(tOgO.shape[2], 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
            for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                tOpO[rest_v, 0, rest_n] = cute.elem_less(
                    tOcO[(0, rest_v), 0, rest_n][3], mO.layout.shape[3]
                )
        for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
            if cute.elem_less(tOcO[0, rest_m, 0][1], mO.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """Reinterpret the MMA accumulator layout as a flat (M, N) view."""
        s = cute.make_layout(acc.layout.shape)
        mn_layout = cute.make_layout(
            ((s.shape[0][1], s.shape[1]), (s.shape[0][0], s.shape[2])),
            stride=((s.stride[0][1], s.stride[1]), (s.stride[0][0], s.stride[2])),
        )
        return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn_layout))

    def _threadquad_reduce(self, val: cutlass.Float32, op: Callable) -> cutlass.Float32:
        """Two-step butterfly reduction within a 4-thread quad (offsets 2 then 1)."""
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: x + y)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 1: SAM-style decomposed relative-position bias
# ─────────────────────────────────────────────────────────────────────────────

class FlashAttentionForwardAmpereRelPos(_FlashAttentionForwardAmpereBase):
    """FA2 + SAM's decomposed (h_rel, w_rel) relative-position bias.

    The bias is loaded once per m-block into SMEM (sRelH / sRelW), staged into
    per-row RMEM tensors (rRelH / rRelW), then added to S = Q·K^T per n-block
    via _pos_modify_scores. Also exposes `call_structured` for structured masks
    (diagonal / A-shape) which skip the mask tensor entirely.
    """

    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
        win_shape: int = 64,
    ):
        super().__init__(head_dim, m_block_size, n_block_size, num_threads)
        self._win_shape = win_shape

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, n_block_size, num_threads, win_shape=0) -> bool:
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        # SMEM: Q + 2·KV + sRelH + sRelW
        smem_usage = (m_block_size * head_dim + n_block_size * head_dim * 2) * 2
        smem_usage += m_block_size * win_shape * 2 * 2
        if smem_usage > utils.get_smem_capacity_in_bytes("sm_80"):
            return False
        if (m_block_size * 2) % num_threads != 0:
            return False
        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        m_rel_H: cute.Tensor,
        m_rel_W: cute.Tensor,
        m_perm_Q: cute.Tensor,
        m_perm_K: cute.Tensor,
        m_block_mask: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)
        ):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(
            not (mQ.element_type == cutlass.Float16 or mQ.element_type == cutlass.BFloat16)
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: Type[cutlass.Numeric] = mQ.element_type

        sQ_layout, sKV_layout, gmem_tiled_copy_QKV, gmem_tiled_copy_O, tiled_mma = \
            self._make_launch_descriptors()
        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sRelH: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._m_block_size * self._win_shape], 128
            ]
            sRelW: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._m_block_size * self._win_shape], 128
            ]

        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        softmax_scale_log2 = softmax_scale * 1.4426950408889634074

        self.kernel(
            mQ, mK, mV, mO,
            m_rel_H, m_rel_W,
            m_perm_Q, m_perm_K,
            m_block_mask,
            softmax_scale_log2,
            sQ_layout, sKV_layout, sO_layout,
            gmem_tiled_copy_QKV, gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        m_rel_H: cute.Tensor,
        m_rel_W: cute.Tensor,
        m_perm_Q: cute.Tensor,
        m_perm_K: cute.Tensor,
        m_block_mask: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, num_head = cute.arch.block_idx()

        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        n_block = n_block_max - 1

        gQ = cute.local_tile(
            mQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        gK = cute.local_tile(
            mK[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (None, 0),
        )
        gV = cute.local_tile(
            mV[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (None, 0),
        )

        # Slice per-batch 1D views of the perm tensors so the kernel only needs
        # 1D indexing (CUTE DSL does not support [batch, idx] on plain tensors).
        perm_Q_batch = m_perm_Q[batch_size, None]
        perm_K_batch = m_perm_K[batch_size, None]

        # inv_softmax_scale: so that (acc_S + rel_pos/scale) * scale = acc_S*scale + rel_pos
        inv_softmax_scale = 1.4426950408889634074 / softmax_scale_log2

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)

        # Relative position bias SMEM: (m_block_size, win_shape), row-major
        sRel_layout = cute.make_layout(
            (self._m_block_size, self._win_shape), stride=(self._win_shape, 1)
        )
        sRelH = storage.sRelH.get_tensor(sRel_layout)
        sRelW = storage.sRelW.get_tensor(sRel_layout)

        # Transposed V view (head_dim, n_block_size) for O MMA
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        # Per-thread GMEM copy partitions
        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)

        # MMA register fragments and accumulator
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))

        # Small-N tensors (first-16-token K/V): currently unused by
        # compute_one_n_block but kept for potential masked-block fast path.
        sK_small = cute.local_tile(sK, (16, self._head_dim_padded), (0, 0))
        sVt_small = cute.local_tile(sVt, (self._head_dim_padded, 16), (0, 0))
        tSrK_small = thr_mma.make_fragment_B(thr_mma.partition_B(sK_small))
        tOrVt_small = thr_mma.make_fragment_B(thr_mma.partition_B(sVt_small))

        acc_O = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_block_size, self._head_dim_padded)),
            cutlass.Float32,
        )
        acc_O.fill(0.0)
        acc_S_small = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_block_size, 16)),
            cutlass.Float32,
        )

        # SMEM copy atoms: ldmatrix for Q/K (normal), ldmatrix.T for V
        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # Small-N SMEM→RMEM partitions (matched to the dead small-N fragments above)
        tSsK_small = smem_thr_copy_K.partition_S(sK_small)
        tSrK_small_copy_view = smem_thr_copy_K.retile(tSrK_small)
        tOsVt_small = smem_thr_copy_V.partition_S(sVt_small)
        tOrVt_small_copy_view = smem_thr_copy_V.retile(tOrVt_small)

        # Predicate tensors: mark valid head_dim indices (seqlen bounds handled per tile)
        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        mcKV = cute.make_identity_tensor(mK.layout.shape)
        cQ = cute.local_tile(
            mcQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        cKV = cute.local_tile(
            mcKV[batch_size, None, num_head, None],
            (self._n_block_size, self._head_dim_padded),
            (n_block, 0),
        )
        tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)

        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (tQsQ.shape[0][1], cute.size(tQsQ, mode=[1]), cute.size(tQsQ, mode=[2])),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        tKVpKV = cute.make_rmem_tensor(
            cute.make_layout(
                (tKsK.shape[0][1], cute.size(tKsK, mode=[1]), cute.size(tKsK, mode=[2])),
                stride=(cute.size(tKsK, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                    tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3]
                )

        # Online softmax state
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # Pre-allocate rRelH/rRelW and bundle pos state. Shape must be
        # identical in both branches of any dynamic `if` (DSL requirement).
        acc_O_mn_ref = self._make_acc_tensor_mn_view(acc_O)
        num_r = cute.size(acc_O_mn_ref.shape[0])
        mcS_ref = cute.make_identity_tensor((mQ.shape[0], mQ.shape[1], mQ.shape[2], mK.shape[1]))
        cS_ref = cute.local_tile(
            mcS_ref[batch_size, None, num_head, None],
            (self._m_block_size, self._n_block_size),
            (m_block, 0),
        )
        tScS_ref_mn = self._make_acc_tensor_mn_view(thr_mma.partition_C(cS_ref))

        rRelH = cute.make_rmem_tensor((num_r, self._win_shape), cutlass.Float32)
        rRelW = cute.make_rmem_tensor((num_r, self._win_shape), cutlass.Float32)
        rRelH.fill(0.0)
        rRelW.fill(0.0)

        pos_state = SimpleNamespace(
            m_rel_H=m_rel_H, m_rel_W=m_rel_W,
            perm_Q=perm_Q_batch, perm_K=perm_K_batch,
            inv_softmax_scale=inv_softmax_scale,
            sRelH=sRelH, sRelW=sRelW,
            rRelH=rRelH, rRelW=rRelW,
        )

        basic_params = SimpleNamespace(
            m_block=m_block, n_block=n_block,
            mQ=mQ, mK=mK,
            batch_size=batch_size, num_head=num_head,
            m_block_mask=m_block_mask,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma, tiled_mma=tiled_mma,
            tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt, acc_O=acc_O,
            tSrK_small=tSrK_small, tOrVt_small=tOrVt_small,
            acc_S_small=acc_S_small,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
            tKVcKV=tKVcKV,
            tKgK=tKgK, tKsK=tKsK,
            tVgV=tVgV, tVsV=tVsV,
            tKVpKV=tKVpKV,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ, tSrQ_copy_view=tSrQ_copy_view,
            tSsK=tSsK, tSrK_copy_view=tSrK_copy_view,
            tOsVt=tOsVt, tOrVt_copy_view=tOrVt_copy_view,
            tSsK_small=tSsK_small, tSrK_small_copy_view=tSrK_small_copy_view,
            tOsVt_small=tOsVt_small, tOrVt_small_copy_view=tOrVt_small_copy_view,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max, row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )

        # Prologue: prefetch Q and the first K tile into SMEM.
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None],
                )
            else:
                tQsQ[None, m, None].fill(0)

        last_block_enabled = m_block_mask[batch_size, num_head, m_block, n_block]
        if last_block_enabled:
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if cute.elem_less(tKVcKV[0, n, 0][1], mK.layout.shape[1]):
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tKgK[None, n, None, n_block],
                        tKsK[None, n, None],
                        pred=tKVpKV[None, n, None],
                    )
                else:
                    tKsK[None, n, None].fill(0)

        cute.arch.cp_async_commit_group()

        # Cooperative GMEM→SMEM load of rel position bias for this m_block.
        # q_range is fixed across all n_blocks so we load once here.
        n_rel_elems = self._m_block_size * self._win_shape
        for j in cutlass.range_constexpr(cute.ceil_div(n_rel_elems, self._num_threads)):
            flat_idx = tidx + j * self._num_threads
            if cute.elem_less(flat_idx, n_rel_elems):
                q_local = flat_idx // self._win_shape
                k_pos = flat_idx % self._win_shape
                q_global_perm = m_block * self._m_block_size + q_local
                if cute.elem_less(q_global_perm, mQ.shape[1]):
                    q_global_orig = perm_Q_batch[q_global_perm]
                    sRelH[q_local, k_pos] = m_rel_H[batch_size, q_global_orig, num_head, k_pos].to(self._dtype)
                    sRelW[q_local, k_pos] = m_rel_W[batch_size, q_global_orig, num_head, k_pos].to(self._dtype)

        # Sync so every thread's SMEM writes are visible before the RMEM preload.
        self.cta_sync_barrier.arrive_and_wait()

        for r in cutlass.range_constexpr(num_r):
            q_idx = tScS_ref_mn[r, 0][1]
            q_local = q_idx - m_block * self._m_block_size
            for k in cutlass.range_constexpr(self._win_shape):
                rRelH[r, k] = sRelH[q_local, k].to(cutlass.Float32)
                rRelW[r, k] = sRelW[q_local, k].to(cutlass.Float32)

        # First n-block: needs seqlen_k padding-mask handling
        basic_params.n_block = n_block_max - 1
        self.compute_one_n_block(
            basic_params, mma_params, gmem_copy_params, smem_copy_params,
            softmax_params, pos_state,
            is_first_n_block=True, in_mask_steps=True,
        )

        # Remaining n-blocks
        for n_tile in range(1, n_block_max, 1):
            basic_params.n_block = n_block_max - n_tile - 1
            self.compute_one_n_block(
                basic_params, mma_params, gmem_copy_params, smem_copy_params,
                softmax_params, pos_state,
                is_first_n_block=False, in_mask_steps=False,
            )

        self.epilogue_store_O(
            mO, acc_O, row_sum,
            sQ, sO_layout, tiled_mma,
            gmem_tiled_copy_O, tidx,
            batch_size, num_head, m_block,
        )

    # ── positional-encoding hook: score-level bias ────────────────────────
    @cute.jit
    def _pos_modify_scores(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        pos_state: SimpleNamespace,
        acc_S: cute.Tensor,
        n_tile_size: cutlass.Constexpr,
        n_tile_coord: cutlass.Int32,
    ):
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)

        mcS = cute.make_identity_tensor((
            basic_params.mQ.shape[0], basic_params.mQ.shape[1],
            basic_params.mQ.shape[2], basic_params.mK.shape[1],
        ))
        cS = cute.local_tile(
            mcS[basic_params.batch_size, None, basic_params.num_head, None],
            (self._m_block_size, n_tile_size),
            (basic_params.m_block, n_tile_coord),
        )
        tScS_mn = self._make_acc_tensor_mn_view(mma_params.thr_mma.partition_C(cS))
        for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
            k_idx = tScS_mn[0, c][3]
            if cute.elem_less(k_idx, basic_params.mK.shape[1]):
                k_idx_orig = pos_state.perm_K[k_idx]
                k_row = k_idx_orig // self._win_shape
                k_col = k_idx_orig % self._win_shape
                for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                    acc_S_mn[r, c] = acc_S_mn[r, c] + (
                        pos_state.rRelH[r, k_row]
                        + pos_state.rRelW[r, k_col]
                    ) * pos_state.inv_softmax_scale

    # ─────────────────────────────────────────────────────────────────────
    # Structured-mask variants: no mask tensor needed.
    #
    #   DIAGONAL  (n_init_blocks=0): each CTA processes only n_block == m_block.
    #   A-SHAPE   (n_init_blocks>0): n_block < n_init_blocks  ∪  n_block == m_block.
    #
    # Benefits vs the GENERAL kernel:
    #   • No (B,H,nm,nn) mask tensor → zero mask GMEM reads.
    #   • Loop iterates n_init_blocks+1 times instead of n_block_max times.
    #   • All processed blocks are always "enabled" → no_mask=True eliminates
    #     dead branches inside compute_one_n_block.
    # ─────────────────────────────────────────────────────────────────────

    @cute.jit
    def call_structured(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        m_rel_H: cute.Tensor,
        m_rel_W: cute.Tensor,
        m_perm_Q: cute.Tensor,
        m_perm_K: cute.Tensor,
        n_init_blocks: cutlass.Int32,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)
        ):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(
            not (mQ.element_type == cutlass.Float16 or mQ.element_type == cutlass.BFloat16)
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: Type[cutlass.Numeric] = mQ.element_type

        sQ_layout, sKV_layout, gmem_tiled_copy_QKV, gmem_tiled_copy_O, tiled_mma = \
            self._make_launch_descriptors()
        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]
            sRelH: cute.struct.Align[cute.struct.MemRange[self._dtype, self._m_block_size * self._win_shape], 128]
            sRelW: cute.struct.Align[cute.struct.MemRange[self._dtype, self._m_block_size * self._win_shape], 128]

        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        softmax_scale_log2 = softmax_scale * 1.4426950408889634074

        self.kernel_structured(
            mQ, mK, mV, mO,
            m_rel_H, m_rel_W,
            m_perm_Q, m_perm_K,
            n_init_blocks,
            softmax_scale_log2,
            sQ_layout, sKV_layout, sO_layout,
            gmem_tiled_copy_QKV, gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel_structured(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        m_rel_H: cute.Tensor,
        m_rel_W: cute.Tensor,
        m_perm_Q: cute.Tensor,
        m_perm_K: cute.Tensor,
        n_init_blocks: cutlass.Int32,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, num_head = cute.arch.block_idx()
        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)

        gQ = cute.local_tile(mQ[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
        gK = cute.local_tile(mK[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (None, 0))
        gV = cute.local_tile(mV[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (None, 0))

        perm_Q_batch = m_perm_Q[batch_size, None]
        perm_K_batch = m_perm_K[batch_size, None]
        inv_softmax_scale = 1.4426950408889634074 / softmax_scale_log2

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)

        sRel_layout = cute.make_layout((self._m_block_size, self._win_shape), stride=(self._win_shape, 1))
        sRelH = storage.sRelH.get_tensor(sRel_layout)
        sRelW = storage.sRelW.get_tensor(sRel_layout)

        sVt = cute.composition(sV, cute.make_layout((self._head_dim_padded, self._n_block_size), stride=(self._n_block_size, 1)))

        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))

        # Small-N tensors: dead in structured mode (no masked blocks); kept
        # for symmetry with the dense kernel.
        sK_small = cute.local_tile(sK, (16, self._head_dim_padded), (0, 0))
        sVt_small = cute.local_tile(sVt, (self._head_dim_padded, 16), (0, 0))
        tSrK_small = thr_mma.make_fragment_B(thr_mma.partition_B(sK_small))
        tOrVt_small = thr_mma.make_fragment_B(thr_mma.partition_B(sVt_small))

        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((self._m_block_size, self._head_dim_padded)), cutlass.Float32)
        acc_O.fill(0.0)
        acc_S_small = cute.make_rmem_tensor(thr_mma.partition_shape_C((self._m_block_size, 16)), cutlass.Float32)

        smem_copy_atom_Q = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
        smem_copy_atom_K = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
        smem_copy_atom_V = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype)
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        tSsK_small = smem_thr_copy_K.partition_S(sK_small)
        tSrK_small_copy_view = smem_thr_copy_K.retile(tSrK_small)
        tOsVt_small = smem_thr_copy_V.partition_S(sVt_small)
        tOrVt_small_copy_view = smem_thr_copy_V.retile(tOrVt_small)

        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        mcKV = cute.make_identity_tensor(mK.layout.shape)
        cQ = cute.local_tile(mcQ[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
        cKV = cute.local_tile(mcKV[batch_size, None, num_head, None], (self._n_block_size, self._head_dim_padded), (m_block, 0))
        tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)

        tQpQ = cute.make_rmem_tensor(cute.make_layout((tQsQ.shape[0][1], cute.size(tQsQ, mode=[1]), cute.size(tQsQ, mode=[2])), stride=(cute.size(tQsQ, mode=[2]), 0, 1)), cutlass.Boolean)
        tKVpKV = cute.make_rmem_tensor(cute.make_layout((tKsK.shape[0][1], cute.size(tKsK, mode=[1]), cute.size(tKsK, mode=[2])), stride=(cute.size(tKsK, mode=[2]), 0, 1)), cutlass.Boolean)
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3])
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3])

        row_max = cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        acc_O_mn_ref = self._make_acc_tensor_mn_view(acc_O)
        num_r = cute.size(acc_O_mn_ref.shape[0])
        mcS_ref = cute.make_identity_tensor((mQ.shape[0], mQ.shape[1], mQ.shape[2], mK.shape[1]))
        cS_ref = cute.local_tile(mcS_ref[batch_size, None, num_head, None], (self._m_block_size, self._n_block_size), (m_block, 0))
        tScS_ref_mn = self._make_acc_tensor_mn_view(thr_mma.partition_C(cS_ref))

        rRelH = cute.make_rmem_tensor((num_r, self._win_shape), cutlass.Float32)
        rRelW = cute.make_rmem_tensor((num_r, self._win_shape), cutlass.Float32)
        rRelH.fill(0.0)
        rRelW.fill(0.0)

        pos_state = SimpleNamespace(
            m_rel_H=m_rel_H, m_rel_W=m_rel_W,
            perm_Q=perm_Q_batch, perm_K=perm_K_batch,
            inv_softmax_scale=inv_softmax_scale,
            sRelH=sRelH, sRelW=sRelW,
            rRelH=rRelH, rRelW=rRelW,
        )

        basic_params = SimpleNamespace(
            m_block=m_block, n_block=m_block,
            mQ=mQ, mK=mK,
            batch_size=batch_size, num_head=num_head,
            m_block_mask=None,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma, tiled_mma=tiled_mma,
            tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt, acc_O=acc_O,
            tSrK_small=tSrK_small, tOrVt_small=tOrVt_small,
            acc_S_small=acc_S_small,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
            tKVcKV=tKVcKV,
            tKgK=tKgK, tKsK=tKsK,
            tVgV=tVgV, tVsV=tVsV,
            tKVpKV=tKVpKV,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ, tSrQ_copy_view=tSrQ_copy_view,
            tSsK=tSsK, tSrK_copy_view=tSrK_copy_view,
            tOsVt=tOsVt, tOrVt_copy_view=tOrVt_copy_view,
            tSsK_small=tSsK_small, tSrK_small_copy_view=tSrK_small_copy_view,
            tOsVt_small=tOsVt_small, tOrVt_small_copy_view=tOrVt_small_copy_view,
        )
        softmax_params = SimpleNamespace(row_max=row_max, row_sum=row_sum, softmax_scale_log2=softmax_scale_log2)

        # ── Prologue: load Q and K[m_block] (diagonal block, always first active) ──
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                cute.copy(gmem_tiled_copy_QKV, tQgQ[None, m, None], tQsQ[None, m, None], pred=tQpQ[None, m, None])
            else:
                tQsQ[None, m, None].fill(0)
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            if cute.elem_less(tKVcKV[0, n, 0][1], mK.layout.shape[1]):
                cute.copy(gmem_tiled_copy_QKV, tKgK[None, n, None, m_block], tKsK[None, n, None], pred=tKVpKV[None, n, None])
            else:
                tKsK[None, n, None].fill(0)
        cute.arch.cp_async_commit_group()

        n_rel_elems = self._m_block_size * self._win_shape
        for j in cutlass.range_constexpr(cute.ceil_div(n_rel_elems, self._num_threads)):
            flat_idx = tidx + j * self._num_threads
            if cute.elem_less(flat_idx, n_rel_elems):
                q_local = flat_idx // self._win_shape
                k_pos = flat_idx % self._win_shape
                q_global_perm = m_block * self._m_block_size + q_local
                if cute.elem_less(q_global_perm, mQ.shape[1]):
                    q_global_orig = perm_Q_batch[q_global_perm]
                    sRelH[q_local, k_pos] = m_rel_H[batch_size, q_global_orig, num_head, k_pos].to(self._dtype)
                    sRelW[q_local, k_pos] = m_rel_W[batch_size, q_global_orig, num_head, k_pos].to(self._dtype)
        self.cta_sync_barrier.arrive_and_wait()

        for r in cutlass.range_constexpr(num_r):
            q_idx = tScS_ref_mn[r, 0][1]
            q_local = q_idx - m_block * self._m_block_size
            for k in cutlass.range_constexpr(self._win_shape):
                rRelH[r, k] = sRelH[q_local, k].to(cutlass.Float32)
                rRelW[r, k] = sRelW[q_local, k].to(cutlass.Float32)

        # ── First active block: the diagonal (m_block) ────────────────────
        basic_params.n_block = m_block
        self.compute_one_n_block(
            basic_params, mma_params, gmem_copy_params, smem_copy_params,
            softmax_params, pos_state,
            is_first_n_block=True, in_mask_steps=True, no_mask=True,
        )

        # ── Initial-token blocks: [n_init_blocks-1 .. 0], skipping m_block ──
        for n in range(n_init_blocks):
            cur_n = n_init_blocks - 1 - n
            if cur_n != m_block:
                basic_params.n_block = cur_n
                self.compute_one_n_block(
                    basic_params, mma_params, gmem_copy_params, smem_copy_params,
                    softmax_params, pos_state,
                    is_first_n_block=False, in_mask_steps=False, no_mask=True,
                )

        self.epilogue_store_O(
            mO, acc_O, row_sum,
            sQ, sO_layout, tiled_mma,
            gmem_tiled_copy_O, tidx,
            batch_size, num_head, m_block,
        )


# Back-compat alias: the SAM patches import `FlashAttentionForwardAmpere`.
FlashAttentionForwardAmpere = FlashAttentionForwardAmpereRelPos


# ─────────────────────────────────────────────────────────────────────────────
# Variant 2: 2D-axial RoPE applied in-kernel
# ─────────────────────────────────────────────────────────────────────────────

class FlashAttentionForwardAmpereRoPE(_FlashAttentionForwardAmpereBase):
    """FA2 + fused 2D-axial RoPE. No rel-pos bias, no token permutations.

    Q is rotated in SMEM once in the prologue; each K tile is rotated in SMEM
    at the top of its n-block (or in the prologue for the first n-block).
    """

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, n_block_size, num_threads) -> bool:
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim % 4 != 0:
            return False  # axial RoPE requires D % 4 == 0
        if num_threads % 32 != 0:
            return False
        head_dim_padded = (head_dim + 31) // 32 * 32
        # SMEM: Q + 2·KV (Q overlaps with O on epilogue) + cos/sin for Q + cos/sin for K
        smem_q = m_block_size * head_dim_padded * 2
        smem_kv = n_block_size * head_dim_padded * 2 * 2
        smem_cs_q = m_block_size * head_dim_padded * 2 * 2
        smem_cs_k = n_block_size * head_dim_padded * 2 * 2
        if smem_q + smem_kv + smem_cs_q + smem_cs_k > utils.get_smem_capacity_in_bytes("sm_80"):
            return False
        if (m_block_size * 2) % num_threads != 0:
            return False
        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        m_cos: cute.Tensor,
        m_sin: cute.Tensor,
        m_block_mask: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)
        ):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(
            not (mQ.element_type == cutlass.Float16 or mQ.element_type == cutlass.BFloat16)
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: Type[cutlass.Numeric] = mQ.element_type

        sQ_layout, sKV_layout, gmem_tiled_copy_QKV, gmem_tiled_copy_O, tiled_mma = \
            self._make_launch_descriptors()
        sO_layout = sQ_layout

        # cos/sin SMEM uses simple row-major layout — read/written by element,
        # not via ldmatrix / cp.async-128 in the rotation pass.
        sCosQ_layout = cute.make_layout(
            (self._m_block_size, self._head_dim_padded),
            stride=(self._head_dim_padded, 1),
        )
        sCosK_layout = cute.make_layout(
            (self._n_block_size, self._head_dim_padded),
            stride=(self._head_dim_padded, 1),
        )

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sCosQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._m_block_size * self._head_dim_padded], 128
            ]
            sSinQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._m_block_size * self._head_dim_padded], 128
            ]
            sCosK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._n_block_size * self._head_dim_padded], 128
            ]
            sSinK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._n_block_size * self._head_dim_padded], 128
            ]

        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        softmax_scale_log2 = softmax_scale * 1.4426950408889634074

        self.kernel(
            mQ, mK, mV, mO, m_cos, m_sin, m_block_mask,
            softmax_scale_log2,
            sQ_layout, sKV_layout, sO_layout,
            sCosQ_layout, sCosK_layout,
            gmem_tiled_copy_QKV, gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        m_cos: cute.Tensor,
        m_sin: cute.Tensor,
        m_block_mask: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sCosQ_layout: cute.Layout,
        sCosK_layout: cute.Layout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, num_head = cute.arch.block_idx()

        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        n_block = n_block_max - 1

        gQ = cute.local_tile(
            mQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        gK = cute.local_tile(
            mK[batch_size, None, num_head, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )
        gV = cute.local_tile(
            mV[batch_size, None, num_head, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)
        sCosQ = storage.sCosQ.get_tensor(sCosQ_layout)
        sSinQ = storage.sSinQ.get_tensor(sCosQ_layout)
        sCosK = storage.sCosK.get_tensor(sCosK_layout)
        sSinK = storage.sSinK.get_tensor(sCosK_layout)

        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))

        acc_O = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_block_size, self._head_dim_padded)),
            cutlass.Float32,
        )
        acc_O.fill(0.0)

        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        mcKV = cute.make_identity_tensor(mK.layout.shape)
        cQ = cute.local_tile(
            mcQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        cKV = cute.local_tile(
            mcKV[batch_size, None, num_head, None],
            (self._n_block_size, self._head_dim_padded),
            (n_block, 0),
        )
        tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)

        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (tQsQ.shape[0][1], cute.size(tQsQ, mode=[1]), cute.size(tQsQ, mode=[2])),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        tKVpKV = cute.make_rmem_tensor(
            cute.make_layout(
                (tKsK.shape[0][1], cute.size(tKsK, mode=[1]), cute.size(tKsK, mode=[2])),
                stride=(cute.size(tKsK, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                    tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3]
                )

        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        pos_state = SimpleNamespace(
            m_cos=m_cos, m_sin=m_sin,
            sK=sK, sCosK=sCosK, sSinK=sSinK,
        )

        basic_params = SimpleNamespace(
            m_block=m_block, n_block=n_block,
            mQ=mQ, mK=mK,
            batch_size=batch_size, num_head=num_head,
            m_block_mask=m_block_mask,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma, tiled_mma=tiled_mma,
            tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt, acc_O=acc_O,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
            tKVcKV=tKVcKV,
            tKgK=tKgK, tKsK=tKsK,
            tVgV=tVgV, tVsV=tVsV,
            tKVpKV=tKVpKV,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ, tSrQ_copy_view=tSrQ_copy_view,
            tSsK=tSsK, tSrK_copy_view=tSrK_copy_view,
            tOsVt=tOsVt, tOrVt_copy_view=tOrVt_copy_view,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max, row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )

        # ────────────────────────────────────────────────────────────────────
        # Prologue: load Q + Q's cos/sin, then rotate Q in SMEM in-place.
        # ────────────────────────────────────────────────────────────────────
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None],
                )
            else:
                tQsQ[None, m, None].fill(0)

        # Cooperative element-wise GMEM→SMEM load of cos/sin for Q's positions.
        n_cs_q = self._m_block_size * self._head_dim_padded
        for j in cutlass.range_constexpr(cute.ceil_div(n_cs_q, self._num_threads)):
            flat = tidx + j * self._num_threads
            if cute.elem_less(flat, n_cs_q):
                m_local = flat // self._head_dim_padded
                d_idx = flat % self._head_dim_padded
                m_global = m_block * self._m_block_size + m_local
                if cute.elem_less(m_global, mQ.shape[1]) and cute.elem_less(d_idx, mQ.layout.shape[3]):
                    sCosQ[m_local, d_idx] = m_cos[m_global, d_idx].to(self._dtype)
                    sSinQ[m_local, d_idx] = m_sin[m_global, d_idx].to(self._dtype)
                else:
                    sCosQ[m_local, d_idx] = self._dtype(1.0)
                    sSinQ[m_local, d_idx] = self._dtype(0.0)

        last_block_enabled = m_block_mask[batch_size, num_head, m_block, n_block]
        if last_block_enabled:
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if cute.elem_less(tKVcKV[0, n, 0][1], mK.layout.shape[1]):
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tKgK[None, n, None, n_block],
                        tKsK[None, n, None],
                        pred=tKVpKV[None, n, None],
                    )
                else:
                    tKsK[None, n, None].fill(0)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # Now sQ and sCosQ/sSinQ are valid — rotate sQ in SMEM in place.
        self._rotate_smem_inplace(
            sQ, sCosQ, sSinQ,
            rows=self._m_block_size,
            d_pad=self._head_dim_padded,
            d_real=mQ.layout.shape[3],
        )

        # Load and rotate K for the first n-block.
        if last_block_enabled:
            self._load_cos_sin_K(mK, m_cos, m_sin, sCosK, sSinK, n_block)
            self.cta_sync_barrier.arrive_and_wait()
            self._rotate_smem_inplace(
                sK, sCosK, sSinK,
                rows=self._n_block_size,
                d_pad=self._head_dim_padded,
                d_real=mK.layout.shape[3],
            )
        # Final sync so all threads see rotated Q/K before MMA
        self.cta_sync_barrier.arrive_and_wait()

        # First n-block (with seqlen-K residue mask)
        basic_params.n_block = n_block_max - 1
        self.compute_one_n_block(
            basic_params, mma_params, gmem_copy_params, smem_copy_params,
            softmax_params, pos_state,
            is_first_n_block=True, in_mask_steps=True,
        )

        # Remaining n-blocks
        for n_tile in range(1, n_block_max, 1):
            basic_params.n_block = n_block_max - n_tile - 1
            self.compute_one_n_block(
                basic_params, mma_params, gmem_copy_params, smem_copy_params,
                softmax_params, pos_state,
                is_first_n_block=False, in_mask_steps=False,
            )

        self.epilogue_store_O(
            mO, acc_O, row_sum,
            sQ, sO_layout, tiled_mma,
            gmem_tiled_copy_O, tidx,
            batch_size, num_head, m_block,
        )

    # ── positional-encoding hook: K rotation per n-block ──────────────────
    @cute.jit
    def _pos_prepare_K(
        self,
        basic_params: SimpleNamespace,
        pos_state: SimpleNamespace,
        block_enabled,
        is_first_n_block: cutlass.Constexpr,
    ):
        # On the first n-block, rotation already happened in the prologue.
        # On subsequent enabled blocks, load this block's cos/sin and rotate sK.
        if cutlass.const_expr(not is_first_n_block):
            if block_enabled:
                self._load_cos_sin_K(
                    basic_params.mK, pos_state.m_cos, pos_state.m_sin,
                    pos_state.sCosK, pos_state.sSinK,
                    basic_params.n_block,
                )
                self.cta_sync_barrier.arrive_and_wait()
                self._rotate_smem_inplace(
                    pos_state.sK, pos_state.sCosK, pos_state.sSinK,
                    rows=self._n_block_size,
                    d_pad=self._head_dim_padded,
                    d_real=basic_params.mK.layout.shape[3],
                )
                self.cta_sync_barrier.arrive_and_wait()

    @cute.jit
    def _rotate_smem_inplace(
        self,
        s_x: cute.Tensor,
        s_cos: cute.Tensor,
        s_sin: cute.Tensor,
        rows: cutlass.Constexpr,
        d_pad: cutlass.Constexpr,
        d_real: cutlass.Int32,
    ):
        """Apply pairwise RoPE rotation to a (rows, d) SMEM tile in-place.

            x[r, 2i  ] ← x[r, 2i  ]·cos − x[r, 2i+1]·sin
            x[r, 2i+1] ← x[r, 2i+1]·cos + x[r, 2i  ]·sin

        cos[r, 2i] == cos[r, 2i+1] (axial RoPE uses repeat_interleave(2)),
        same for sin — so we read one (cos, sin) per pair.
        """
        tidx, _, _ = cute.arch.thread_idx()
        n_pairs = rows * (d_pad // 2)
        for j in cutlass.range_constexpr(cute.ceil_div(n_pairs, self._num_threads)):
            flat = tidx + j * self._num_threads
            if cute.elem_less(flat, n_pairs):
                pairs_per_row = d_pad // 2
                r = flat // pairs_per_row
                p = flat % pairs_per_row
                k0 = 2 * p
                k1 = k0 + 1
                if cute.elem_less(k1, d_real):
                    a = s_x[r, k0].to(cutlass.Float32)
                    b = s_x[r, k1].to(cutlass.Float32)
                    c = s_cos[r, k0].to(cutlass.Float32)
                    s = s_sin[r, k0].to(cutlass.Float32)
                    s_x[r, k0] = (a * c - b * s).to(self._dtype)
                    s_x[r, k1] = (b * c + a * s).to(self._dtype)

    @cute.jit
    def _load_cos_sin_K(
        self, mK, mCos, mSin, sCosK, sSinK,
        n_block_idx: cutlass.Int32,
    ):
        """Cooperative GMEM→SMEM load of cos/sin for K's positions in this n_block."""
        tidx, _, _ = cute.arch.thread_idx()
        rows = self._n_block_size
        d_pad = self._head_dim_padded
        n_cs = rows * d_pad
        for j in cutlass.range_constexpr(cute.ceil_div(n_cs, self._num_threads)):
            flat = tidx + j * self._num_threads
            if cute.elem_less(flat, n_cs):
                n_local = flat // d_pad
                d_idx = flat % d_pad
                n_global = n_block_idx * rows + n_local
                if cute.elem_less(n_global, mK.shape[1]) and cute.elem_less(
                    d_idx, mK.layout.shape[3]
                ):
                    sCosK[n_local, d_idx] = mCos[n_global, d_idx].to(self._dtype)
                    sSinK[n_local, d_idx] = mSin[n_global, d_idx].to(self._dtype)
                else:
                    sCosK[n_local, d_idx] = self._dtype(1.0)
                    sSinK[n_local, d_idx] = self._dtype(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark harness (rel-pos variant)
# ─────────────────────────────────────────────────────────────────────────────

def run(
    dtype: Type[cutlass.Numeric],
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_head: int,
    head_dim: int,
    softmax_scale: float = 1.0,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 128,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    import torch
    import cutlass.torch as cutlass_torch

    sparsity = kwargs.get("sparsity", 0.0)

    win_shape = int(math.sqrt(seqlen_q))
    if not FlashAttentionForwardAmpereRelPos.can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads, win_shape
    ):
        raise TypeError(
            f"Unsupported config: {dtype}, head_dim={head_dim}, "
            f"m={m_block_size}, n={n_block_size}, threads={num_threads}"
        )

    num_m_blocks = math.ceil(seqlen_q / m_block_size)
    num_n_blocks = math.ceil(seqlen_k / n_block_size)

    print(
        f"FlashAttention Ampere SM80 | dtype={dtype} B={batch_size} "
        f"Sq={seqlen_q} Sk={seqlen_k} H={num_head} D={head_dim} "
        f"scale={softmax_scale} m={m_block_size} n={n_block_size} "
        f"threads={num_threads}"
    )

    def create_tensor(seqlen):
        shape = (batch_size, seqlen, num_head, head_dim)
        t = (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=cutlass_torch.dtype(dtype))
            .cuda()
        )
        ct = (
            from_dlpack(t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(
                mode=3,
                stride_order=t.dim_order(),
                divisibility=(128 // dtype.width),
            )
        )
        return ct, t

    def create_pos(seqlen):
        shape = (batch_size, seqlen, num_head, int(math.sqrt(seqlen)))
        t = (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=cutlass_torch.dtype(dtype))
            .cuda()
        )
        ct = (
            from_dlpack(t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
        )
        return ct, t

    def create_perm(seqlen):
        t = torch.stack(
            [torch.randperm(seqlen, dtype=torch.int32) for _ in range(batch_size)]
        ).cuda()
        ct = from_dlpack(t, assumed_align=4)
        return ct, t

    def create_block_mask(nm, nn, sparsity_rate):
        """(B, H, num_m_blocks, num_n_blocks) int32; 1=compute, 0=skip."""
        t = torch.ones(batch_size, num_head, nm, nn, dtype=torch.int32, device="cuda")
        if sparsity_rate > 0.0:
            n_zero = int(sparsity_rate * nm * nn)
            idx = torch.randperm(nm * nn, device="cuda")[:n_zero]
            t.view(batch_size, num_head, -1)[:, :, idx] = 0
        ct = from_dlpack(t, assumed_align=4)
        return ct, t

    def create_A_mask(nm, nn, sparsity):
        t = torch.zeros(batch_size, num_head, nm, nn, dtype=torch.int32, device="cuda")
        t = t + torch.eye(nm, nn, dtype=torch.int32, device="cuda")
        t[:, :, :, :int(sparsity * nn)] = 1
        ct = from_dlpack(t, assumed_align=4)
        return ct, t

    q, q_torch = create_tensor(seqlen_q)
    k, k_torch = create_tensor(seqlen_k)
    v, v_torch = create_tensor(seqlen_k)
    o, o_torch = create_tensor(seqlen_q)
    rel_h, rel_h_torch = create_pos(seqlen_q)
    rel_w, rel_w_torch = create_pos(seqlen_q)
    perm_q, perm_q_torch = create_perm(seqlen_q)
    perm_k, perm_k_torch = create_perm(seqlen_k)
    block_mask, block_mask_torch = create_A_mask(num_m_blocks, num_n_blocks, sparsity)

    fa2_fwd = FlashAttentionForwardAmpereRelPos(head_dim, m_block_size, n_block_size, num_threads, win_shape)

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled_fa2_fwd = cute.compile(
        fa2_fwd, q, k, v, o, rel_h, rel_w, perm_q, perm_k, block_mask, softmax_scale, current_stream,
        options="",
    )

    if not skip_ref_check:
        compiled_fa2_fwd(q, k, v, o, rel_h, rel_w, perm_q, perm_k, block_mask, softmax_scale, current_stream)
        torch.cuda.synchronize()

        q_ref = q_torch.permute(0, 2, 1, 3).float()
        k_ref = k_torch.permute(0, 2, 1, 3).float()
        v_ref = v_torch.permute(0, 2, 1, 3).float()

        orig_q = perm_q_torch.long().cpu()
        orig_k = perm_k_torch.long().cpu()

        rel_pos_list = []
        for b in range(batch_size):
            oq = orig_q[b]
            ok = orig_k[b]
            k_row = ok // win_shape
            k_col = ok % win_shape
            rH = rel_h_torch[b:b+1, oq, :, :].permute(0, 2, 1, 3).float()
            rW = rel_w_torch[b:b+1, oq, :, :].permute(0, 2, 1, 3).float()
            rel_pos_list.append(rH[:, :, :, k_row] + rW[:, :, :, k_col])
        rel_pos = torch.cat(rel_pos_list, dim=0)

        scores = (torch.matmul(q_ref, k_ref.transpose(-2, -1)) * softmax_scale + rel_pos)

        enabled = block_mask_torch.bool()
        mask_expanded = (
            enabled
            .unsqueeze(-1)
            .expand(batch_size, num_head, num_m_blocks, num_n_blocks, n_block_size)
            .reshape(batch_size, num_head, num_m_blocks, num_n_blocks * n_block_size)
            [:, :, :, :seqlen_k]
            .repeat_interleave(m_block_size, dim=2)
            [:, :, :seqlen_q, :]
        )
        scores = scores.masked_fill(~mask_expanded, float("-inf"))

        ref_softmax = torch.nn.functional.softmax(scores, dim=-1).nan_to_num(0.0)
        ref_o = torch.matmul(ref_softmax, v_ref).permute(0, 2, 1, 3).to(cutlass_torch.dtype(dtype))
        torch.testing.assert_close(o_torch.cpu(), ref_o.cpu(), atol=1e-02, rtol=1e-04)
        print("Results verified successfully!")

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = sum(
            t.numel() * t.element_size() for t in [q_torch, k_torch, v_torch, o_torch]
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    num_masks = max(workspace_count, 1)
    prebuilt_masks = [create_block_mask(num_m_blocks, num_n_blocks, sparsity) for _ in range(num_masks)]
    mask_idx = [0]

    def generate_tensors():
        q_w, _ = create_tensor(seqlen_q)
        k_w, _ = create_tensor(seqlen_k)
        v_w, _ = create_tensor(seqlen_k)
        o_w, _ = create_tensor(seqlen_q)
        rel_h_w, _ = create_pos(seqlen_q)
        rel_w_w, _ = create_pos(seqlen_q)
        perm_q_w, _ = create_perm(seqlen_q)
        perm_k_w, _ = create_perm(seqlen_k)
        bm_w, _ = prebuilt_masks[mask_idx[0] % num_masks]
        mask_idx[0] += 1
        return testing.JitArguments(
            q_w, k_w, v_w, o_w, rel_h_w, rel_w_w,
            perm_q_w, perm_k_w, bm_w, softmax_scale, current_stream,
        )

    avg_time_us = testing.benchmark(
        compiled_fa2_fwd,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    return avg_time_us


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention v2 with CuTe DSL (Ampere SM80)")
    parser.add_argument("--dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--seqlen_q", type=int, default=196)
    parser.add_argument("--seqlen_k", type=int, default=196)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--softmax_scale", type=float, default=0.5)
    parser.add_argument("--m_block_size", type=int, default=64)
    parser.add_argument("--n_block_size", type=int, default=64)
    parser.add_argument("--num_threads", type=int, default=128)
    parser.add_argument("--warmup_iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--use_cold_l2", action="store_true", default=False)
    parser.add_argument("--sparsity", type=float, default=0.0,
                        help="fraction of (m_block, n_block) pairs to skip (0.0 = dense)")
    args = parser.parse_args()
    run(
        args.dtype, args.batch_size, args.seqlen_q, args.seqlen_k,
        args.num_head, args.head_dim, args.softmax_scale,
        args.m_block_size, args.n_block_size, args.num_threads,
        args.warmup_iterations, args.iterations,
        args.skip_ref_check, args.use_cold_l2,
        sparsity=args.sparsity,
    )
    print("PASS")
