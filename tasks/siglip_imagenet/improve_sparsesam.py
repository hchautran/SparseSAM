#!/usr/bin/env python3
"""Autoresearch-style improvement loop for SigLIP `sparsesam_partial`.

Each iteration:
  1. Apply a candidate change (mutate a module-level constant or knob)
  2. Run COCO 5K retrieval with sb=5, r=0.5, attn-only
  3. Measure sum of i2t R@1+R@5+R@10 + image-encode time
  4. KEEP if (sum > current best) or (sum within 0.5pp + time faster);
     DISCARD otherwise → revert the change
  5. Append a row to `result_siglip_improvements.tsv`

The current "best" config persists across iterations: each new candidate
is layered on top of the kept set so the search is incremental.

Captions are encoded once (text encoder unchanged across iterations).
Image features are re-encoded per iteration (the vision patch is what
each candidate touches).
"""

import os
import sys
import csv
import time
import json
import argparse
import contextlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

from tqdm import tqdm
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

from algos.registry import apply_siglip, remove_all_siglip
from algos import _pe_stage as _ps
from algos import _pe_stage_sparse as _pss
from tasks.siglip_imagenet.eval_siglip_retrieval import (
    CocoKarpathyTest, encode_images, encode_texts, recall_at_k,
)


# ──────────────────────────────────────────────────────────────────────
# Candidate-change helpers — mutate module-level knobs in place.
# ──────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def patch_attr(obj: Any, name: str, new_value: Any):
    """Temporarily set `obj.name = new_value`; restore on exit. Used so
    DISCARD'd iterations don't leave global state dirty."""
    old = getattr(obj, name)
    try:
        setattr(obj, name, new_value)
        yield
    finally:
        setattr(obj, name, old)


def _invalidate_caches():
    """Mask + perm caches in `_pe_stage_sparse` are keyed on (S, ratio,
    band, scale, ...). When we change `band`/`scale` mid-process the
    keys still match unless we explicitly clear, so flush both."""
    _pss._SPARSE_MASK_CACHE.clear()
    _pss._PERM_CACHE.clear()


def _invalidate_kernel_caches():
    """Kernel (`_KERNEL_CACHE`) is keyed on (dtype, head_dim) only — it
    doesn't include block sizes. To switch tile sizes we must flush both
    the kernel cache and the compiled-fn cache so they rebuild from the
    current `_BLOCK_CANDIDATES`."""
    _ps._KERNEL_CACHE.clear()
    _ps._COMPILED_CACHE.clear()
    _pss._SPARSE_MASK_CACHE.clear()    # mask is keyed on (..., m_blk, n_blk)
    _pss._PERM_CACHE.clear()           # perm is keyed on (..., n_block)


@contextlib.contextmanager
def _ctx_block_candidates(candidates: tuple):
    """Temporarily replace `_pe_stage._BLOCK_CANDIDATES`. Forces the
    kernel to rebuild with the new (m_block, n_block, threads) tile
    on entry, restores the original tuple on exit."""
    old = _ps._BLOCK_CANDIDATES
    try:
        _ps._BLOCK_CANDIDATES = tuple(candidates)
        _invalidate_kernel_caches()
        yield
    finally:
        _ps._BLOCK_CANDIDATES = old
        _invalidate_kernel_caches()


# ──────────────────────────────────────────────────────────────────────
# Eval primitives — load model+data once, run image encoding per iter.
# ──────────────────────────────────────────────────────────────────────

class HarnessState:
    """One-time setup: model, tokenizer, transform, dataset, text features."""

    def __init__(self, model_id: str, dataset_root: str,
                 device: str, dtype: torch.dtype, batch_size: int,
                 num_workers: int):
        print(f"Loading {model_id} …")
        self.model = AutoModel.from_pretrained(model_id).eval().to(device, dtype=dtype)
        self.tokenizer  = AutoTokenizer.from_pretrained(model_id)
        self.image_proc = AutoImageProcessor.from_pretrained(model_id)
        self.device = device
        self.dtype = dtype

        target = self.image_proc.size.height
        pil_to_tv = {0: transforms.InterpolationMode.NEAREST,
                     2: transforms.InterpolationMode.BILINEAR,
                     3: transforms.InterpolationMode.BICUBIC}
        interp = pil_to_tv.get(int(getattr(self.image_proc, "resample", 2)),
                                transforms.InterpolationMode.BILINEAR)
        self.transform = transforms.Compose([
            transforms.Resize((target, target), interpolation=interp),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_proc.image_mean,
                                 std=self.image_proc.image_std),
        ])

        print(f"Indexing COCO Karpathy 5K @ {dataset_root}")
        self.ds = CocoKarpathyTest(dataset_root, transform=self.transform)
        self.loader = DataLoader(
            self.ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        # Encode captions once — independent of vision patch.
        print("Encoding captions (one-time) …")
        t0 = time.perf_counter()
        self.text_feats = encode_texts(self.model, self.tokenizer,
                                       self.ds.captions, device, dtype)
        print(f"  text feats: {tuple(self.text_feats.shape)}  ({time.perf_counter()-t0:.1f}s)")


@torch.no_grad()
def measure(state: HarnessState) -> Tuple[Dict[str, float], float]:
    """Encode images + compute recall. Returns (metrics, encode_seconds)."""
    t0 = time.perf_counter()
    image_feats = encode_images(state.model, state.loader, state.device, state.dtype)
    elapsed = time.perf_counter() - t0
    metrics = recall_at_k(
        image_feats, state.text_feats,
        state.ds.caption_image_idx, state.ds.image_to_caption_idx,
        ks=[1, 5, 10], device=state.device,
    )
    return metrics, elapsed


def metric_sum(metrics: Dict[str, float]) -> float:
    """Sum of i2t {R@1, R@5, R@10} — primary quality signal."""
    return float(metrics["i2t_R@1"] + metrics["i2t_R@5"] + metrics["i2t_R@10"])


@torch.no_grad()
def compute_sparsity(S: int, ratio: float, band: int, kbs: float,
                     n_block: int = 64) -> float:
    """Theoretical sparsity of the banded-diagonal + keep-bar block mask
    (`_make_A_mask`). Independent of input data — depends only on
    (S, ratio, band, scale, n_block). Returns fraction of zero blocks
    in the (num_m × num_n) block grid."""
    from algos._pe_stage_sparse import _make_A_mask
    mask = _make_A_mask(B=1, H=1, T=S, ratio=ratio, m_block=n_block,
                        n_block=n_block, band_width=band,
                        keep_bar_scale=kbs, device="cpu")
    total = mask.numel()
    active = int(mask.sum())
    return 1.0 - active / total


def score(metrics: Dict[str, float], sparsity: float, alpha: float = 0.5) -> float:
    """Combined Pareto-style score: quality + α · sparsity. α=0.5 weighs
    a 100% sparsity gain as worth 0.5 sum_i2t points (i.e. ~16% relative
    quality on COCO 5K)."""
    return metric_sum(metrics) + alpha * sparsity


# ──────────────────────────────────────────────────────────────────────
# Iteration runner.
#
# Each iteration is a (name, change_fn, run_args) triple. `change_fn` is
# a callable that returns a context manager applying the change; we run
# the eval inside the context, so DISCARD'd iterations cleanly revert.
# ──────────────────────────────────────────────────────────────────────

class _NullCM:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _ctx_set_module_const(module, name, value):
    """Return a context manager that temporarily sets `module.name`
    (with cache invalidation on entry)."""
    @contextlib.contextmanager
    def _cm():
        old = getattr(module, name)
        try:
            setattr(module, name, value)
            _invalidate_caches()
            yield
        finally:
            setattr(module, name, old)
            _invalidate_caches()
    return _cm()


def run_iteration(state: HarnessState, name: str, run_args: dict,
                  cm_factory: Callable[[], Any],
                  capture_keys: Optional[List[str]] = None,
                  ) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    """Apply the patch under cm_factory(), run the partial patch with
    `run_args`, measure, and return (metrics, elapsed, applied_state).

    `applied_state` is a snapshot of (named module constants) ∪
    (current run_args) captured inside the patched context."""
    remove_all_siglip(state.model)
    class _A: pass
    a = _A()
    a.ratio = [run_args["ratio"]]
    a.partial_start_block = run_args["start_block"]
    a.mlp_merge = run_args["mlp_merge"]
    a.group_size = run_args["group_size"]
    a.sparse_ratio = None

    applied_state: Dict[str, Any] = {}
    with cm_factory():
        if capture_keys:
            applied_state = {k: getattr(_pss, k) for k in capture_keys}
        # Also capture run-level args (start_block, group_size, mlp_merge, ratio).
        applied_state.update({
            "start_block": int(run_args["start_block"]),
            "group_size":  int(run_args["group_size"]),
            "mlp_merge":   bool(run_args["mlp_merge"]),
            "ratio":       float(run_args["ratio"]),
        })
        apply_siglip(state.model, "sparsesam_partial",
                     args=a, ratio=run_args["ratio"])
        metrics, elapsed = measure(state)

    remove_all_siglip(state.model)
    return metrics, elapsed, applied_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/siglip2-base-patch16-512")
    p.add_argument("--coco-root", default="/media/volume/Chau/SAM_Quantization/data/coco")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--start-block", type=int, default=5)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--no-mlp-merge", dest="mlp_merge", action="store_false", default=False)
    p.add_argument("--mlp-merge", dest="mlp_merge", action="store_true")
    p.add_argument("--tsv", default="result_siglip_improvements.tsv")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Score = sum_i2t + alpha * sparsity. α=0.5 weighs "
                        "100% sparsity at 0.5 sum_i2t units (~16% rel quality).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    state = HarnessState(args.model, args.coco_root, args.device, dtype,
                         args.batch_size, args.num_workers)

    run_args = dict(
        ratio=args.ratio, start_block=args.start_block,
        group_size=args.group_size, mlp_merge=args.mlp_merge,
    )

    # ── Iteration plan ────────────────────────────────────────────────
    # Each entry: (name, factory(): -> contextmanager, run_args_overrides).
    # The factory wraps module-constant changes; run_args_overrides
    # changes per-call args (start_block, group_size, ratio, mlp_merge).
    # KEEP propagates BOTH into the persistent base state.
    Iter = Tuple[str, Callable[[], Any], Dict[str, Any]]
    iterations: List[Iter] = []

    def kbs(v):
        return lambda: _ctx_set_module_const(_pss, "_KEEP_BAR_SCALE", v)

    def band(v):
        return lambda: _ctx_set_module_const(_pss, "_DIAG_BAND_WIDTH", v)

    def both(s, b):
        @contextlib.contextmanager
        def cm():
            with _ctx_set_module_const(_pss, "_KEEP_BAR_SCALE", s):
                with _ctx_set_module_const(_pss, "_DIAG_BAND_WIDTH", b):
                    yield
        return cm

    iterations.append(("baseline", lambda: _NullCM(), {}))

    # Phase 1 (15): keep_bar_scale fine sweep.
    for s in [0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0,
              2.25, 2.5, 3.0, 4.0, 5.0, 8.0]:
        iterations.append((f"kbs={s}", kbs(s), {}))

    # Phase 2 (8): band_width sweep (layered on whatever kbs is best).
    for b in [1, 2, 3, 4, 5, 6, 7, 8]:
        iterations.append((f"band={b}", band(b), {}))

    # Phase 3 (11): start_block sweep.
    for sb in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]:
        iterations.append((f"sb={sb}", lambda: _NullCM(), {"start_block": sb}))

    # Phase 4 (4): group_size sweep. SigLIP base-512 has S=1024 — divisible
    # by 2/4/8/16/32 cleanly.
    for gs in [2, 8, 16, 32]:
        iterations.append((f"gs={gs}", lambda: _NullCM(), {"group_size": gs}))

    # Phase 5 (2): mlp_merge toggle (current run is False; check True).
    iterations.append(("mlp_merge=True", lambda: _NullCM(), {"mlp_merge": True}))
    iterations.append(("mlp_merge=False", lambda: _NullCM(), {"mlp_merge": False}))

    # Phase 6 (60): full grid (kbs × band × sb) with ratio fixed at 0.5.
    grid_configs: List[Tuple[float, int, int]] = []
    for s in [1.25, 1.5, 1.75, 2.0]:
        for b in [1, 2, 3]:
            for sb in [3, 4, 5, 6, 7]:
                grid_configs.append((s, b, sb))
    for s, b, sb in grid_configs[:60]:
        iterations.append((
            f"grid kbs={s} band={b} sb={sb}",
            both(s, b),
            {"start_block": sb},
        ))

    print(f"Iteration plan: {len(iterations)} configs")

    # State across iterations — kept changes accumulate.
    kept_constants: Dict[str, Any] = {
        "_KEEP_BAR_SCALE": _pss._KEEP_BAR_SCALE,
        "_DIAG_BAND_WIDTH": _pss._DIAG_BAND_WIDTH,
    }

    @contextlib.contextmanager
    def kept_ctx():
        olds = {k: getattr(_pss, k) for k in kept_constants}
        for k, v in kept_constants.items():
            setattr(_pss, k, v)
        _invalidate_caches()
        try:
            yield
        finally:
            for k, v in olds.items():
                setattr(_pss, k, v)
            _invalidate_caches()

    # ── Header for the TSV log ────────────────────────────────────────
    tsv_path = os.path.join(_REPO, args.tsv)
    write_header = not os.path.exists(tsv_path)
    f = open(tsv_path, "a", newline="")
    w = csv.writer(f, delimiter="\t")
    if write_header:
        w.writerow([
            "iter", "name", "i2t_R@1", "i2t_R@5", "i2t_R@10",
            "t2i_R@1", "t2i_R@5", "t2i_R@10",
            "sum_i2t", "sparsity", "score", "encode_s",
            "delta_score", "delta_s",
            "decision", "applied_state",
        ])
    f.flush()

    # ── Run ───────────────────────────────────────────────────────────
    best: Optional[Tuple[Dict[str, float], float]] = None  # (metrics, elapsed)
    print(f"\nWriting log → {tsv_path}\n")

    capture_keys = list(kept_constants.keys())
    # Persistent run_args base — kept across iterations on KEEP.
    kept_run_args = dict(run_args)

    for i, (name, cm_factory, run_overrides) in enumerate(iterations):
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"iter {i:02d}: {name}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Effective run_args = kept base + this iteration's override.
        eff_run_args = {**kept_run_args, **run_overrides}

        # Layer this iteration's module-const change on top of kept set.
        @contextlib.contextmanager
        def stacked():
            with kept_ctx():
                with cm_factory():
                    yield
        try:
            metrics, elapsed, applied_state = run_iteration(
                state, name, eff_run_args, stacked, capture_keys=capture_keys,
            )
        except Exception as e:
            print(f"  iteration crashed: {e!r}")
            w.writerow([i, name, "", "", "", "", "", "", "", "", "", "",
                        "ERROR", json.dumps({**kept_constants, **kept_run_args},
                                            default=str)])
            f.flush()
            continue

        m_sum = metric_sum(metrics)
        sparsity = compute_sparsity(
            S=image_S(state), ratio=eff_run_args["ratio"],
            band=applied_state.get("_DIAG_BAND_WIDTH", 1),
            kbs=applied_state.get("_KEEP_BAR_SCALE", 1.0),
        )
        sc = score(metrics, sparsity, alpha=args.alpha)

        if best is None:
            decision = "KEEP_BASELINE"
            delta_score = 0.0
            delta_s = 0.0
            best = (metrics, elapsed, sparsity, sc)
            kept_constants = {k: applied_state[k] for k in capture_keys
                              if k in applied_state}
            kept_run_args = {k: applied_state[k] for k in run_args
                             if k in applied_state}
        else:
            best_score = best[3]
            best_s     = best[1]
            delta_score = sc - best_score
            delta_s     = elapsed - best_s

            kept = False
            if sc > best_score + 1e-4:
                decision = "KEEP_BETTER_SCORE"
                kept = True
            else:
                decision = "DISCARD"

            if kept:
                best = (metrics, elapsed, sparsity, sc)
                kept_constants = {k: applied_state[k] for k in capture_keys
                                  if k in applied_state}
                kept_run_args = {k: applied_state[k] for k in run_args
                                 if k in applied_state}

        msg = (f"  i2t R@1={metrics['i2t_R@1']*100:.2f}  "
               f"sum_i2t={m_sum:.4f}  sparsity={sparsity*100:.1f}%  "
               f"score={sc:.4f}  encode={elapsed:.1f}s  "
               f"Δscore={delta_score:+.4f}  Δs={delta_s:+.1f}  → {decision}")
        print(msg)

        w.writerow([
            i, name,
            f"{metrics['i2t_R@1']:.4f}", f"{metrics['i2t_R@5']:.4f}",
            f"{metrics['i2t_R@10']:.4f}",
            f"{metrics['t2i_R@1']:.4f}", f"{metrics['t2i_R@5']:.4f}",
            f"{metrics['t2i_R@10']:.4f}",
            f"{m_sum:.4f}", f"{sparsity:.4f}", f"{sc:.4f}", f"{elapsed:.1f}",
            f"{delta_score:+.4f}", f"{delta_s:+.1f}",
            decision, json.dumps(applied_state, default=str),
        ])
        f.flush()

    f.close()

    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Best: sum_i2t={metric_sum(best[0]):.4f}  sparsity={best[2]*100:.1f}%  "
          f"score={best[3]:.4f}  encode={best[1]:.1f}s")
    print(f"Kept constants: {kept_constants}  +  {kept_run_args}")
    print(f"Log → {tsv_path}")


def image_S(state: HarnessState) -> int:
    """Number of image patch tokens for the loaded model."""
    cfg = state.model.config.vision_config
    grid = cfg.image_size // cfg.patch_size
    return grid * grid


if __name__ == "__main__":
    main()
