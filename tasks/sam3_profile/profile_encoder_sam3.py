#!/usr/bin/env python3
"""Profile the vendored Meta SAM3 image-encoder ViT, optionally comparing
against the SparseSAM MLP-merge patch.

The model loaded is `sam3.model.vitdet.ViT` — the shared vision backbone of
SAM3 / SAM3.1 (1008x1008 input, patch=14, embed_dim=1024, depth=32,
window_size=24, global attention at blocks 7/15/23/31). Per-block hooks
collect attn / MLP / total timings.

Usage
-----
  # Mock weights (no HF auth, no download — exercises the full encoder graph):
  python tasks/sam3_profile/profile_encoder_sam3.py --mock

  # Mock + SparseSAM MLP-merge patch:
  python tasks/sam3_profile/profile_encoder_sam3.py --mock \
      --algo sparsesam --ratio 0.5

  # Real SAM3 checkpoint (requires HF auth & accepted model license):
  python tasks/sam3_profile/profile_encoder_sam3.py --version sam3 \
      --algo sparsesam --ratio 0.5

  # Or load from a local checkpoint:
  python tasks/sam3_profile/profile_encoder_sam3.py \
      --ckpt /path/to/sam3.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam3"))


# ─────────────────────────────────────────────────────────────────────────────
# CUDA timing
# ─────────────────────────────────────────────────────────────────────────────


class CUDATimer:
    def __init__(self):
        self._start = None
        self.times_ms: list = []

    def start(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._start = e

    def stop(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        torch.cuda.synchronize()
        self.times_ms.append(self._start.elapsed_time(e))

    def reset(self):
        self.times_ms.clear()

    def mean(self):
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0


def _attach(module, name, timers, handles):
    t = CUDATimer()
    timers[name] = t
    handles.append(module.register_forward_pre_hook(lambda m, i: t.start()))
    handles.append(module.register_forward_hook(lambda m, i, o: t.stop()))


def attach_timers(vit):
    """Attach hooks to a `sam3.model.vitdet.ViT`."""
    timers, handles = {}, []
    a = lambda m, n: _attach(m, n, timers, handles)

    a(vit, "ViT (total)")
    a(vit.patch_embed, "patch_embed")
    for i, blk in enumerate(vit.blocks):
        a(blk, f"block[{i:02d}]")
        a(blk.norm1, f"block[{i:02d}].norm1")
        a(blk.attn, f"block[{i:02d}].attn")
        a(blk.norm2, f"block[{i:02d}].norm2")
        a(blk.mlp, f"block[{i:02d}].mlp")
    return timers, handles


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def _get(timers, key):
    return timers[key].mean() if key in timers else 0.0


def print_report(timers, n_runs, batch_size, vit, label="baseline"):
    n_blocks = len(vit.blocks)
    global_ids = set(vit.full_attn_ids)

    total_ms = _get(timers, "ViT (total)")
    sum_blk = sum_attn = sum_mlp = 0.0
    print(f"\n{'=' * 78}")
    print(
        f"  SAM3 ViT  |  {label}  |  bs={batch_size}  runs={n_runs}  "
        f"total={total_ms:.1f}ms"
    )
    print(f"{'=' * 78}")
    print(
        f"  {'block':<10} {'total':>8}  {'attn':>8}  {'mlp':>8}  "
        f"{'attn%':>6}  {'mlp%':>6}  type"
    )
    print(f"  {'-' * 64}")
    for i in range(n_blocks):
        tag = f"block[{i:02d}]"
        blk = _get(timers, tag)
        att = _get(timers, tag + ".attn")
        mlp = _get(timers, tag + ".mlp")
        sum_blk += blk
        sum_attn += att
        sum_mlp += mlp
        a_pct = att / blk * 100 if blk > 0 else 0
        m_pct = mlp / blk * 100 if blk > 0 else 0
        kind = "global" if i in global_ids else f"win{vit.blocks[i].window_size}"
        print(
            f"  block[{i:02d}]  {blk:>8.3f}  {att:>8.3f}  {mlp:>8.3f}  "
            f"{a_pct:>5.1f}%  {m_pct:>5.1f}%  {kind}"
        )
    a_pct = sum_attn / sum_blk * 100 if sum_blk > 0 else 0
    m_pct = sum_mlp / sum_blk * 100 if sum_blk > 0 else 0
    blk_pct = sum_blk / total_ms * 100 if total_ms > 0 else 0
    print(f"  {'-' * 64}")
    print(
        f"  {'all blocks':<10} {sum_blk:>8.1f}  {sum_attn:>8.1f}  "
        f"{sum_mlp:>8.1f}  {a_pct:>5.1f}%  {m_pct:>5.1f}%  "
        f"({blk_pct:.1f}% of total)"
    )
    pe = _get(timers, "patch_embed")
    print(f"\n  patch_embed={pe:.2f}ms")
    print(f"{'=' * 78}\n")


def print_comparison(t_base, t_patch, vit, n_runs, batch_size, algo, ratio):
    n_blocks = len(vit.blocks)
    global_ids = set(vit.full_attn_ids)

    def spd(b, t):
        return b / t if t > 1e-6 else float("inf")

    def fmt(s):
        return f"{s:.2f}x" if s != float("inf") else "  inf"

    total_b = _get(t_base, "ViT (total)")
    total_p = _get(t_patch, "ViT (total)")

    print(f"\n{'=' * 96}")
    print(
        f"  Baseline vs {algo.upper()} ratio={ratio}  |  bs={batch_size}  runs={n_runs}"
    )
    print(f"{'=' * 96}")
    print(
        f"  {'block':<10} {'total':>13}  {'spd':>5} | "
        f"{'attn':>13}  {'spd':>5} | {'mlp':>13}  {'spd':>5}  type"
    )
    print(
        f"  {'':<10} {'base':>6} {'patch':>6}  {'':>5} | "
        f"{'base':>6} {'patch':>6}  {'':>5} | {'base':>6} {'patch':>6}"
    )
    print(f"  {'-' * 96}")
    s_b_blk = s_p_blk = s_b_att = s_p_att = s_b_mlp = s_p_mlp = 0.0
    for i in range(n_blocks):
        tag = f"block[{i:02d}]"
        b_blk = _get(t_base, tag)
        p_blk = _get(t_patch, tag)
        b_att = _get(t_base, tag + ".attn")
        p_att = _get(t_patch, tag + ".attn")
        b_mlp = _get(t_base, tag + ".mlp")
        p_mlp = _get(t_patch, tag + ".mlp")
        s_b_blk += b_blk
        s_p_blk += p_blk
        s_b_att += b_att
        s_p_att += p_att
        s_b_mlp += b_mlp
        s_p_mlp += p_mlp
        kind = "global" if i in global_ids else f"win{vit.blocks[i].window_size}"
        print(
            f"  block[{i:02d}]  {b_blk:>6.2f} {p_blk:>6.2f}  {fmt(spd(b_blk, p_blk)):>5} | "
            f"{b_att:>6.2f} {p_att:>6.2f}  {fmt(spd(b_att, p_att)):>5} | "
            f"{b_mlp:>6.2f} {p_mlp:>6.2f}  {fmt(spd(b_mlp, p_mlp)):>5}  {kind}"
        )
    print(f"  {'-' * 96}")
    print(
        f"  {'Σ blocks':<10} {s_b_blk:>6.1f} {s_p_blk:>6.1f}  "
        f"{fmt(spd(s_b_blk, s_p_blk)):>5} | "
        f"{s_b_att:>6.1f} {s_p_att:>6.1f}  {fmt(spd(s_b_att, s_p_att)):>5} | "
        f"{s_b_mlp:>6.1f} {s_p_mlp:>6.1f}  {fmt(spd(s_b_mlp, s_p_mlp)):>5}"
    )
    print(
        f"  {'TOTAL':<10} {total_b:>6.1f} {total_p:>6.1f}  "
        f"{fmt(spd(total_b, total_p)):>5}"
    )
    print(f"{'=' * 96}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────


def load_sam3_vit(mock: bool, version: str, ckpt: str | None, device, dtype):
    """Returns the `sam3.model.vitdet.ViT` instance ready for forward.

    `mock=True` builds the same architecture with random weights and skips
    any HF download. Otherwise the full SAM3 image model is built via the
    vendored `build_sam3_image_model` (optionally with a local ckpt path).
    """
    from sam3.model_builder import (
        _create_vit_backbone,
        build_sam3_image_model,
    )

    if mock:
        print("Building mock SAM3 ViT (random weights, no checkpoint) ...")
        vit = _create_vit_backbone(compile_mode=None)
        vit = vit.to(device=device, dtype=dtype).eval()
        return vit, None

    print(f"Building full SAM3 image model (version={version}, ckpt={ckpt}) ...")
    model = build_sam3_image_model(
        device="cuda" if device == "cuda" else "cpu",
        eval_mode=True,
        checkpoint_path=ckpt,
        load_from_HF=(ckpt is None),
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    # Trim down to just the ViT trunk and cast to requested dtype.
    vit = model.backbone.vision_backbone.trunk.to(dtype=dtype).eval()
    return vit, model


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────


def _run_profile(vit, dummy, n_warmup, n_runs, label):
    timers, handles = attach_timers(vit)
    print(f"  Warming up [{label}] ({n_warmup} passes) ...")
    with torch.no_grad():
        for _ in range(n_warmup):
            vit(dummy)
    for t in timers.values():
        t.reset()

    print(f"  Profiling  [{label}] ({n_runs} passes) ...")
    with torch.no_grad():
        for _ in range(n_runs):
            vit(dummy)

    for h in handles:
        h.remove()
    return timers


def main():
    p = argparse.ArgumentParser(
        description="Profile vendored SAM3 image-encoder ViT components",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use random-weight ViT (no HF download, no auth).",
    )
    p.add_argument(
        "--version", type=str, default="sam3", choices=["sam3", "sam3.1"]
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Local SAM3 checkpoint .pt (otherwise downloaded from HF).",
    )

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--img-size", type=int, default=1008)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="SAM3's fused MLP (addmm_act) always casts intermediates to "
        "bf16, so fp16 weights will dtype-mismatch in fc2. Use bf16.",
    )

    p.add_argument(
        "--algo",
        type=str,
        default="none",
        choices=["none", "sparsesam"],
        help="Run a second pass with this SparseSAM variant and "
        "print a baseline-vs-patched comparison table.",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Token-keep ratio for the MLP-merge patch (0 < ratio < 1).",
    )
    p.add_argument(
        "--no-mlp-merge",
        action="store_true",
        help="Use the patched Block class but run the full MLP "
        "(useful for measuring patch overhead alone).",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available.", file=sys.stderr)
        sys.exit(1)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.dtype
    ]
    device = "cuda"

    vit, _ = load_sam3_vit(
        mock=args.mock, version=args.version, ckpt=args.ckpt, device=device, dtype=dtype
    )
    dummy = torch.randn(
        args.batch_size, 3, args.img_size, args.img_size, device=device, dtype=dtype
    )

    print("\n── Baseline (stock SAM3 ViT) ──")
    t_base = _run_profile(vit, dummy, args.n_warmup, args.n_runs, "baseline")

    if args.algo == "none":
        print_report(t_base, args.n_runs, args.batch_size, vit, label="baseline")
        return

    from algos.sparsesam.sam3 import apply_patch, remove_patch

    print(f"\n── SparseSAM MLP-merge ratio={args.ratio} ──")
    apply_patch(vit, ratio=args.ratio, mlp_merge=(not args.no_mlp_merge))
    t_patch = _run_profile(
        vit, dummy, args.n_warmup, args.n_runs, f"sparsesam r={args.ratio}"
    )
    remove_patch(vit)

    print_report(t_base, args.n_runs, args.batch_size, vit, label="baseline")
    print_report(
        t_patch, args.n_runs, args.batch_size, vit, label=f"sparsesam r={args.ratio}"
    )
    print_comparison(
        t_base, t_patch, vit, args.n_runs, args.batch_size, args.algo, args.ratio
    )


if __name__ == "__main__":
    main()
