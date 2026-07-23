#!/usr/bin/env python3
"""L4 encoder benchmark: peak memory, latency/speedup, and GFLOPs for the SAM-HQ
ViT image encoder, baseline vs sparsesam across density ratios.

Uses a randomly-initialized encoder (no checkpoint) and random 1024x1024 fp16
input — valid for memory/latency/GFLOPs, which do not depend on trained weights.
Mirrors the real run path in tasks/sam_hq44k/eval_hq44k.py:
    sam_model_registry[type](checkpoint=None).to('cuda').half()
    image_encoder(preprocessed_1024x1024_fp16)

GFLOPs note: sparsesam runs attention inside a custom CUTLASS CUTE kernel that
fvcore cannot trace/count. So fvcore GFLOPs is exact for the dense baseline; for
sparsesam we report the fvcore-countable part when traceable and otherwise fall
back to a per-block analytic estimate (see analytic_encoder_gflops).
"""
import os
import sys
import gc
import time
import argparse
import statistics

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam-hq"))

from segment_anything import sam_model_registry
from algos.registry import apply_sam, remove_all_sam


def reset_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def time_encoder(encoder, x, iters, warmup):
    for _ in range(warmup):
        encoder(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoder(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)  # ms
    return statistics.median(ts), min(ts)


def analytic_encoder_gflops(encoder, seqlen_tokens, density):
    """Rough per-encoder GFLOPs for one image. Counts the two dominant terms per
    block: attention (QK^T + PV, scaled by `density` for sparsesam) and MLP.
    Baseline uses density=1.0. Windowed vs global token counts are approximated
    by the full token grid — an upper bound on attention, exact on MLP."""
    blocks = encoder.blocks
    total = 0.0
    for blk in blocks:
        dim = blk.attn.qkv.in_features
        heads = blk.attn.num_heads
        N = seqlen_tokens
        # attention: 2 * (QK^T: N*N*dim) + (PV: N*N*dim) ~= 4*N*N*dim  (fwd, mac=2 flops)
        attn = 2 * (2.0 * N * N * dim) * density
        # qkv proj + out proj: 2 * N * dim * (3*dim) + 2 * N * dim * dim
        proj = 2 * N * dim * (3 * dim) + 2 * N * dim * dim
        # mlp: 2 layers dim->4dim->dim
        mlp = 2 * (2.0 * N * dim * (4 * dim))
        total += attn + proj + mlp
    return total / 1e9


def fvcore_gflops(encoder, x):
    try:
        from fvcore.nn import FlopCountAnalysis
        import logging
        logging.getLogger("fvcore").setLevel(logging.ERROR)
        fca = FlopCountAnalysis(encoder, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        g = fca.total() / 1e9
        return g
    except Exception as e:
        return None


def build_encoder(model_type):
    sam = sam_model_registry[model_type](checkpoint=None).to("cuda").half()
    return sam.image_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", default="vit_l")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=1024)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)} cc{torch.cuda.get_device_capability(0)}")
    print(f"model={args.model_type} img={args.img_size} iters={args.iters} warmup={args.warmup}\n")

    rows = []
    for bs in args.batch_sizes:
        x = torch.randn(bs, 3, args.img_size, args.img_size, device="cuda", dtype=torch.float16)

        # ---- baseline (dense) ----
        enc = build_encoder(args.model_type)
        enc.eval()
        # token grid = (img/patch)^2
        patch = enc.patch_embed.proj.kernel_size[0]
        Ntok = (args.img_size // patch) ** 2
        reset_mem()
        med, best = time_encoder(enc, x, args.iters, args.warmup)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        g_fv = fvcore_gflops(enc, x)
        g_an = analytic_encoder_gflops(enc, Ntok, 1.0) * bs
        rows.append(("baseline", "-", bs, med, best, peak, 1.0, g_fv, g_an))
        base_med = med
        del enc; reset_mem()

        # ---- sparsesam across ratios ----
        for r in args.ratios:
            enc = build_encoder(args.model_type)
            enc.eval()
            apply_sam(enc, "sparsesam", ratio=r)
            reset_mem()
            try:
                med, best = time_encoder(enc, x, args.iters, args.warmup)
                peak = torch.cuda.max_memory_allocated() / 1024**2
            except Exception as e:
                print(f"sparsesam r={r} bs={bs} FAILED: {type(e).__name__}: {e}")
                remove_all_sam(enc); del enc; reset_mem()
                continue
            g_fv = fvcore_gflops(enc, x)
            g_an = analytic_encoder_gflops(enc, Ntok, r) * bs
            rows.append((f"sparsesam", r, bs, med, best, peak, base_med / med, g_fv, g_an))
            remove_all_sam(enc); del enc; reset_mem()

    # ---- report ----
    hdr = f"{'algo':10s} {'ratio':>5s} {'bs':>3s} {'lat_med_ms':>10s} {'lat_min_ms':>10s} {'peak_MB':>9s} {'speedup':>7s} {'GFLOPs_fv':>10s} {'GFLOPs_an':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for algo, r, bs, med, best, peak, sp, gfv, gan in rows:
        rr = f"{r:.2f}" if isinstance(r, float) else r
        fv = f"{gfv:.1f}" if gfv is not None else "n/a"
        print(f"{algo:10s} {rr:>5s} {bs:>3d} {med:>10.2f} {best:>10.2f} {peak:>9.1f} {sp:>7.2f} {fv:>10s} {gan:>10.1f}")


if __name__ == "__main__":
    main()
