#!/usr/bin/env python3
"""Separate implementation gain from algorithmic gain.

A latency comparison between SparseSAM (fused CUTE kernel) and baselines that
run stock PyTorch attention conflates two things: a better kernel and a better
algorithm. This script measures both on the same GPU by inserting a
**fused-dense control** — the identical CUTE kernel with *no* sparsification:

  A  baseline        stock SAM-HQ attention (manual QK^T + decomposed rel-pos
                     bias + softmax + PV), full MLP
  B  fused-dense     SparseSAM patch at ratio=1.0: same CUTE kernel, dense mask,
                     full MLP. Differs from A only in implementation.
  C  attn-sparse     SparseSAM at ratio<1 with mlp_merge=False: A-shape sparse
                     attention, full MLP. Differs from B only in the algorithm.
  D  sparsesam       ratio<1 with mlp_merge=True: adds the keep-token MLP.
  E  tome            baseline algorithm (bipartite merge) on stock attention.

Then:
  A/B  = implementation gain alone (kernel, no algorithm)
  B/D  = algorithmic gain alone (algorithm, at fixed implementation)
  A/D  = the headline number, = the product of the two

Note B still pays SparseSAM's token-permutation cost, which the algorithm needs
but a pure kernel swap would not. That makes B slightly slow, i.e. it understates
the implementation gain and overstates the algorithmic gain — the conservative
direction for the algorithm claim is the opposite, so read A/B as a lower bound.
"""
import os, sys, gc, time, argparse, statistics

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam-hq"))

from segment_anything import sam_model_registry
from algos.registry import apply_sam


def reset_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def build_encoder(model_type):
    """Fresh encoder per config — the patches mutate module classes in place."""
    sam = sam_model_registry[model_type](checkpoint=None).to("cuda").half()
    return sam.image_encoder.eval()


@torch.no_grad()
def time_encoder(encoder, x, iters, warmup):
    for _ in range(warmup):
        encoder(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        encoder(x)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", default="vit_l")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)} cc{torch.cuda.get_device_capability(0)}")
    print(f"model={args.model_type} bs={args.batch_size} img={args.img_size} "
          f"iters={args.iters} warmup={args.warmup}\n")

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size,
                    device="cuda", dtype=torch.float16)
    # SparseSAM's patch is not reachable through the registry at ratio=1.0
    # (the registry short-circuits "nothing to merge"), so call it directly.
    from algos.sparsesam.sam import apply_patch as sparsesam_patch

    configs = [("A baseline", None, None)]
    configs.append(("B fused-dense", "fused_dense", 1.0))
    for r in args.ratios:
        configs.append((f"C attn-sparse {r:.2f}", "attn_only", r))
    for r in args.ratios:
        configs.append((f"D sparsesam {r:.2f}", "sparsesam", r))
    for r in args.ratios:
        configs.append((f"E tome {r:.2f}", "tome", r))

    rows = []
    for label, kind, ratio in configs:
        enc = build_encoder(args.model_type)
        if kind == "fused_dense":
            sparsesam_patch(enc, algo="tome", ratio=1.0)
        elif kind == "attn_only":
            sparsesam_patch(enc, algo="tome", ratio=ratio, mlp_merge=False)
        elif kind == "sparsesam":
            sparsesam_patch(enc, algo="tome", ratio=ratio, mlp_merge=True)
        elif kind == "tome":
            apply_sam(enc, "tome", ratio=ratio)
        reset_mem()
        try:
            med, mn = time_encoder(enc, x, args.iters, args.warmup)
            peak = torch.cuda.max_memory_allocated() / 1024**2
        except torch.OutOfMemoryError:
            print(f"{label}: OOM")
            del enc; reset_mem(); continue
        rows.append((label, med, mn, peak))
        del enc; reset_mem()

    base = rows[0][1]
    fused_dense = next((m for l, m, _, _ in rows if l.startswith("B ")), None)

    hdr = (f"{'config':22s} {'lat_ms':>8s} {'min_ms':>8s} {'peak_MB':>9s} "
           f"{'vs A':>7s} {'vs B':>7s}")
    print(hdr); print("-" * len(hdr))
    for label, med, mn, peak in rows:
        vs_b = f"{fused_dense/med:>6.2f}x" if fused_dense else "    n/a"
        print(f"{label:22s} {med:>8.2f} {mn:>8.2f} {peak:>9.1f} "
              f"{base/med:>6.2f}x {vs_b:>7s}")

    if fused_dense:
        print(f"\nimplementation gain (A/B, kernel only, no algorithm): "
              f"{base/fused_dense:.2f}x")
        for label, med, _, _ in rows:
            if label.startswith("D "):
                print(f"  {label}: total {base/med:.2f}x = "
                      f"implementation {base/fused_dense:.2f}x "
                      f"x algorithmic {fused_dense/med:.2f}x")


if __name__ == "__main__":
    main()
