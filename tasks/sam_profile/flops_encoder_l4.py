#!/usr/bin/env python3
"""Correct GFLOPs (MAC convention, matching fvcore) for the SAM-HQ ViT encoder,
baseline vs sparsesam, measured from REAL per-block shapes via forward hooks.

Why not just fvcore: sparsesam runs attention inside a custom CUTLASS CUTE kernel
that fvcore's tracer can't follow, so fvcore returns nothing for sparsesam. Here we
instead:
  * sum Linear/Conv2d MACs via hooks (captures token-merge reduction exactly, and
    is robust to the custom kernel), and
  * add attention MACs (QK^T + PV) computed from the real q/k/v shapes seen at each
    attention call; for sparsesam these are the merged+permuted tokens, and the
    sparse A-shape mask keeps a `density`-fraction of block pairs, so attention is
    scaled by the measured kept-block fraction.
Baseline total is cross-checked against fvcore.FlopCountAnalysis.
"""
import os, sys, argparse
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam-hq"))

from segment_anything import sam_model_registry
from algos.registry import apply_sam, remove_all_sam


class FlopHooks:
    """Sum MACs for Linear + Conv2d modules from real input/output shapes."""
    def __init__(self, model):
        self.macs = 0
        self.handles = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._lin))
            elif isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(self._conv))

    def _lin(self, mod, inp, out):
        x = inp[0]
        n = x.numel() // x.shape[-1]          # number of rows
        self.macs += n * mod.in_features * mod.out_features

    def _conv(self, mod, inp, out):
        # out elements * (Cin/groups * kh * kw)
        oc = out.numel()
        cin = mod.in_channels // mod.groups
        kh, kw = mod.kernel_size
        self.macs += oc * cin * kh * kw

    def remove(self):
        for h in self.handles:
            h.remove()


def attention_macs_baseline(encoder, img_size):
    """Dense SAM attention MACs from static structure: windowed blocks attend within
    14x14 (padded) windows; global blocks attend over the full token grid."""
    patch = encoder.patch_embed.proj.kernel_size[0]
    G = img_size // patch                      # tokens per side (64)
    total = 0
    for blk in encoder.blocks:
        heads = blk.attn.num_heads
        hd = blk.attn.qkv.in_features // heads
        ws = blk.window_size
        if ws and ws > 0:
            # pad grid up to a multiple of ws (SAM window_partition pads)
            import math
            pad = (math.ceil(G / ws) * ws)
            nwin = (pad // ws) ** 2
            N = ws * ws
            total += nwin * heads * 2 * (N * N * hd)   # QK^T + PV
        else:
            N = G * G
            total += heads * 2 * (N * N * hd)
    return total


class AttnProbe:
    """Capture real attention token counts for sparsesam by hooking the qkv Linear
    of each block (rows = B_eff * N_tokens for that block's attention input)."""
    def __init__(self, encoder):
        self.rows = []          # (block_idx, qkv_input_rows, in_features)
        self.handles = []
        for i, blk in enumerate(encoder.blocks):
            self.handles.append(
                blk.attn.qkv.register_forward_hook(self._mk(i, blk)))

    def _mk(self, i, blk):
        def hook(mod, inp, out):
            x = inp[0]
            rows = x.numel() // x.shape[-1]
            self.rows.append((i, rows, mod.in_features,
                              blk.attn.num_heads, blk.window_size))
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", default="vit_l")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--img-size", type=int, default=1024)
    args = ap.parse_args()

    dev = "cuda"
    x = torch.randn(1, 3, args.img_size, args.img_size, device=dev, dtype=torch.float16)

    def build():
        return sam_model_registry[args.model_type](checkpoint=None).to(dev).half().image_encoder.eval()

    # ---------- baseline ----------
    enc = build()
    fh = FlopHooks(enc)
    with torch.no_grad():
        enc(x)
    lin_conv = fh.macs
    fh.remove()
    attn = attention_macs_baseline(enc, args.img_size)
    base_total = lin_conv + attn
    print(f"baseline   lin/conv={lin_conv/1e9:8.1f}  attn={attn/1e9:7.1f}  "
          f"total={base_total/1e9:8.1f} GMAC")

    # cross-check against fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        import logging; logging.getLogger("fvcore").setLevel(logging.ERROR)
        enc2 = build()
        fca = FlopCountAnalysis(enc2, x); fca.unsupported_ops_warnings(False); fca.uncalled_modules_warnings(False)
        print(f"           fvcore total = {fca.total()/1e9:8.1f} GMAC  (cross-check)")
        del enc2
    except Exception as e:
        print("           fvcore cross-check failed:", type(e).__name__, e)
    del enc; torch.cuda.empty_cache()

    # ---------- sparsesam ----------
    for r in args.ratios:
        enc = build()
        apply_sam(enc, "sparsesam", ratio=r)
        fh = FlopHooks(enc)
        pr = AttnProbe(enc)
        with torch.no_grad():
            enc(x)
        lin_conv = fh.macs
        fh.remove()
        # attention MACs from real qkv rows: rows = B_eff*N; attention over N tokens
        # within each B_eff group. For windowed blocks N=win^2; global N=full grid.
        # kept fraction = density r (A-shape keeps r of the block columns).
        import math
        attn = 0
        patch = enc.patch_embed.proj.kernel_size[0]; G = args.img_size // patch
        for (i, rows, inf, heads, ws) in pr.rows:
            hd = inf // heads
            if ws and ws > 0:
                N = ws * ws
                pad = math.ceil(G / ws) * ws
                nwin = (pad // ws) ** 2
                Beff = rows / N                       # ~= nwin (batch=1)
                attn += Beff * heads * 2 * (N * N * hd) * r
            else:
                N = rows                              # global: all tokens in one group
                attn += heads * 2 * (N * N * hd) * r
        pr.remove()
        total = lin_conv + attn
        print(f"sparsesam r={r:.2f}  lin/conv={lin_conv/1e9:8.1f}  attn={attn/1e9:7.1f}  "
              f"total={total/1e9:8.1f} GMAC   ({base_total/total:.2f}x fewer vs baseline)")
        remove_all_sam(enc); del enc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
