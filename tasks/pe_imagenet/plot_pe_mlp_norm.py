#!/usr/bin/env python3
"""Per-token MLP update norm  ‖Δ_MLP‖²  across selected blocks of the
Perception Encoder (PE-Core-L14-336 by default), as 2D spatial heatmaps.

Captures the MLP residual at each block by monkey-patching its
`ResidualAttentionBlock.forward` to record (x_in_to_mlp, x_out). The
delta added by the MLP path equals (x_out − x_in_to_mlp) per token; we
take the L2 norm and reshape to a (grid, grid) image (cls token is
discarded).

Outputs PNG, SVG, PDF triplets to `--out-dir`.
"""

import argparse
import os
import sys
import types
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "perception_models"))

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms


# Cyan colormap, matching the SAM-HQ keep_and_mlp figure.
CYAN_SOLID = LinearSegmentedColormap.from_list('cyan_solid', [
    (0.00, '#f0f9fa'),
    (0.20, '#a8dee2'),
    (0.50, '#3aaab4'),
    (0.80, '#1a6c75'),
    (1.00, '#0a3a40'),
])


def _patch_block_capture(block, store: List[Dict[str, torch.Tensor]]):
    """Replace `block.forward` with a wrapper that records x_in_to_mlp and
    x_out. Original logic of ResidualAttentionBlock.forward:

        x = x + drop_path1(ls_1(attn(ln_1(x))))    # post-attn residual
        x = x + drop_path2(ls_2(mlp(ln_2(x))))     # post-MLP residual
    """
    orig_forward = block.forward

    def patched(self, x, attn_mask=None):
        x = x + self.drop_path1(self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask)))
        x_attn = x.detach().clone()
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        store.append({
            'x_attn': x_attn,
            'mlp_delta': (x - x_attn).detach().clone(),
        })
        return x

    block.forward = types.MethodType(patched, block)
    block._orig_forward = orig_forward


def _restore_block(block):
    if hasattr(block, '_orig_forward'):
        block.forward = block._orig_forward
        del block._orig_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='PE-Core-L14-336')
    ap.add_argument('--checkpoint', default=None,
                    help='Local checkpoint; default = HF download')
    ap.add_argument('--image', default='./input_imgs/butterfly.png')
    ap.add_argument('--block-indices', type=int, nargs='+',
                    default=[0, 5, 10, 15, 20, 23],
                    help='Transformer block indices to render (PE-Core-L has 24).')
    ap.add_argument('--ncols', type=int, default=3,
                    help='Subplot grid: ncols (rows = ceil(n / ncols)).')
    ap.add_argument('--out-dir', default='./benchmark_results')
    ap.add_argument('--out-name', default=None,
                    help='Basename for output files (no ext). Default = pe_mlp_norm.')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--shared-colors', action='store_true',
                    help='Use one (vmin, vmax) across all panels (raw scale).')
    args = ap.parse_args()

    # ── Load model ────────────────────────────────────────────────────────
    print(f'Loading {args.model} (ckpt={args.checkpoint or "HF default"}) …')
    model = pe.CLIP.from_config(args.model, pretrained=True,
                                 checkpoint_path=args.checkpoint)
    model = model.to(args.device).eval()
    visual = model.visual
    grid = visual.image_size // visual.patch_size
    n_layers = len(visual.transformer.resblocks)
    print(f'  image_size={visual.image_size}  patch={visual.patch_size}  '
          f'grid={grid}×{grid}  layers={n_layers}  cls={visual.use_cls_token}')

    bad = [i for i in args.block_indices if not (0 <= i < n_layers)]
    if bad:
        raise SystemExit(f'block-indices {bad} out of [0, {n_layers}).')

    # ── Image preprocessing — match the eval transform ───────────────────
    transform = pe_transforms.get_image_transform(visual.image_size)
    img = Image.open(args.image).convert('RGB')
    x = transform(img).unsqueeze(0).to(args.device)
    print(f'  input tensor {tuple(x.shape)}  dtype={x.dtype}')

    # ── Hook every block we want to inspect ──────────────────────────────
    captures: Dict[int, List[Dict[str, torch.Tensor]]] = {
        i: [] for i in args.block_indices
    }
    for i in args.block_indices:
        _patch_block_capture(visual.transformer.resblocks[i], captures[i])

    with torch.no_grad():
        _ = model.encode_image(x)

    for i in args.block_indices:
        _restore_block(visual.transformer.resblocks[i])

    # ── Per-token MLP update norm, reshape to grid×grid ──────────────────
    norms: Dict[int, np.ndarray] = {}
    for i in args.block_indices:
        rec = captures[i][0]
        delta = rec['mlp_delta'].float()         # (1, T, C)
        n_tok = delta.shape[1]
        # PE-Core uses cls token at position 0; drop it before reshape.
        if visual.use_cls_token:
            delta = delta[:, 1:, :]
        n_patch = delta.shape[1]
        assert n_patch == grid * grid, \
            f'block {i}: expected {grid*grid} patch tokens, got {n_patch}'
        per_tok = delta.norm(dim=-1)[0].cpu().numpy()
        norms[i] = per_tok.reshape(grid, grid)
        print(f'  block {i:2d}  ‖Δ_MLP‖ ∈ [{per_tok.min():.3f}, {per_tok.max():.3f}]'
              f'  mean={per_tok.mean():.3f}')

    # ── Plot ──────────────────────────────────────────────────────────────
    n = len(args.block_indices)
    ncols = args.ncols
    nrows = (n + ncols - 1) // ncols

    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'axes.titlesize':    16,
        'axes.titleweight':  'semibold',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 3.6 * nrows),
                             squeeze=False)

    # Optionally share color scale across panels.
    if args.shared_colors:
        g_lo = float(min(m.min() for m in norms.values()))
        g_hi = float(max(m.max() for m in norms.values()))

    for k, i in enumerate(args.block_indices):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        m = norms[i]
        lo, hi = (g_lo, g_hi) if args.shared_colors else (m.min(), m.max())
        norm = PowerNorm(gamma=0.5, vmin=lo, vmax=hi) if lo < hi else None
        im = ax.imshow(m, cmap=CYAN_SOLID, norm=norm,
                       interpolation='nearest',
                       vmin=lo if norm is None else None,
                       vmax=hi if norm is None else None)
        ax.set_title(f'block {i}')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis('off')

    fig.suptitle(
        rf'PE  $\|\mathbf{{\Delta}}_{{\mathrm{{MLP}}}}\|_2$  per token  '
        rf'({args.model}, image={os.path.basename(args.image)})',
        fontsize=18,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(args.out_dir, exist_ok=True)
    base = args.out_name or 'pe_mlp_norm'
    for ext in ('png', 'svg', 'pdf'):
        out = os.path.join(args.out_dir, f'{base}.{ext}')
        kwargs = {'dpi': 160} if ext == 'png' else {}
        fig.savefig(out, bbox_inches='tight', **kwargs)
        print(f'  → {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
