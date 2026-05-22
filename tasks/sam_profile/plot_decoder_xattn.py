"""Plot SAM-HQ mask-decoder cross-attention from each output token onto
the image. Captures softmax(QK^T) of the *cross_attn_token_to_image*
inside every TwoWayAttentionBlock plus the final token→image attention,
reshapes each token's row to 64x64, and overlays it on the input.

Token layout (after sam-hq mask_decoder_hq.py:168):
    0      IoU
    1..4   SAM mask tokens (full + 3 multimask candidates)
    5      HQ token (HF)
    6..N   prompt point/box embeddings

Default plots one panel per mask/HQ token from the final attention layer
(closest to mask prediction). Pass --all-layers for a 3xN grid showing
the evolution across the two TwoWayAttentionBlocks + the final attention.

Example:
    python tasks/sam_profile/plot_decoder_xattn.py \
        --image ./input_imgs/butterfly.png --point 700,500 \
        --out ./benchmark_results/decoder_xattn.png
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'algos', '3rd_party', 'sam-hq'))

from segment_anything import sam_model_registry, SamPredictor  # noqa: E402


def _build_cyan_overlay() -> LinearSegmentedColormap:
    """Cyan→magenta overlay with baked-in alpha — same as autoresearch."""
    N = 256
    rgba = np.zeros((N, 4))
    rgb_stops = np.array([
        (0.55, 0.96, 1.00),
        (0.05, 0.85, 0.95),
        (0.30, 0.45, 0.92),
        (0.95, 0.18, 0.65),
    ])
    rgb_pos = np.array([0.0, 0.45, 0.78, 1.0])
    t = np.linspace(0, 1, N)
    for ch in range(3):
        rgba[:, ch] = np.interp(t, rgb_pos, rgb_stops[:, ch])
    rgba[:, 3] = np.clip((t - 0.05) / (1 - 0.05), 0.0, 1.0) ** 0.55 * 0.97
    return LinearSegmentedColormap.from_list('cyan_overlay', rgba, N=N)


CYAN_OVERLAY = _build_cyan_overlay()


def install_attn_capture(modules):
    """Monkey-patch each `Attention` module's forward to also stash softmax
    weights into a per-module list. Returns (captures, restore_fn)."""
    captures = {id(m): [] for m in modules}
    originals = {id(m): m.forward for m in modules}

    def make_fwd(mod):
        def fwd(q, k, v):
            qp = mod._separate_heads(mod.q_proj(q), mod.num_heads)
            kp = mod._separate_heads(mod.k_proj(k), mod.num_heads)
            vp = mod._separate_heads(mod.v_proj(v), mod.num_heads)
            d = qp.shape[-1]
            attn = (qp @ kp.transpose(-2, -1)) / math.sqrt(d)
            attn = attn.softmax(dim=-1)
            captures[id(mod)].append(attn.detach().cpu())
            out = mod._recombine_heads(attn @ vp)
            return mod.out_proj(out)
        return fwd

    for m in modules:
        # Assign as instance attribute (not method) so we don't have to
        # accept `self` — the closure already captures `mod`.
        m.forward = make_fwd(m)

    def restore():
        for m in modules:
            m.forward = originals[id(m)]
    return captures, restore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-type', default='vit_l')
    ap.add_argument('--ckpt', default='./ckts/sam_hq_vit_l.pth')
    ap.add_argument('--image', default='./input_imgs/butterfly.png')
    ap.add_argument('--point', default='700,500',
                    help='Foreground point "x,y" in input-image coords.')
    ap.add_argument('--out', default='./benchmark_results/decoder_xattn.png')
    ap.add_argument('--head-agg', default='mean', choices=['mean', 'max'])
    ap.add_argument('--all-layers', action='store_true',
                    help='Plot all decoder layers (rows) not just the final.')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    # SAM-HQ's prompt_encoder hard-codes fp16 (prompt_encoder.py:196), so
    # the whole model has to run in half precision for shapes/dtypes to
    # line up through the cross-attention path.
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt).to(device).eval().half()
    transformer = sam.mask_decoder.transformer

    cross_modules = [lyr.cross_attn_token_to_image for lyr in transformer.layers]
    cross_modules.append(transformer.final_attn_token_to_image)
    captures, restore = install_attn_capture(cross_modules)

    img = np.array(Image.open(args.image).convert('RGB'))
    H_img, W_img = img.shape[:2]
    px, py = [int(v) for v in args.point.split(',')]

    predictor = SamPredictor(sam)
    predictor.set_image(img)
    pt = np.array([[px, py]])
    label = np.array([1])
    with torch.inference_mode():
        _masks, ious, _low = predictor.predict(
            point_coords=pt, point_labels=label,
            multimask_output=True, hq_token_only=False,
        )
    restore()

    # One forward → one capture per module. (B=1, H, num_tokens, 4096).
    layer_attn = []
    for m in cross_modules:
        a = captures[id(m)][0]
        a = a.mean(dim=1) if args.head_agg == 'mean' else a.max(dim=1).values
        layer_attn.append(a[0])  # (num_tokens, 4096)

    token_idx = [1, 2, 3, 4, 5]
    token_names = ['mask 0', 'mask 1', 'mask 2', 'mask 3', 'HQ']
    layer_names = [f'layer {i}' for i in range(len(transformer.layers))] + ['final']

    rows = layer_attn if args.all_layers else [layer_attn[-1]]
    row_labels = layer_names if args.all_layers else [layer_names[-1]]

    n_rows = len(rows)
    n_cols = len(token_idx)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.6 * n_cols, 4.8 * n_rows),
                             squeeze=False)

    for ri, attn in enumerate(rows):
        for ci, ti in enumerate(token_idx):
            ax = axes[ri, ci]
            a = attn[ti].float().numpy().reshape(64, 64)
            # Power curve spreads the heavy-tailed distribution so the
            # overlay isn't just one bright pixel on a flat background.
            a_norm = (a / (a.max() + 1e-9)) ** 0.35
            ax.imshow(img, zorder=1)
            ax.imshow(
                a_norm, cmap=CYAN_OVERLAY, vmin=0, vmax=1,
                extent=(0, W_img, H_img, 0),
                interpolation='bilinear', zorder=2,
            )
            ax.scatter([px], [py], marker='*', s=420, c='#ffe600',
                       edgecolors='black', linewidths=1.6, zorder=10)
            title = (f'{row_labels[ri]} — {token_names[ci]}'
                     if n_rows > 1 else token_names[ci])
            ax.set_title(title, fontsize=18)
            ax.set_xticks([]); ax.set_yticks([])

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches='tight')
    iou_str = ', '.join(f'{i:.3f}' for i in ious)
    print(f'saved -> {args.out}  (predicted IoUs: [{iou_str}])')


if __name__ == '__main__':
    main()
