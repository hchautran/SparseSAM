"""Per-patch L2 norm of the MLP update at one or more encoder blocks
compared to the residual it sits on, plotted alongside the mask-decoder
cross-attention each patch receives.

For every chosen block, captures via hooks:
  * x_attn (norm2 input)            -> ||residual||₂  per patch  (64x64)
  * mlp(norm2(x_attn))               -> ||MLP update||₂ per patch
  * decoder cross_attn_token_to_image (final layer, mask + HQ tokens summed,
    averaged over heads)             -> decoder attention per patch

Renders a (n_blocks x 4) grid: one row per block, columns =
||MLP update||, ||residual||, signed difference, scatter-with-correlation.

Color scales are SHARED across rows for each map column so "magenta" means
the same magnitude in block 0 as in block 23. Pass --shared-colors=False
to revert to per-row scaling.

Example:
    python tasks/sam_profile/plot_mlp_update_vs_decoder.py \
        --block-indices 0 5 11 17 23 \
        --out ./benchmark_results/mlp_update_vs_decoder.png
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'algos', '3rd_party', 'sam-hq'))

from segment_anything import sam_model_registry, SamPredictor  # noqa: E402


def _build_cyan_overlay() -> LinearSegmentedColormap:
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


def install_decoder_capture(modules):
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
        m.forward = make_fwd(m)

    def restore():
        for m in modules:
            m.forward = originals[id(m)]
    return captures, restore


def _str2bool(v):
    return str(v).lower() in ('1', 'true', 'yes', 'y', 't')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-type', default='vit_l')
    ap.add_argument('--ckpt', default='./ckts/sam_hq_vit_l.pth')
    ap.add_argument('--image', default='./input_imgs/butterfly.png')
    ap.add_argument('--point', default='700,500')
    ap.add_argument('--block-indices', type=int, nargs='+',
                    default=[0, 5, 11, 17, 23],
                    help='Blocks to inspect. vit_l globals: 5,11,17,23.')
    ap.add_argument('--head-agg', default='mean', choices=['mean', 'max'])
    ap.add_argument('--shared-colors', type=_str2bool, default=False,
                    help='Share color scale across blocks per metric. Default '
                         'False so each panel spans its own range — within-block '
                         'variation stays visible. Set True for cross-block '
                         'magnitude comparison (read absolute values off the '
                         'colorbars in either case).')
    ap.add_argument('--out-dir', default='./benchmark_results',
                    help='Directory to write per-block figures into. '
                         'One file per block: mlp_update_vs_decoder_b<i>.png.')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt).to(device).eval().half()

    block_idxs = list(args.block_indices)
    enc_blocks = [sam.image_encoder.blocks[i] for i in block_idxs]

    enc_caps = {i: {} for i in block_idxs}
    handles = []
    for i, blk in zip(block_idxs, enc_blocks):
        def make_pre(idx):
            def pre(_mod, inputs):
                enc_caps[idx]['x_attn'] = inputs[0].detach().clone()
            return pre
        def make_post(idx):
            def post(_mod, _inp, output):
                enc_caps[idx]['mlp_delta'] = output.detach().clone()
            return post
        handles.append(blk.norm2.register_forward_pre_hook(make_pre(i)))
        handles.append(blk.mlp.register_forward_hook(make_post(i)))

    transformer = sam.mask_decoder.transformer
    cross_modules = [lyr.cross_attn_token_to_image for lyr in transformer.layers]
    cross_modules.append(transformer.final_attn_token_to_image)
    dec_cap, restore = install_decoder_capture(cross_modules)

    img = np.array(Image.open(args.image).convert('RGB'))
    H_img, W_img = img.shape[:2]
    px, py = [int(v) for v in args.point.split(',')]
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    with torch.inference_mode():
        _masks, ious, _ = predictor.predict(
            point_coords=np.array([[px, py]]),
            point_labels=np.array([1]),
            multimask_output=True, hq_token_only=False,
        )
    for h in handles:
        h.remove()
    restore()

    # Decoder map shared across rows.
    final_attn = dec_cap[id(cross_modules[-1])][0]
    if args.head_agg == 'mean':
        final_attn = final_attn.mean(dim=1)
    else:
        final_attn = final_attn.max(dim=1).values
    decoder_per_token = final_attn[0].float()
    decoder_map = decoder_per_token[1:6].sum(dim=0).numpy().reshape(64, 64)
    d_flat = decoder_map.reshape(-1)

    # Per-block metrics.  Encoder metric is the per-patch MLP update
    # magnitude itself: ||(x_attn + mlp_delta) − x_attn||₂ = ||mlp_delta||₂.
    rows = []
    for i in block_idxs:
        x_attn = enc_caps[i]['x_attn'].float()
        mlp_delta = enc_caps[i]['mlp_delta'].float()
        res_mag = x_attn.norm(dim=-1)[0].cpu().numpy()
        mlp_mag = mlp_delta.norm(dim=-1)[0].cpu().numpy()

        e_flat = mlp_mag.reshape(-1)
        e_n = (e_flat - e_flat.mean()) / (e_flat.std() + 1e-9)
        d_n = (d_flat - d_flat.mean()) / (d_flat.std() + 1e-9)
        pearson = float((e_n * d_n).mean())
        e_rank = np.argsort(np.argsort(e_flat))
        d_rank = np.argsort(np.argsort(d_flat))
        er = (e_rank - e_rank.mean()) / (e_rank.std() + 1e-9)
        dr = (d_rank - d_rank.mean()) / (d_rank.std() + 1e-9)
        spearman = float((er * dr).mean())

        rows.append(dict(
            block=i, mlp=mlp_mag, res=res_mag,
            pearson=pearson, spearman=spearman,
        ))

    # Per-block (lo, hi) covering BOTH ||MLP|| and ||residual|| so the
    # two overlays in a panel share one colorbar — magnitudes are
    # directly comparable side-by-side.
    for r in rows:
        r['lo'] = float(min(r['mlp'].min(), r['res'].min()))
        r['hi'] = float(max(r['mlp'].max(), r['res'].max()))

    if args.shared_colors:
        g_lo = min(r['lo'] for r in rows)
        g_hi = max(r['hi'] for r in rows)
        for r in rows:
            r['lo'], r['hi'] = g_lo, g_hi

    # Solid (no-alpha) cyan colormap for stand-alone heatmaps.
    CYAN_SOLID = LinearSegmentedColormap.from_list(
        'cyan_solid',
        [
            (0.00, '#ffffff'),
            (0.20, '#e0fbff'),
            (0.45, '#7ff7ff'),
            (0.70, '#00bcd4'),
            (0.88, '#005f73'),
            (1.00, '#001f2e'),
        ],
        N=256,
    )

    def overlay(ax, m: np.ndarray, title: str, cb_label: str,
                lo: float | None, hi: float | None):
        v_lo = float(m.min()) if lo is None else float(lo)
        v_hi = float(m.max()) if hi is None else float(hi)
        norm = PowerNorm(gamma=0.35, vmin=v_lo, vmax=v_hi)
        im = ax.imshow(m, cmap=CYAN_SOLID, norm=norm,
                       interpolation='nearest')
        ax.set_title(title, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=cb_label)

    os.makedirs(args.out_dir, exist_ok=True)
    saved = []
    for ri, row in enumerate(rows):
        i = row['block']
        win = enc_blocks[ri].window_size
        layout = 'global' if win == 0 else f'win={win}'

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        overlay(axes[0], row['mlp'],
                f'block {i} ({layout}) — ||MLP update||',
                'L2 norm', row['lo'], row['hi'])
        overlay(axes[1], row['res'],
                f'block {i} ({layout}) — ||residual||',
                'L2 norm', row['lo'], row['hi'])

        ax = axes[2]
        ax.scatter(row['mlp'].reshape(-1), d_flat,
                   s=8, alpha=0.4, c='#117a8b',
                   edgecolors='none', rasterized=True)
        ax.set_xlabel('||MLP update||', fontsize=12)
        ax.set_ylabel('decoder attn (mask+HQ)', fontsize=12)
        ax.set_title(f'block {i}   r={row["pearson"]:+.3f}  ρ={row["spearman"]:+.3f}',
                     fontsize=14)
        ax.grid(alpha=0.3)

        out_path = os.path.join(args.out_dir, f'mlp_update_vs_decoder_b{i:02d}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        saved.append(out_path)

    iou_str = ', '.join(f'{i:.3f}' for i in ious)
    print(f'wrote {len(saved)} files to {args.out_dir}/   IoUs=[{iou_str}]')
    for p in saved:
        print(f'  {p}')
    print(f'shared color ranges (across {len(rows)} blocks):')
    if args.shared_colors:
        print(f'  shared (mlp ∪ res)  ∈ [{rows[0]["lo"]:.2f}, {rows[0]["hi"]:.2f}]')
    print('block  ||MLP||              ||res||              r       rho')
    for r in rows:
        print(f'  {r["block"]:>2}  '
              f'[{r["mlp"].min():6.2f},{r["mlp"].max():6.2f}]  '
              f'[{r["res"].min():6.2f},{r["res"].max():6.2f}]  '
              f'{r["pearson"]:+.3f}  {r["spearman"]:+.3f}')


if __name__ == '__main__':
    main()
