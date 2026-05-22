"""Per-patch correlation between sparsesam's keep-score and the MLP /
residual magnitudes at one or more encoder blocks. No decoder involved
— this answers "does sparsesam's importance ranking line up with where
the MLP actually does work, or with where the residual norms are big?".

Per block, captures via hooks:
  * input to attn (post norm1, post window_partition for windowed)
       -> tile_stride_matching's per-group score (z-normalized std + dissim)
       -> broadcast to (64, 64)
  * x_attn (norm2 input)             -> ||residual||₂  per patch
  * mlp(norm2(x_attn))                -> ||MLP update||₂ per patch

Renders one PNG per block (1 x 5 panels):
    keep_score | ||MLP update|| | ||residual||
              | scatter k vs ||MLP|| | scatter k vs ||residual||

Each scatter shows Pearson r and Spearman ρ in the title. The MLP and
residual heatmaps share a colorbar (per-block).

Example:
    python tasks/sam_profile/plot_keep_vs_mlp_res.py \
        --image ./input_imgs/butterfly.png --block-indices 0 5 11 17 23
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'algos', '3rd_party', 'sam-hq'))
sys.path.insert(0, _ROOT)

from segment_anything import sam_model_registry, SamPredictor  # noqa: E402
from segment_anything.modeling.image_encoder import window_unpartition  # noqa: E402
from algos.sparsesam.z_utils import get_z_order  # noqa: E402


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

# Warm palette for the lower (dissimilarity) panels — complements cyan so
# the two metric families are visually distinct.
ORANGE_SOLID = LinearSegmentedColormap.from_list(
    'orange_solid',
    [
        (0.00, '#ffffff'),
        (0.20, '#fff3e0'),
        (0.45, '#ffd180'),
        (0.70, '#ff7043'),
        (0.88, '#bf360c'),
        (1.00, '#3e0000'),
    ],
    N=256,
)


def _str2bool(v):
    return str(v).lower() in ('1', 'true', 'yes', 'y', 't')


def compute_keep_score_map(blk, x_in, group_size: int) -> np.ndarray:
    """Reproduces tile_stride_matching's per-group score on captured input,
    averaged across heads + batch, returned as a (64, 64) raster map.

    Handles both global blocks (x_in shape (1, 64, 64, C)) and windowed
    blocks (x_in shape (1*nw, win, win, C)) — for the latter, score is
    computed per window then window_unpartition stitches back to (64, 64).
    """
    B_, H, W, C = x_in.shape
    nh = blk.attn.num_heads
    D = C // nh
    N = H * W
    assert N % group_size == 0, f'N={N} not divisible by group_size={group_size}'

    # qkv linear weights are half — keep input in original dtype, cast
    # the resulting K to float for the score computation.
    qkv = blk.attn.qkv(x_in.view(B_, N, C))
    qkv = qkv.view(B_, N, 3, nh, D).permute(2, 0, 3, 1, 4)
    _q, k, _v = qkv.reshape(3, B_ * nh, N, D).unbind(0)
    k = k.float()  # cast after qkv

    z_perm = get_z_order(H, W, device=k.device)
    n_groups = N // group_size
    k_z = k[:, z_perm, :]
    k_grp = k_z.view(k.shape[0], n_groups, group_size, D)

    grp_std = k_grp.std(dim=2).mean(dim=-1)
    grp_mean = k_grp.mean(dim=2)
    gm_norm = F.normalize(grp_mean, dim=-1)
    sim = gm_norm @ gm_norm.transpose(1, 2)
    avg_sim = (sim.sum(dim=-1) - 1.0) / max(n_groups - 1, 1)
    dissim = -avg_sim

    eps = 1e-6
    std_n = ((grp_std - grp_std.mean(-1, keepdim=True))
             / (grp_std.std(-1, keepdim=True) + eps))
    dis_n = ((dissim - dissim.mean(-1, keepdim=True))
             / (dissim.std(-1, keepdim=True) + eps))
    grp_score = std_n + dis_n                       # (B_*nh, n_groups)

    # Convert raw scores to per-(head, window) ranks normalized to [0, 1].
    # Higher rank == higher keep priority. This makes scores commensurable
    # across heads AND windows (raw z-scores are standardised within each
    # row, so a "+2" in window A means something different from "+2" in B).
    n = max(n_groups - 1, 1)
    grp_rank = grp_score.argsort(dim=-1).argsort(dim=-1).float() / n

    # Broadcast each group's score back to its `group_size` z-positions,
    # then unpermute to raster -> (B_, nh, H, W); average heads.
    z_score = grp_rank.unsqueeze(-1).expand(-1, -1, group_size).reshape(B_ * nh, N)
    raster = torch.empty_like(z_score)
    raster[:, z_perm] = z_score
    raster = raster.view(B_, nh, H, W).mean(dim=1)  # (B_, H, W)

    if blk.window_size == 0:
        return raster[0].cpu().numpy()

    # Windowed: stitch (B*nw, win, win) -> (B, 64, 64) via window_unpartition.
    H_full = W_full = 64
    ws = blk.window_size
    pad_h = (ws - H_full % ws) % ws
    pad_w = (ws - W_full % ws) % ws
    Hp, Wp = H_full + pad_h, W_full + pad_w
    raster_4d = raster.unsqueeze(-1)                # (B*nw, win, win, 1)
    full = window_unpartition(raster_4d, ws, (Hp, Wp), (H_full, W_full))
    return full[0, :, :, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-type', default='vit_l')
    ap.add_argument('--ckpt', default='./ckts/sam_hq_vit_l.pth')
    ap.add_argument('--image', default='./input_imgs/butterfly.png')
    ap.add_argument('--block-indices', type=int, nargs='+',
                    default=[0, 5, 11, 17, 23])
    ap.add_argument('--source-block-windowed', type=int, default=0,
                    help='Block whose K is used to derive the keep-score for '
                         'all WINDOWED blocks. Production reuses block 0 '
                         '(perm_cache is keyed by (win, ratio, n_block) and '
                         'reset per-image-forward, so the first windowed '
                         'block to fire wins). Override to probe '
                         'counterfactuals.')
    ap.add_argument('--source-block-global', type=int, default=5,
                    help='Block whose K is used to derive the keep-score for '
                         'all GLOBAL blocks. Production reuses block 5 (the '
                         'first global block) for the same reason as above.')
    ap.add_argument('--group-size', type=int, default=4)
    ap.add_argument('--ratio', type=float, default=0.5,
                    help='Keep-ratio for the partial-MLP sparsesam panel '
                         '(top fraction of patches by keep_score that get '
                         'the MLP applied).')
    ap.add_argument('--shared-colors', type=_str2bool, default=False,
                    help='Cross-block-shared colorbar per metric. Default '
                         'False = per-block scaling so within-image variation '
                         'stays visible.')
    ap.add_argument('--out-dir', default='./benchmark_results')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt).to(device).eval().half()

    block_idxs = list(args.block_indices)
    src_w = args.source_block_windowed
    src_g = args.source_block_global
    src_blocks = {src_w, src_g}

    # Capture x_in for plotted blocks AND for source blocks (so
    # `compute_keep_score_map` has K for the source even if the user did
    # not request to plot it).
    capture_idxs = sorted(set(block_idxs) | src_blocks)
    enc_caps = {i: {} for i in capture_idxs}
    handles = []
    for i in capture_idxs:
        blk = sam.image_encoder.blocks[i]

        def make_attn_pre(idx):
            def pre(_mod, inputs):
                enc_caps[idx]['x_in'] = inputs[0].detach().clone()
            return pre
        handles.append(blk.attn.register_forward_pre_hook(make_attn_pre(i)))

        if i in block_idxs:
            def make_norm2_pre(idx):
                def pre(_mod, inputs):
                    enc_caps[idx]['x_attn'] = inputs[0].detach().clone()
                return pre
            def make_mlp_post(idx):
                def post(_mod, _inp, output):
                    enc_caps[idx]['mlp_delta'] = output.detach().clone()
                return post
            handles.append(blk.norm2.register_forward_pre_hook(make_norm2_pre(i)))
            handles.append(blk.mlp.register_forward_hook(make_mlp_post(i)))

    # `--source-block-windowed` may be a global block (counterfactual:
    # apply the first-global keep_score to every windowed block instead
    # of production's "first-windowed wins" behavior). The resulting
    # (64, 64) global pattern is broadcast to all windowed blocks.
    if sam.image_encoder.blocks[src_g].window_size != 0:
        raise SystemExit(
            f'--source-block-global={src_g} is a windowed block. Pick a '
            f'global block (win=0).'
        )

    img = np.array(Image.open(args.image).convert('RGB'))
    predictor = SamPredictor(sam)
    predictor.set_image(img)               # runs the encoder forward
    for h in handles:
        h.remove()

    # Compute the two source keep-scores ONCE — these are the maps
    # production reuses across all blocks of the same window class.
    keep_src_w = compute_keep_score_map(
        sam.image_encoder.blocks[src_w],
        enc_caps[src_w]['x_in'],
        args.group_size,
    )
    keep_src_g = compute_keep_score_map(
        sam.image_encoder.blocks[src_g],
        enc_caps[src_g]['x_in'],
        args.group_size,
    )

    rows = []
    for i in block_idxs:
        blk = sam.image_encoder.blocks[i]
        # Apply the SAME keep-score that production would reuse here.
        keep_map = keep_src_g if blk.window_size == 0 else keep_src_w
        keep_src_idx = src_g if blk.window_size == 0 else src_w
        x_attn = enc_caps[i]['x_attn'].float()
        mlp_delta = enc_caps[i]['mlp_delta'].float()
        res_mag = x_attn.norm(dim=-1)[0].cpu().numpy()
        mlp_mag = mlp_delta.norm(dim=-1)[0].cpu().numpy()

        # Baseline post-block output: x_attn + mlp(norm2(x_attn)).
        baseline_out = x_attn + mlp_delta

        def per_token_cossim(x_4d):
            """Per-patch mean cosine similarity to all other patches in
            the (1, H, W, C) tensor; returns (H, W) numpy."""
            _, H, W, C = x_4d.shape
            x = x_4d[0].reshape(H * W, C)
            xn = F.normalize(x, dim=-1)
            sim = xn @ xn.T
            avg_sim = (sim.sum(dim=-1) - 1.0) / max(H * W - 1, 1)
            return avg_sim.reshape(H, W).cpu().numpy()

        baseline_cossim = per_token_cossim(baseline_out)

        # Production-faithful "patches actually updated by sparsesam":
        #   * windowed blocks  -> top `ratio` patches by keep_score get MLP
        #   * global blocks    -> ALL patches get MLP (production runs full
        #     MLP on globals; partial-MLP applies only when window_size>0)
        N = keep_map.size
        if blk.window_size == 0:
            keep_mask_2d = np.ones(keep_map.shape, dtype=bool)
        else:
            keep_n = max(1, int(round(args.ratio * N)))
            flat_keep = keep_map.reshape(-1)
            top_idx = np.argpartition(-flat_keep, keep_n - 1)[:keep_n]
            mask = np.zeros(N, dtype=bool)
            mask[top_idx] = True
            keep_mask_2d = mask.reshape(keep_map.shape)

        # Partial-MLP output tensor: kept tokens get full MLP, others stay
        # at x_attn. Compute the same per-patch dissimilarity on it.
        keep_t = torch.from_numpy(keep_mask_2d).to(baseline_out.device)
        partial_out = torch.where(
            keep_t.unsqueeze(0).unsqueeze(-1), baseline_out, x_attn,
        )
        partial_cossim = per_token_cossim(partial_out)

        def corr(a, b):
            an = (a - a.mean()) / (a.std() + 1e-9)
            bn = (b - b.mean()) / (b.std() + 1e-9)
            return float((an * bn).mean())

        def spearman(a, b):
            ar = np.argsort(np.argsort(a))
            br = np.argsort(np.argsort(b))
            return corr(ar.astype(np.float64), br.astype(np.float64))

        k_flat = keep_map.reshape(-1)
        m_flat = mlp_mag.reshape(-1)
        r_flat = res_mag.reshape(-1)

        rows.append(dict(
            block=i, keep_src=keep_src_idx,
            keep=keep_map, keep_mask=keep_mask_2d.astype(np.float32),
            mlp=mlp_mag, res=res_mag,
            baseline_cossim=baseline_cossim, partial_cossim=partial_cossim,
            r_mlp=corr(k_flat, m_flat), rho_mlp=spearman(k_flat, m_flat),
            r_res=corr(k_flat, r_flat), rho_res=spearman(k_flat, r_flat),
        ))

    # Per-block (lo, hi) covering both ||MLP|| and ||res||.
    for r in rows:
        r['mr_lo'] = float(min(r['mlp'].min(), r['res'].min()))
        r['mr_hi'] = float(max(r['mlp'].max(), r['res'].max()))
        r['k_lo'] = float(r['keep'].min())
        r['k_hi'] = float(r['keep'].max())
    if args.shared_colors:
        g_mr_lo = min(r['mr_lo'] for r in rows)
        g_mr_hi = max(r['mr_hi'] for r in rows)
        g_k_lo = min(r['k_lo'] for r in rows)
        g_k_hi = max(r['k_hi'] for r in rows)
        for r in rows:
            r['mr_lo'], r['mr_hi'] = g_mr_lo, g_mr_hi
            r['k_lo'], r['k_hi'] = g_k_lo, g_k_hi

    def heatmap(ax, m, title, lo, hi, cb_label, cmap=CYAN_SOLID):
        norm = PowerNorm(gamma=0.35, vmin=lo, vmax=hi) if lo < hi else None
        im = ax.imshow(m, cmap=cmap, norm=norm,
                       interpolation='nearest', vmin=lo if norm is None else None,
                       vmax=hi if norm is None else None)
        ax.set_title(title, fontsize=18)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    os.makedirs(args.out_dir, exist_ok=True)
    saved = []
    for ri, row in enumerate(rows):
        i = row['block']
        win = sam.image_encoder.blocks[i].window_size
        layout = 'global' if win == 0 else f'win={win}'
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))

        heatmap(axes[0, 0], row['keep_mask'],
                r'MLP applied $\mathbf{X}_k$ mask',
                0.0, 1.0, '')
        heatmap(axes[0, 1], row['mlp'],
                r'MLP update norm $\|\mathbf{\Delta}_{\mathrm{MLP}}\|^2$',
                float(row['mlp'].min()), float(row['mlp'].max()), '')
        d_lo = float(min(row['baseline_cossim'].min(), row['partial_cossim'].min()))
        d_hi = float(max(row['baseline_cossim'].max(), row['partial_cossim'].max()))
        heatmap(axes[1, 0], row['baseline_cossim'],
                'dissimilarity map\ndense MLP',
                d_lo, d_hi, '', cmap=ORANGE_SOLID)
        heatmap(axes[1, 1], row['partial_cossim'],
                'dissimilarity map\nresidual-consistent MLP',
                d_lo, d_hi, '', cmap=ORANGE_SOLID)
        fig.suptitle(f'block {i}', fontsize=20)

        out = os.path.join(args.out_dir, f'keep_and_mlp_b{i:02d}.png')
        plt.tight_layout()
        plt.savefig(out, dpi=140, bbox_inches='tight')
        plt.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
        plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close(fig)
        saved.append(out)

    print(f'wrote {len(saved)} files to {args.out_dir}/')
    print('block  ||MLP||              ||res||              keep_score          '
          'r(k,MLP)  ρ(k,MLP)  r(k,res)  ρ(k,res)')
    for r in rows:
        print(f'  {r["block"]:>2}  '
              f'[{r["mlp"].min():6.2f},{r["mlp"].max():6.2f}]  '
              f'[{r["res"].min():6.2f},{r["res"].max():6.2f}]  '
              f'[{r["keep"].min():+6.2f},{r["keep"].max():+6.2f}]  '
              f'{r["r_mlp"]:+8.3f}  {r["rho_mlp"]:+8.3f}  '
              f'{r["r_res"]:+8.3f}  {r["rho_res"]:+8.3f}')


if __name__ == '__main__':
    main()
