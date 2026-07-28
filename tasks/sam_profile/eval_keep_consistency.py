"""Per-block ρ(keep_score, MLP) and ρ(keep_score, residual) measured
across N images from a dataset (default: DIS5K-VD), so we can tell
whether the single-image numbers in `plot_keep_vs_mlp_res.py` are
consistent or sample-dependent.

Iterates over images, runs encoder forward once each, computes the same
per-block keep_score / MLP / residual maps as the plot script, and
reports per-block (Pearson r, Spearman ρ) mean ± std across the dataset.

Outputs:
  ./benchmark_results/keep_consistency.tsv      — per-(image, block) rows
  ./benchmark_results/keep_consistency.png      — block-vs-ρ summary plot
                                                   with shaded ±1 std band

Example:
    python tasks/sam_profile/eval_keep_consistency.py \
        --num-images 50 --source-block-windowed 5 \
        --out-tsv ./benchmark_results/keep_consistency_src5.tsv \
        --out-plot ./benchmark_results/keep_consistency_src5.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'algos', '3rd_party', 'sam-hq'))
sys.path.insert(0, _ROOT)

from segment_anything import sam_model_registry, SamPredictor  # noqa: E402
from segment_anything.modeling.image_encoder import window_unpartition  # noqa: E402
from algos.sparsesam.z_utils import get_z_order  # noqa: E402


def compute_keep_score_map(blk, x_in, group_size: int) -> np.ndarray:
    B_, H, W, C = x_in.shape
    nh = blk.attn.num_heads
    D = C // nh
    N = H * W
    qkv = blk.attn.qkv(x_in.view(B_, N, C))
    qkv = qkv.view(B_, N, 3, nh, D).permute(2, 0, 3, 1, 4)
    _q, k, _v = qkv.reshape(3, B_ * nh, N, D).unbind(0)
    k = k.float()

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
    grp_score = std_n + dis_n
    n = max(n_groups - 1, 1)
    grp_rank = grp_score.argsort(dim=-1).argsort(dim=-1).float() / n

    z_score = grp_rank.unsqueeze(-1).expand(-1, -1, group_size).reshape(B_ * nh, N)
    raster = torch.empty_like(z_score)
    raster[:, z_perm] = z_score
    raster = raster.view(B_, nh, H, W).mean(dim=1)

    if blk.window_size == 0:
        return raster[0].cpu().numpy()
    H_full = W_full = 64
    ws = blk.window_size
    pad_h = (ws - H_full % ws) % ws
    pad_w = (ws - W_full % ws) % ws
    Hp, Wp = H_full + pad_h, W_full + pad_w
    raster_4d = raster.unsqueeze(-1)
    full = window_unpartition(raster_4d, ws, (Hp, Wp), (H_full, W_full))
    return full[0, :, :, 0].cpu().numpy()


def pearson(a, b):
    an = (a - a.mean()) / (a.std() + 1e-9)
    bn = (b - b.mean()) / (b.std() + 1e-9)
    return float((an * bn).mean())


def spearman(a, b):
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(ar, br)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-type', default='vit_l')
    ap.add_argument('--ckpt', default='./ckts/sam_hq_vit_l.pth')
    ap.add_argument('--image-dir', default='./data/DIS5K/DIS-VD/im')
    ap.add_argument('--num-images', type=int, default=50)
    ap.add_argument('--source-block-windowed', type=int, default=5,
                    help='Default 5 (counterfactual): apply the first global '
                         "block's smooth keep_score to every windowed block. "
                         'Pass 0 to recover production behavior.')
    ap.add_argument('--source-block-global', type=int, default=5)
    ap.add_argument('--block-indices', type=int, nargs='+',
                    default=list(range(24)))
    ap.add_argument('--group-size', type=int, default=4)
    ap.add_argument('--out-tsv', default='./benchmark_results/keep_consistency.tsv')
    ap.add_argument('--out-plot', default='./benchmark_results/keep_consistency.png')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt).to(device).eval().half()
    encoder = sam.image_encoder

    if encoder.blocks[args.source_block_global].window_size != 0:
        raise SystemExit('--source-block-global must be a global block (win=0).')
    src_w, src_g = args.source_block_windowed, args.source_block_global
    block_idxs = list(args.block_indices)
    capture_idxs = sorted(set(block_idxs) | {src_w, src_g})

    image_dir = Path(args.image_dir)
    paths = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))[:args.num_images]
    if not paths:
        raise SystemExit(f'no images found in {image_dir}')
    print(f'evaluating on {len(paths)} images from {image_dir}')

    predictor = SamPredictor(sam)

    rows_per_image = []   # list of dicts: {block, image, r_mlp, rho_mlp, r_res, rho_res}
    for img_idx, img_path in enumerate(paths):
        enc_caps = {i: {} for i in capture_idxs}
        handles = []
        for i in capture_idxs:
            blk = encoder.blocks[i]

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

        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            with torch.inference_mode():
                predictor.set_image(img)
        except Exception as e:
            print(f'  [{img_idx+1}/{len(paths)}] skipping {img_path.name}: {e}')
            for h in handles:
                h.remove()
            continue
        for h in handles:
            h.remove()

        keep_src_w = compute_keep_score_map(
            encoder.blocks[src_w], enc_caps[src_w]['x_in'], args.group_size,
        )
        keep_src_g = compute_keep_score_map(
            encoder.blocks[src_g], enc_caps[src_g]['x_in'], args.group_size,
        )

        for i in block_idxs:
            blk = encoder.blocks[i]
            keep_map = keep_src_g if blk.window_size == 0 else keep_src_w
            x_attn = enc_caps[i]['x_attn'].float()
            mlp_delta = enc_caps[i]['mlp_delta'].float()
            res_mag = x_attn.norm(dim=-1)[0].cpu().numpy()
            mlp_mag = mlp_delta.norm(dim=-1)[0].cpu().numpy()

            k = keep_map.reshape(-1)
            m = mlp_mag.reshape(-1)
            r = res_mag.reshape(-1)
            rows_per_image.append(dict(
                image=img_path.name, block=i,
                r_mlp=pearson(k, m), rho_mlp=spearman(k, m),
                r_res=pearson(k, r), rho_res=spearman(k, r),
            ))

        del enc_caps
        if (img_idx + 1) % 10 == 0:
            print(f'  [{img_idx+1}/{len(paths)}] done')

    # ── write TSV ──
    os.makedirs(os.path.dirname(args.out_tsv) or '.', exist_ok=True)
    with open(args.out_tsv, 'w') as f:
        f.write('image\tblock\tr_mlp\trho_mlp\tr_res\trho_res\n')
        for r in rows_per_image:
            f.write(f'{r["image"]}\t{r["block"]}\t'
                    f'{r["r_mlp"]:.4f}\t{r["rho_mlp"]:.4f}\t'
                    f'{r["r_res"]:.4f}\t{r["rho_res"]:.4f}\n')
    print(f'\nwrote {args.out_tsv}  ({len(rows_per_image)} rows)')

    # ── per-block aggregation ──
    by_block = {}
    for r in rows_per_image:
        by_block.setdefault(r['block'], []).append(r)
    blocks_sorted = sorted(by_block.keys())
    summary = []
    for b in blocks_sorted:
        rs = by_block[b]
        s = dict(
            block=b, n=len(rs),
            mlp_mean=float(np.mean([x['rho_mlp'] for x in rs])),
            mlp_std=float(np.std([x['rho_mlp'] for x in rs])),
            res_mean=float(np.mean([x['rho_res'] for x in rs])),
            res_std=float(np.std([x['rho_res'] for x in rs])),
        )
        summary.append(s)

    print('\n  block  n   ρ(k,MLP) mean±std    ρ(k,res) mean±std')
    for s in summary:
        print(f'   {s["block"]:>2}  {s["n"]:>3}   '
              f'{s["mlp_mean"]:+.3f} ± {s["mlp_std"]:.3f}     '
              f'{s["res_mean"]:+.3f} ± {s["res_std"]:.3f}')

    # ── plot ──
    blocks = np.array([s['block'] for s in summary])
    mlp_m = np.array([s['mlp_mean'] for s in summary])
    mlp_s = np.array([s['mlp_std']  for s in summary])
    res_m = np.array([s['res_mean'] for s in summary])
    res_s = np.array([s['res_std']  for s in summary])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(blocks, mlp_m, '-o', color='#117a8b', label='ρ(keep, ||MLP update||)')
    ax.fill_between(blocks, mlp_m - mlp_s, mlp_m + mlp_s,
                    color='#117a8b', alpha=0.2)
    ax.plot(blocks, res_m, '-s', color='#d62728', label='ρ(keep, ||residual||)')
    ax.fill_between(blocks, res_m - res_s, res_m + res_s,
                    color='#d62728', alpha=0.2)
    # Mark global blocks for context.
    for b in blocks:
        if encoder.blocks[int(b)].window_size == 0:
            ax.axvline(b, color='grey', alpha=0.25, linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xlabel('block index')
    ax.set_ylabel('Spearman ρ')
    ax.set_title(
        f'Per-block keep-score correlation across {len(paths)} images   '
        f'(src_windowed=block {src_w}, src_global=block {src_g})',
        fontsize=12,
    )
    ax.set_xticks(blocks)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(args.out_plot) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=140, bbox_inches='tight')
    print(f'wrote {args.out_plot}')


if __name__ == '__main__':
    main()
