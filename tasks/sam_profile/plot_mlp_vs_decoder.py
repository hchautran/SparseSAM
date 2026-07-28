"""Correlate sparsesam's per-token "MLP-update importance" with the
mask-decoder cross-attention each patch receives.

For one global encoder block (default 23, the last global = encoder
output tap), capture the input to attention, recompute K via the qkv
linear, and run the same per-group score `tile_stride_matching` uses to
rank tokens (z-normalized std + dissimilarity, averaged across heads).
Broadcast each group's score to its `group_size` member patches and
unpermute back to raster → continuous 64x64 "importance" map.

Separately monkey-patch the decoder's `cross_attn_token_to_image` to
capture softmax weights, sum across mask + HQ tokens (indices 1-5),
average heads → 64x64 "decoder attention received" map.

Plot 1x3:  keep-score overlay  |  decoder-attn overlay  |  scatter.

Example:
    python tasks/sam_profile/plot_mlp_vs_decoder.py \
        --image ./input_imgs/butterfly.png --point 700,500 \
        --block-idx 23 --out ./benchmark_results/mlp_vs_decoder.png
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'algos', '3rd_party', 'sam-hq'))
sys.path.insert(0, _ROOT)

from segment_anything import sam_model_registry, SamPredictor  # noqa: E402
from algos.sparsesam.z_utils import get_z_order  # noqa: E402


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
    """Monkey-patch the decoder Attention modules to record softmax weights."""
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


def compute_keep_score_map(blk, x, group_size: int) -> np.ndarray:
    """Reproduces tile_stride_matching's per-group score on captured K and
    returns a (H, W) raster map averaged across heads.

    Higher value -> sparsesam keeps this group first (more "important")."""
    B_, H, W, C = x.shape
    nh = blk.attn.num_heads
    D = C // nh
    N = H * W
    assert N % group_size == 0

    qkv = blk.attn.qkv(x.view(B_, N, C))
    qkv = qkv.view(B_, N, 3, nh, D).permute(2, 0, 3, 1, 4)
    _q, k, _v = qkv.reshape(3, B_ * nh, N, D).unbind(0)
    k = k.float()

    z_perm = get_z_order(H, W, device=k.device)
    n_groups = N // group_size
    k_z = k[:, z_perm, :]
    k_grp = k_z.view(k.shape[0], n_groups, group_size, D)

    grp_std = k_grp.std(dim=2).mean(dim=-1)            # (BH, n_groups)
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
    grp_score = std_n + dis_n                            # (BH, n_groups)

    # Average across heads + batch -> single (n_groups,)
    grp_score = grp_score.view(B_, nh, n_groups).mean(dim=(0, 1))

    # Broadcast each group's score to its group_size member z-positions,
    # then un-permute to raster (H, W).
    z_score = grp_score.unsqueeze(-1).expand(-1, group_size).reshape(N)
    raster = torch.empty_like(z_score)
    raster[z_perm] = z_score                           # invert the perm
    return raster.view(H, W).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-type', default='vit_l')
    ap.add_argument('--ckpt', default='./ckts/sam_hq_vit_l.pth')
    ap.add_argument('--image', default='./input_imgs/butterfly.png')
    ap.add_argument('--point', default='700,500')
    ap.add_argument('--block-idx', type=int, default=23,
                    help='Global block to score (vit_l globals: 5,11,17,23).')
    ap.add_argument('--group-size', type=int, default=4)
    ap.add_argument('--head-agg', default='mean', choices=['mean', 'max'])
    ap.add_argument('--out', default='./benchmark_results/mlp_vs_decoder.png')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt).to(device).eval().half()

    # Encoder hook: input to chosen block's attn (post norm1, pre qkv).
    blk = sam.image_encoder.blocks[args.block_idx]
    if blk.window_size != 0:
        raise SystemExit(f'block {args.block_idx} is windowed (win={blk.window_size}); '
                         'pick a global block (vit_l: 5, 11, 17, 23).')

    enc_cap = {}
    def enc_hook(_mod, inputs):
        enc_cap['x'] = inputs[0].detach().clone()
    enc_handle = blk.attn.register_forward_pre_hook(enc_hook)

    # Decoder monkey-patches.
    transformer = sam.mask_decoder.transformer
    cross_modules = [lyr.cross_attn_token_to_image for lyr in transformer.layers]
    cross_modules.append(transformer.final_attn_token_to_image)
    dec_cap, restore = install_decoder_capture(cross_modules)

    # Run.
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
    enc_handle.remove()
    restore()

    # Encoder side: keep-score map at chosen block.
    x = enc_cap['x']                                   # (1, H, W, C) global
    score_map = compute_keep_score_map(blk, x, args.group_size)   # (H, W)

    # Decoder side: sum mask+HQ token attention from final layer, avg heads.
    final_attn = dec_cap[id(cross_modules[-1])][0]      # (1, nh, T, 4096)
    if args.head_agg == 'mean':
        final_attn = final_attn.mean(dim=1)
    else:
        final_attn = final_attn.max(dim=1).values
    decoder_per_token = final_attn[0].float()           # (T, 4096)
    # Tokens 1..5 = 4 SAM mask + 1 HQ (HF). Sum -> per-patch attention.
    decoder_map = decoder_per_token[1:6].sum(dim=0).numpy().reshape(64, 64)

    # Sanity: shapes match.
    assert score_map.shape == decoder_map.shape == (64, 64)

    # Pearson correlation over the 4096 patches.
    s_flat = score_map.reshape(-1)
    d_flat = decoder_map.reshape(-1)
    s_norm = (s_flat - s_flat.mean()) / (s_flat.std() + 1e-9)
    d_norm = (d_flat - d_flat.mean()) / (d_flat.std() + 1e-9)
    pearson = float((s_norm * d_norm).mean())

    # Spearman (rank correlation) — robust to the heavy-tailed decoder map.
    s_rank = np.argsort(np.argsort(s_flat))
    d_rank = np.argsort(np.argsort(d_flat))
    sr = (s_rank - s_rank.mean()) / (s_rank.std() + 1e-9)
    dr = (d_rank - d_rank.mean()) / (d_rank.std() + 1e-9)
    spearman = float((sr * dr).mean())

    # ── plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

    up_h, up_w = H_img // 64, W_img // 64

    def overlay(ax, m: np.ndarray, title: str):
        m_n = (m - m.min()) / (m.max() - m.min() + 1e-9)
        m_n = m_n ** 0.35
        ax.imshow(img, zorder=1)
        ax.imshow(m_n, cmap=CYAN_OVERLAY, vmin=0, vmax=1,
                  extent=(0, W_img, H_img, 0),
                  interpolation='bilinear', zorder=2)
        ax.scatter([px], [py], marker='*', s=420, c='#ffe600',
                   edgecolors='black', linewidths=1.6, zorder=10)
        ax.set_title(title, fontsize=18)
        ax.set_xticks([]); ax.set_yticks([])

    overlay(axes[0], score_map, f'sparsesam keep-score (block {args.block_idx})')
    overlay(axes[1], decoder_map, 'decoder cross-attn (mask+HQ, final layer)')

    # Scatter.
    ax = axes[2]
    ax.scatter(s_flat, d_flat, s=8, alpha=0.35, c='#117a8b',
               edgecolors='none', rasterized=True)
    ax.set_xlabel('encoder keep-score', fontsize=14)
    ax.set_ylabel('decoder attn (mask+HQ sum)', fontsize=14)
    ax.set_title(f'r={pearson:+.3f}   ρ={spearman:+.3f}', fontsize=18)
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches='tight')
    iou_str = ', '.join(f'{i:.3f}' for i in ious)
    print(f'saved -> {args.out}')
    print(f'  block={args.block_idx}  point=({px},{py})  IoUs=[{iou_str}]')
    print(f'  Pearson r = {pearson:+.4f}   Spearman ρ = {spearman:+.4f}')


if __name__ == '__main__':
    main()
