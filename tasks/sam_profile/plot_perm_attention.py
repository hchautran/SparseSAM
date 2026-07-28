"""Plot SAM-HQ attention before vs. after `tile_stride_matching` permutation.

Captures Q/K from one or more transformer blocks, computes
softmax(QK^T + rel_pos) in natural order, then re-indexes by the
permutation that sparsesam's FA2 kernel applies. The "after" panels are
the same attention scores reordered so kept tokens occupy the upper-left
block (matching the A-mask).

Examples:
    # default: all four global blocks of vit_l, side-by-side before/after
    python tasks/sam_profile/plot_perm_attention.py

    # one block
    python tasks/sam_profile/plot_perm_attention.py --block-indices 23

    # custom set
    python tasks/sam_profile/plot_perm_attention.py --block-indices 0 5 11 17 23
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

from segment_anything import sam_model_registry  # noqa: E402
from segment_anything.modeling.image_encoder import add_decomposed_rel_pos  # noqa: E402
from algos.sparsesam.sam import tile_stride_matching  # noqa: E402


# Matches autoresearch/plot_findings.py:CYAN_ATTN — white background,
# dark-teal peak, so the two attention figure families share one palette.
CYAN_CMAP = LinearSegmentedColormap.from_list(
    'cyan_attn',
    [
        (0.00, '#ffffff'),  # white background
        (0.10, '#e9f8fa'),  # very faint cyan
        (0.30, '#a5dfe6'),  # light cyan
        (0.55, '#3fb6c0'),  # vivid cyan
        (0.80, '#11697a'),  # mid teal
        (1.00, '#062a36'),  # deep teal (near-black)
    ],
    N=256,
)
BAR_COLOR = '#d62728'  # red — contrasts with white background


def preprocess_image(path: str, target: int = 1024) -> torch.Tensor:
    img = np.array(Image.open(path).convert('RGB'))
    H, W = img.shape[:2]
    scale = target / max(H, W)
    nH, nW = int(round(H * scale)), int(round(W * scale))
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    t = F.interpolate(t, size=(nH, nW), mode='bilinear', align_corners=False)
    t = F.pad(t, (0, target - nW, 0, target - nH))
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    return (t - mean) / std


def pick_render_n_block(N: int, target: int = 64) -> int:
    """Largest divisor of N <= target that yields >= 8 cells per side.

    Global ViT-L blocks (N=4096) → 64 (matches the cute kernel granularity).
    Windowed blocks (N=196)      → 14 (one cell per row of the 14x14 window).
    Falls back to N itself when no clean divisor exists.
    """
    if N % target == 0 and N // target >= 8:
        return target
    for nb in range(min(N, target), 0, -1):
        if N % nb == 0 and 8 <= N // nb <= 64:
            return nb
    return N


def compute_attn_pair(blk, x, ratio, group_size, n_block):
    """Return (A_nat, A_perm, keep_n, meta) for one block's captured input.

    Both maps are block-downsampled (mean over each render_nb x render_nb
    sub-tile) at a cute-style granularity matching
    `autoresearch/plot_findings.py:plot_attn_masked_2x2`'s `downsample_blk`,
    so the two figure families share the same visual language.
    """
    B_, H, W, C = x.shape
    nh = blk.attn.num_heads
    D = C // nh
    scale = blk.attn.scale
    N = H * W

    qkv = blk.attn.qkv(x.view(B_, N, C))
    qkv = qkv.view(B_, N, 3, nh, D).permute(2, 0, 3, 1, 4)
    q, k, _v = qkv.reshape(3, B_ * nh, N, D).unbind(0)

    pre = (q * scale) @ k.transpose(-2, -1)
    pre = add_decomposed_rel_pos(
        pre, q,
        blk.attn.rel_pos_h, blk.attn.rel_pos_w,
        (H, W), (H, W),
    )
    A_nat = pre.softmax(dim=-1)

    perm, _inv, _ = tile_stride_matching(
        k, H, W, ratio=ratio, group_size=group_size, n_block=n_block,
    )
    A_perm = (A_nat.gather(1, perm.unsqueeze(-1).expand(-1, -1, N))
                   .gather(2, perm.unsqueeze(1).expand(-1, N, -1)))

    render_nb = pick_render_n_block(N, target=n_block)
    nb = N // render_nb
    A_nat = A_nat.view(B_ * nh, nb, render_nb, nb, render_nb).mean(dim=(2, 4))
    A_perm = A_perm.view(B_ * nh, nb, render_nb, nb, render_nb).mean(dim=(2, 4))

    num_n_blocks = math.ceil(N / n_block)
    keep_n = min(int(ratio * num_n_blocks) * n_block, N)
    meta = dict(B_=B_, H=H, W=W, nh=nh, N=N, N_render=nb,
                render_nb=render_nb, win=blk.window_size)
    return A_nat, A_perm, keep_n, meta


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-type', default='vit_l')
    ap.add_argument('--ckpt', default='./ckts/sam_hq_vit_l.pth')
    ap.add_argument('--image', default='./input_imgs/dog.jpg')
    ap.add_argument('--block-indices', type=int, nargs='+',
                    default=[5, 11, 17, 23],
                    help='Blocks to inspect. vit_l globals: 5,11,17,23. '
                         'Defaults to all globals.')
    ap.add_argument('--head-idx', type=int, default=0)
    ap.add_argument('--window-idx', type=int, default=0,
                    help='For windowed blocks: which window of the (B*nw) batch.')
    ap.add_argument('--ratio', type=float, default=0.5)
    ap.add_argument('--group-size', type=int, default=4)
    ap.add_argument('--n-block', type=int, default=64)
    ap.add_argument('--out', default='./benchmark_results/perm_attention.png')
    ap.add_argument('--device', default='cuda')
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt).to(device).eval()
    encoder = sam.image_encoder.float()

    block_idxs = list(args.block_indices)
    captured: dict = {}
    handles = []
    for i in block_idxs:
        blk = encoder.blocks[i]

        def make_hook(idx):
            def pre_hook(_mod, inputs):
                captured[idx] = inputs[0].detach().clone()
            return pre_hook

        handles.append(blk.attn.register_forward_pre_hook(make_hook(i)))

    img = preprocess_image(args.image).to(device).float()
    with torch.inference_mode():
        encoder(img)
    for h in handles:
        h.remove()

    n_rows = len(block_idxs)
    panel = 5.0
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(2 * panel, panel * n_rows),
                             squeeze=False)

    for row, i in enumerate(block_idxs):
        blk = encoder.blocks[i]
        A_nat, A_perm, keep_n, meta = compute_attn_pair(
            blk, captured[i],
            ratio=args.ratio, group_size=args.group_size, n_block=args.n_block,
        )
        sample_idx = max(0, min(args.window_idx, meta['B_'] - 1))
        flat = sample_idx * meta['nh'] + args.head_idx
        A_nat_one = A_nat[flat].detach().cpu().numpy()
        A_perm_one = A_perm[flat].detach().cpu().numpy()
        is_global = (meta['win'] == 0)
        layout = 'global' if is_global else f'win={meta["win"]}'

        # Shared linear range (matches autoresearch attn_masked_2x2):
        # 99th percentile of the natural-order map keeps the colorbar
        # scale invariant under permutation.
        vmax = float(np.quantile(A_nat_one, 0.99))
        panels = [
            (axes[row, 0], A_nat_one, f'block {i} — before'),
            (axes[row, 1], A_perm_one, f'block {i} — after'),
        ]
        for ax, A, title in panels:
            ax.imshow(A, cmap=CYAN_CMAP, aspect='equal',
                      vmin=0, vmax=vmax, interpolation='nearest')
            ax.set_title(title, fontsize=18)
            ax.set_xticks([])
            ax.set_yticks([])
        print(f'block {i:>2} {layout:<10} N={meta["N"]:<5} '
              f'render={meta["N_render"]}x{meta["N_render"]} '
              f'B*={meta["B_"]:<3} nh={meta["nh"]} keep_n={keep_n}')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
