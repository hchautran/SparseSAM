"""Side-by-side segmentation results from baseline SAM-HQ vs. several
encoder-compression algorithms at a fixed density. Reports encoder
forward-pass latency below each panel.

Per algo:
  1. revert any prior patch -> apply the algo's encoder patch at `--ratio`
  2. time encoder forward via `predictor.set_image()` (warmup + N runs)
  3. run mask decoder with one foreground point prompt
  4. compose figure: image + mask overlay, with `algo · NN ms` caption

Example:
    python tasks/sam_hq44k/plot_algo_seg_results.py \
        --image ./input_imgs/example1.png --point 700,500 \
        --algos tome gradtome sparge sparsesam --ratio 0.3 \
        --out ./benchmark_results/algo_seg_results.png
"""

import argparse
import os
import sys
import time
import types

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'algos', '3rd_party', 'sam-hq'))

from segment_anything import sam_model_registry, SamPredictor  # noqa: E402
from algos.registry import apply_sam, remove_all_sam  # noqa: E402


LABELS = {
    'baseline':  'baseline SAM-HQ',
    'tome':      'ToMe',
    'gradtome':  'GradToMe',
    'sparge':    'SpargeAttn',
    'sparsesam': 'SparseSAM (ours)',
}

# Distinct accent colors per algorithm so each mask outline reads quickly.
COLORS = {
    'baseline':  '#475569',  # slate
    'tome':      '#94a3b8',  # cool grey
    'gradtome':  '#f59e0b',  # amber
    'sparge':    '#a78bfa',  # violet
    'sparsesam': '#0891b2',  # cyan-600 (ours)
}


def disable_hq_fusion(sam_model) -> None:
    """Override SAM-HQ mask_decoder.forward to return SAM-only masks
    (skip the `masks_sam + masks_hq` fusion). Encoder-compression effects
    are otherwise hidden by the HQ refinement branch."""
    decoder = sam_model.mask_decoder

    def fwd(self, image_embeddings, image_pe, sparse_prompt_embeddings,
            dense_prompt_embeddings, multimask_output, hq_token_only,
            interm_embeddings):
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)
        hq_features = (self.embedding_encoder(image_embeddings)
                       + self.compress_vit_feat(vit_features))
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            hq_features=hq_features,
        )
        if multimask_output:
            mask_slice = slice(1, self.num_mask_tokens - 1)
            ip = iou_pred[:, mask_slice]
            ip, max_idx = torch.max(ip, dim=1)
            ip = ip.unsqueeze(1)
            mm = masks[:, mask_slice]
            masks_sam = mm[torch.arange(mm.size(0)), max_idx].unsqueeze(1)
        else:
            mask_slice = slice(0, 1)
            ip = iou_pred[:, mask_slice]
            masks_sam = masks[:, mask_slice]
        return masks_sam, ip   # SAM masks only — no HQ branch added

    decoder.forward = types.MethodType(fwd, decoder)


def time_encoder(predictor: SamPredictor, img: np.ndarray,
                 n_warmup: int = 2, n_runs: int = 3) -> float:
    """Mean encoder set_image latency in milliseconds. Synchronizes CUDA."""
    for _ in range(n_warmup):
        predictor.reset_image()
        predictor.set_image(img)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        predictor.reset_image()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.set_image(img)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000.0)


def run_one(predictor: SamPredictor, img: np.ndarray,
            points: list[tuple[int, int]] | None = None,
            box: tuple[int, int, int, int] | None = None,
            multimask: bool = False):
    kwargs = dict(multimask_output=multimask, hq_token_only=False)
    if box is not None:
        kwargs['box'] = np.array(box, dtype=np.float32)
    if points:
        kwargs['point_coords'] = np.array(points, dtype=np.float32)
        kwargs['point_labels'] = np.ones(len(points), dtype=np.int32)
    masks, ious, _ = predictor.predict(**kwargs)
    return masks[0], float(ious[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-type', default='vit_l')
    ap.add_argument('--ckpt', default='./ckts/sam_hq_vit_l.pth')
    ap.add_argument('--image', default='./input_imgs/example1.png')
    ap.add_argument('--ratio', type=float, default=0.3)
    ap.add_argument('--algos', nargs='+',
                    default=['tome', 'gradtome', 'sparge', 'sparsesam'])
    ap.add_argument('--points', default=None,
                    help='Foreground points "x1,y1;x2,y2;...".')
    ap.add_argument('--box', default='520,280,960,880',
                    help='Bounding box "xmin,ymin,xmax,ymax". Default: a '
                         'box around the butterfly in example1.png.')
    ap.add_argument('--multimask', action='store_true',
                    help='Use SAM 3-mask output (picks max-IoU candidate). '
                         'Off by default — calibrated single mask is more '
                         'reliable for full-object segmentation.')
    ap.add_argument('--include-baseline', action='store_true', default=True)
    ap.add_argument('--no-baseline', dest='include_baseline', action='store_false')
    ap.add_argument('--out', default='./benchmark_results/algo_seg_results.png')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt).to(device).eval().half()
    disable_hq_fusion(sam)
    encoder = sam.image_encoder
    predictor = SamPredictor(sam)

    img = np.array(Image.open(args.image).convert('RGB'))
    H, W = img.shape[:2]
    points = None
    box = None
    if args.points:
        points = [tuple(int(c) for c in p.split(','))
                  for p in args.points.split(';')]
    if args.box:
        box = tuple(int(v) for v in args.box.split(','))
    if not points and not box:
        points = [(W // 2, H // 2)]

    schedule = (['baseline'] if args.include_baseline else []) + list(args.algos)
    results = []
    for name in schedule:
        remove_all_sam(encoder, sam.mask_decoder)
        if name != 'baseline':
            print(f'  applying {name} @ ratio={args.ratio} ...', flush=True)
            apply_sam(encoder, name=name, ratio=args.ratio)
        else:
            print(f'  baseline (no patch) ...', flush=True)
        t_ms = time_encoder(predictor, img)
        mask, iou = run_one(predictor, img,
                            points=points, box=box,
                            multimask=args.multimask)
        results.append(dict(name=name, mask=mask, iou=iou, t_ms=t_ms))
        print(f'    {name:<12s}  encoder={t_ms:6.1f} ms  IoU≈{iou:.3f}',
              flush=True)
    remove_all_sam(encoder, sam.mask_decoder)

    # ── plot ──
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.4), squeeze=False)
    axes = axes[0]
    for ax, r in zip(axes, results):
        ax.imshow(img)
        # Mask overlay: solid color with alpha; outline with edge contour.
        c = COLORS.get(r['name'], '#0891b2')
        from matplotlib.colors import to_rgba
        rgba = list(to_rgba(c)); rgba[3] = 0.45
        layer = np.zeros((*r['mask'].shape, 4), dtype=np.float32)
        layer[r['mask']] = rgba
        ax.imshow(layer)
        ax.contour(r['mask'].astype(np.uint8), levels=[0.5],
                   colors=c, linewidths=2.0)
        if box is not None:
            x0, y0, x1, y1 = box
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2.5, edgecolor='#ffe600',
                facecolor='none', zorder=10,
            ))
        if points:
            for (qx, qy) in points:
                ax.scatter([qx], [qy], marker='*', s=320, c='#ffe600',
                           edgecolors='black', linewidths=1.4, zorder=10)
        ax.set_title(f'{LABELS.get(r["name"], r["name"])}\n'
                     f'encoder: {r["t_ms"]:.1f} ms',
                     fontsize=18, color=c)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f'SAM-HQ ({args.model_type}) on {os.path.basename(args.image)}   '
        f'density={args.ratio}',
        fontsize=20,
    )
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
