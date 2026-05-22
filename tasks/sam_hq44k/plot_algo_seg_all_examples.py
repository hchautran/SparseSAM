"""Run baseline + 4 compression algos on each SAM-HQ demo image
(example0..7) using the per-image prompts from the official demo.
Writes one PNG per image plus a combined stacked figure.

Per-image prompts (box / points / multi-box + hq_token_only) match the
SAM-HQ demo: https://github.com/SysCV/sam-hq/blob/main/demo_hq_inference.py

Example:
    python tasks/sam_hq44k/plot_algo_seg_all_examples.py --ratio 0.25
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'algos', '3rd_party', 'sam-hq'))

from segment_anything import sam_model_registry, SamPredictor  # noqa: E402
from algos.registry import apply_sam, remove_all_sam  # noqa: E402


PROMPTS = {
    0: dict(box=[4, 13, 1007, 1023],                     hq=False),
    1: dict(box=[306, 132, 925, 893],                    hq=True),
    2: dict(points=[(495, 518), (217, 140)],             hq=True),
    3: dict(points=[(221, 482), (498, 633), (750, 379)], hq=False),
    4: dict(box=[64, 76, 940, 919],                      hq=True),
    5: dict(points=[(373, 363), (452, 575)],             hq=False),
    6: dict(box=[181, 196, 757, 495],                    hq=False),
    7: dict(boxes=[[45, 260, 515, 470], [310, 228, 424, 296]], hq=False),
}

LABELS = {
    'baseline':  'baseline',
    'tome':      'ToMe',
    'gradtome':  'GradToMe',
    'sparge':    'SpargeAttn',
    'sparsesam': 'SparseSAM',
}

MASK_COLOR = '#0891b2'   # cyan-600 — same color for every algo's mask


def time_encoder(predictor: SamPredictor, img: np.ndarray,
                 n_warmup: int = 2, n_runs: int = 3) -> float:
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


def predict_with(predictor: SamPredictor, prompt: dict):
    """Returns (binary_mask_HxW, mean_iou)."""
    if 'boxes' in prompt:
        bt = torch.tensor(prompt['boxes'], device=predictor.device,
                          dtype=torch.float)
        boxes_t = predictor.transform.apply_boxes_torch(
            bt, predictor.original_size,
        )
        masks, ious, _ = predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=boxes_t,
            multimask_output=False, hq_token_only=prompt['hq'],
        )
        mask = (masks[:, 0] > 0).any(dim=0).cpu().numpy()
        return mask, float(ious.mean())

    pc, pl, box = None, None, None
    if 'points' in prompt:
        pc = np.array(prompt['points'], dtype=np.float32)
        pl = np.ones(len(prompt['points']), dtype=np.int32)
    if 'box' in prompt:
        box = np.array(prompt['box'], dtype=np.float32)
    masks, ious, _ = predictor.predict(
        point_coords=pc, point_labels=pl, box=box,
        multimask_output=False, hq_token_only=prompt['hq'],
    )
    return masks[0], float(ious[0])


def _draw_prompt(ax, prompt):
    if 'box' in prompt:
        x0, y0, x1, y1 = prompt['box']
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                linewidth=2.0, edgecolor='#ffe600',
                                facecolor='none', zorder=10))
    if 'boxes' in prompt:
        for x0, y0, x1, y1 in prompt['boxes']:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    linewidth=2.0, edgecolor='#ffe600',
                                    facecolor='none', zorder=10))
    if 'points' in prompt:
        for (qx, qy) in prompt['points']:
            ax.scatter([qx], [qy], marker='*', s=240, c='#ffe600',
                       edgecolors='black', linewidths=1.4, zorder=10)


def draw_input_panel(ax, img, prompt):
    """Original image with the input prompt overlay only — no mask."""
    ax.imshow(img)
    _draw_prompt(ax, prompt)
    ax.set_title('input', fontsize=22, color='black')
    ax.set_xticks([]); ax.set_yticks([])


def draw_panel(ax, img, mask, name, prompt, t_ms, baseline_t):
    ax.imshow(img)
    rgba = list(to_rgba(MASK_COLOR)); rgba[3] = 0.45
    layer = np.zeros((*mask.shape, 4), dtype=np.float32)
    layer[mask] = rgba
    ax.imshow(layer)
    ax.contour(mask.astype(np.uint8), levels=[0.5],
               colors=MASK_COLOR, linewidths=2.0)
    # Prompt overlay only on the leading "input" panel — keep mask
    # panels uncluttered so segmentation differences read at a glance.
    ax.set_title(f'{LABELS.get(name, name)} — {t_ms:.0f} ms',
                 fontsize=22, color='black')
    ax.set_xticks([]); ax.set_yticks([])

    # Speedup badge (skip baseline itself).
    if name != 'baseline' and baseline_t > 0:
        sp = baseline_t / t_ms
        if sp >= 1:
            txt = f'{sp:.2f}× faster'
            color = '#16a34a'   # green-600
        else:
            txt = f'{1.0 / sp:.2f}× slower'
            color = '#dc2626'   # red-600
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, color=color,
                ha='right', va='top', fontsize=20, fontweight='bold',
                zorder=20,
                bbox=dict(facecolor='white', alpha=0.92,
                          edgecolor=color, linewidth=1.4, pad=5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-type', default='vit_l')
    ap.add_argument('--ckpt', default='./ckts/sam_hq_vit_l.pth')
    ap.add_argument('--ratio', type=float, default=0.25)
    ap.add_argument('--algos', nargs='+',
                    default=['tome', 'sparge', 'sparsesam'])
    ap.add_argument('--examples', type=int, nargs='+',
                    default=sorted(PROMPTS.keys()))
    ap.add_argument('--out-dir',
                    default='./benchmark_results/algo_seg_examples')
    args = ap.parse_args()

    device = torch.device('cuda')
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt).to(device).eval().half()
    encoder = sam.image_encoder
    predictor = SamPredictor(sam)
    os.makedirs(args.out_dir, exist_ok=True)

    schedule = ['baseline'] + list(args.algos)
    n_cols = len(schedule) + 1   # extra leading column = input + prompt
    n_ex = len(args.examples)
    fig_c, axes_c = plt.subplots(n_ex, n_cols,
                                 figsize=(4.4 * n_cols, 4.6 * n_ex),
                                 squeeze=False)

    for r_idx, ex in enumerate(args.examples):
        prompt = PROMPTS[ex]
        path = f'./input_imgs/example{ex}.png'
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        print(f'\n=== example{ex} (size {img.shape[:2]}) hq={prompt["hq"]} ===')

        per_algo = []
        for name in schedule:
            remove_all_sam(encoder, sam.mask_decoder)
            if name != 'baseline':
                apply_sam(encoder, name=name, ratio=args.ratio)
            t_ms = time_encoder(predictor, img)
            mask, iou = predict_with(predictor, prompt)
            per_algo.append(dict(name=name, mask=mask, iou=iou, t_ms=t_ms))
            print(f'  {name:<12s}  encoder={t_ms:6.1f} ms  IoU≈{iou:.3f}')
        remove_all_sam(encoder, sam.mask_decoder)

        baseline_t = next((p['t_ms'] for p in per_algo
                           if p['name'] == 'baseline'), 0.0)

        # Per-image figure
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(4.4 * n_cols, 4.8),
                                 squeeze=False)
        draw_input_panel(axes[0][0], img, prompt)
        for ax, r in zip(axes[0][1:], per_algo):
            draw_panel(ax, img, r['mask'], r['name'], prompt,
                       r['t_ms'], baseline_t)
        # No suptitle — keep panels clean. Density shown in filename.
        out = os.path.join(args.out_dir, f'algo_seg_example{ex}.png')
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {out}')

        # Combined row
        draw_input_panel(axes_c[r_idx][0], img, prompt)
        for ax, r in zip(axes_c[r_idx][1:], per_algo):
            draw_panel(ax, img, r['mask'], r['name'], prompt,
                       r['t_ms'], baseline_t)

    # No suptitle on the combined figure either.
    out = os.path.join(args.out_dir, 'algo_seg_all_examples.png')
    fig_c.tight_layout(rect=(0, 0, 1, 0.98))
    fig_c.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig_c)
    print(f'\nsaved combined -> {out}')


if __name__ == '__main__':
    main()
