#!/usr/bin/env python3
"""Render a polished version of the cluster-replace probe figure from a
saved CSV. Reads cols: image_idx, k, mode, iou, boundary_iou."""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def render(csv_path: str, out_path: str, dataset_name: str = "COIFT") -> None:
    df = pd.read_csv(csv_path)
    n_imgs = df['image_idx'].nunique()

    agg = (df.groupby(['mode', 'k'], as_index=False)
             .agg(miou=('iou', 'mean'),
                  std=('iou', 'std'),
                  sem=('iou', lambda s: s.std() / np.sqrt(len(s)))))

    base = agg[agg['mode'] == 'baseline']
    centroid = agg[(agg['mode'] == 'centroid') & (agg['k'] > 0)].sort_values('k')
    random_  = agg[(agg['mode'] == 'random')   & (agg['k'] > 0)].sort_values('k')
    base_mu = float(base['miou'].iloc[0]) if not base.empty else None

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 30,
        'axes.titlesize': 34,
        'axes.titleweight': 'semibold',
        'axes.labelsize': 32,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'legend.fontsize': 30,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#e6e6e6',
        'grid.linewidth': 0.8,
        'xtick.color': '#444',
        'ytick.color': '#444',
        'axes.edgecolor': '#444',
    })

    fig, ax = plt.subplots(1, 1, figsize=(13, 8.0), dpi=160)

    C_CENT = '#0891b2'   # cyan-600
    C_RAND = '#d1242f'   # red (unused when random hidden)
    C_BASE = '#475569'   # slate-600

    if base_mu is not None:
        ax.axhline(base_mu, ls=(0, (5, 3)), color=C_BASE, lw=1.5, alpha=0.9, zorder=1)
        ax.text(centroid['k'].max() * 1.05, base_mu,
                f' baseline {base_mu:.3f}',
                va='center', ha='left', fontsize=24, color=C_BASE)

    # shaded SEM band + mean line
    def plot_band(d, color, label, marker, z):
        if d.empty:
            return
        x, y, e = d['k'].values, d['miou'].values, d['sem'].values
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.15, lw=0, zorder=z)
        ax.plot(x, y, color=color, lw=2.0, zorder=z + 1)
        ax.scatter(x, y, color=color, s=42, marker=marker,
                   edgecolor='white', linewidths=1.2, zorder=z + 2, label=label)

    plot_band(centroid, C_CENT, None, 'o', 4)

    # Annotate the "elbow" — k where centroid recovers ~90% of baseline gain
    if base_mu is not None and not centroid.empty:
        kvals  = centroid['k'].values
        mious  = centroid['miou'].values
        floor  = mious[0]
        target = floor + 0.90 * (base_mu - floor)
        idx    = int(np.argmin(np.abs(mious - target)))
        kx, my = kvals[idx], mious[idx]
        ax.annotate(
            f'k={kx} · {(my/base_mu)*100:.0f}% of baseline',
            xy=(kx, my),
            xytext=(kx * 0.18, my - 0.18),
            fontsize=24, color=C_CENT,
            arrowprops=dict(arrowstyle='-', color=C_CENT, lw=1.4, alpha=0.7,
                            connectionstyle='arc3,rad=-0.2'),
        )

    ax.set_xscale('log', base=2)
    ax.set_xlabel('num clusters')
    ax.set_ylabel('mIoU')

    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 1024])
    ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128', '256', '1024'])
    ax.set_xlim(0.85, 2200)
    ax.set_ylim(-0.02, 1.0)


    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--dataset', default='COIFT')
    args = p.parse_args()
    out = args.out or args.csv.replace('.csv', '_polished.png')
    render(args.csv, out, args.dataset)


if __name__ == '__main__':
    main()
