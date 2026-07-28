#!/usr/bin/env python3
"""Top-1 accuracy vs keep-ratio per algo for PE/ImageNet, styled to match
the SAM-HQ `algo_miou_vs_ratio_combined.png` figure.

Reads `.outputs/pe_imagenet/{baseline,attn_only,attn_plus_mlp}/pe_clip_<MODEL>.csv`.
Layout:  1 row × 2 cols, columns = {attn_only, attn+MLP}; one curve per algo,
hue encodes mean speedup, baseline as a dashed horizontal reference.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


# slow → fast (ours last, drawn on top)
ALGO_ORDER = ['tome_partial', 'sparge', 'sparsesam_partial']

# Mean wall-clock speedup vs PE baseline (251.4s @ 50k images), averaged
# across attn_only + attn_plus_mlp variants where available.
ALGO_SPEEDUP = {
    'tome_partial':      0.85,    # slower than baseline
    'sparge':            0.95,    # ~baseline
    'sparsesam_partial': 1.40,    # ours, fastest
}

# Cyan ramp: pale slate (slow) → deep teal (fast). Matches plot_algo_sweep.py.
_HUE = LinearSegmentedColormap.from_list('speedup_cyan', [
    (0.00, '#e2e8f0'),
    (0.30, '#a5dfe6'),
    (0.55, '#3fb6c0'),
    (0.80, '#11697a'),
    (1.00, '#062a36'),
])
_HUE_LO, _HUE_HI = 0.5, 1.6


def _speedup_to_color(sp: float, lo: float = _HUE_LO, hi: float = _HUE_HI) -> tuple:
    t = max(0.0, min(1.0, (sp - lo) / (hi - lo)))
    return _HUE(t)


COLORS = {a: _speedup_to_color(s) for a, s in ALGO_SPEEDUP.items()}
MARKERS = {
    'tome_partial':      'o',
    'sparge':            'D',
    'sparsesam_partial': '*',
}
LABELS  = {
    'tome_partial':      f'ToMe ({ALGO_SPEEDUP["tome_partial"]:.1f}×)',
    'sparge':            f'SpargeAttn ({ALGO_SPEEDUP["sparge"]:.1f}×)',
    'sparsesam_partial': f'SparseSAM (ours, {ALGO_SPEEDUP["sparsesam_partial"]:.1f}×)',
}


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size':         14,
        'axes.titlesize':    18,
        'axes.titleweight':  'semibold',
        'axes.labelsize':    16,
        'xtick.labelsize':   13,
        'ytick.labelsize':   13,
        'legend.fontsize':   15,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.color':        '#e6e6e6',
        'grid.linewidth':    0.8,
        'xtick.color':       '#444',
        'ytick.color':       '#444',
        'axes.edgecolor':    '#444',
    })


def load(model: str, variant: str, base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = os.path.join(base_dir, variant, f'pe_clip_{model}.csv')
    sw = pd.read_csv(path)
    base = pd.read_csv(os.path.join(base_dir, 'baseline', f'pe_clip_{model}.csv'))
    return sw, base


def _draw_panel(ax, d_all, b_mu) -> None:
    if b_mu is not None:
        ax.axhline(b_mu, ls=(0, (5, 3)), color='#475569', lw=1.4, alpha=0.9, zorder=1)
        ax.text(0.98, 0.04, f'baseline {b_mu:.3f}',
                va='bottom', ha='right', fontsize=14, color='#475569',
                transform=ax.transAxes, zorder=20,
                bbox=dict(facecolor='white', edgecolor='#cbd5e1',
                          alpha=1.0, pad=3))

    for algo in ALGO_ORDER:
        d_algo = d_all[d_all['algorithm'] == algo]
        if d_algo.empty:
            continue
        d = (d_algo
             .groupby('ratio', as_index=False)
             .agg(acc=('acc1', 'mean'),
                  sem=('acc1', lambda s: s.std() / max(np.sqrt(len(s)), 1)))
             .sort_values('ratio'))
        x, y, e = d['ratio'].values, d['acc'].values, d['sem'].values
        c = COLORS[algo]
        ours = (algo == 'sparsesam_partial')
        lw = 5.0 if ours else 4.0
        ms = 520 if ours else 260
        z_base = 6 if ours else 3
        ax.fill_between(x, y - e, y + e, color=c,
                        alpha=0.18 if ours else 0.12, lw=0, zorder=z_base - 1)
        # dark stroke under the line for legibility
        ax.plot(x, y, color='#1f2937', lw=lw + 2.6,
                zorder=z_base - 1, alpha=0.35, solid_capstyle='round')
        ax.plot(x, y, color=c, lw=lw, zorder=z_base, solid_capstyle='round')
        ax.scatter(x, y, color=c, s=ms, marker=MARKERS[algo],
                   edgecolor='#1f2937',
                   linewidths=2.0 if ours else 1.6,
                   zorder=z_base + 1, label=LABELS[algo])

    ax.set_xlim(0.20, 0.80)
    ymin = float(d_all['acc1'].min())
    ymax = float(b_mu) if b_mu is not None else float(d_all['acc1'].max())
    pad = max(0.02, 0.06 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)


def render_combined(model: str, base_dir: str, out_path: str) -> None:
    sw_attn,    base = load(model, 'attn_only',     base_dir)
    sw_attnmlp, _    = load(model, 'attn_plus_mlp', base_dir)

    style()
    cols = [
        ('without MLP compressing', sw_attn),
        ('with MLP compressing',    sw_attnmlp),
    ]
    fig, axes = plt.subplots(1, len(cols),
                             figsize=(5.5 * len(cols), 4.0),
                             dpi=160, squeeze=False)

    b_mu = float(base['acc1'].mean()) if not base.empty else None
    for c, (col_label, sw) in enumerate(cols):
        ax = axes[0, c]
        _draw_panel(ax, sw, b_mu)
        ax.set_title(col_label, loc='left', color='#222')
        ax.set_xlabel('density')
        if c == 0:
            ax.set_ylabel('Top-1 Acc', fontsize=16)
        else:
            ax.set_ylabel('')

    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Vertical colorbar on the right + legend below.
    fig.tight_layout(rect=(0, 0.10, 0.93, 1))

    cax = fig.add_axes([0.945, 0.18, 0.012, 0.74])
    sm = ScalarMappable(norm=Normalize(_HUE_LO, _HUE_HI), cmap=_HUE)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='vertical')
    cb.set_label('average speedup', fontsize=12, labelpad=6)
    cb.ax.tick_params(labelsize=11)
    cb.ax.axhline(1.0, color='#475569', lw=1.2)

    fig.legend(handles, labels, loc='lower center', ncols=len(labels),
               frameon=False, fontsize=15, bbox_to_anchor=(0.46, 0.0),
               markerscale=0.6, handlelength=2.0,
               columnspacing=1.4, handletextpad=0.5)

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    fig.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    fig.savefig(out_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"Plot → {out_path}")
    print(f"Plot → {out_path.replace('.png', '.pdf')}")
    print(f"Plot → {out_path.replace('.png', '.svg')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='PE-Core-L14-336')
    p.add_argument('--base-dir', default='./.outputs/pe_imagenet')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    out = args.out or os.path.join(
        args.base_dir, 'plots', f'algo_acc_vs_ratio_combined.png',
    )
    render_combined(args.model, args.base_dir, out)


if __name__ == '__main__':
    main()
