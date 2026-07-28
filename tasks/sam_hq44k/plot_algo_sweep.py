#!/usr/bin/env python3
"""mIoU vs keep-ratio per algo, one panel per dataset.

Reads `.outputs/sam_hq44k/<model>/{baseline,attn_only,attn_plus_mlp}/results.csv`.
Same visual style as plot_cluster_probe.py — cyan-leaning palette, SEM bands,
no legend frame chrome, baseline as a dashed horizontal reference.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


ALGO_ORDER = ['piecewise', 'gradtome', 'tome', 'sparge', 'sparsesam']  # slow → fast (ours last, drawn on top)

# Mean encoder speedup vs SAM-HQ baseline (ratio of baseline_time / algo_time
# at the densities we sweep). Higher = faster; <1.0 means slower-than-baseline.
ALGO_SPEEDUP = {
    'piecewise': 0.65,   # slower-than-baseline (0.6–0.7× per measurement)
    'gradtome':  0.70,
    'tome':      0.90,
    'sparge':    1.10,
    'sparsesam': 2.20,   # ours, fastest
}

# Hue scale: line color encodes mean speedup. Pale-to-deep cyan ramp,
# but with line weight + opacity bumped (below) so the slow end is still
# easy to follow visually.
_HUE = LinearSegmentedColormap.from_list('speedup_cyan', [
    (0.00, '#e2e8f0'),   # slate-200 — slowest
    (0.30, '#a5dfe6'),   # light cyan
    (0.55, '#3fb6c0'),   # vivid cyan
    (0.80, '#11697a'),   # mid teal
    (1.00, '#062a36'),   # deep teal — fastest
])


_HUE_LO, _HUE_HI = 0.5, 2.4


def _speedup_to_color(sp: float, lo: float = _HUE_LO, hi: float = _HUE_HI) -> tuple:
    t = max(0.0, min(1.0, (sp - lo) / (hi - lo)))
    return _HUE(t)


COLORS = {a: _speedup_to_color(s) for a, s in ALGO_SPEEDUP.items()}
MARKERS = {'tome': 'o', 'gradtome': 's', 'sparge': 'D',
           'piecewise': '^', 'sparsesam': '*'}
LABELS  = {
    'tome':      f'ToMe ({ALGO_SPEEDUP["tome"]:.1f}×)',
    'gradtome':  f'GradToMe ({ALGO_SPEEDUP["gradtome"]:.1f}×)',
    'sparge':    f'SpargeAttn ({ALGO_SPEEDUP["sparge"]:.1f}×)',
    'piecewise': f'Piecewise Attn ({ALGO_SPEEDUP["piecewise"]:.1f}×)',
    'sparsesam': f'SparseSAM (ours, {ALGO_SPEEDUP["sparsesam"]:.1f}×)',
}


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 28,
        'axes.titlesize': 34,
        'axes.titleweight': 'semibold',
        'axes.labelsize': 30,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'legend.fontsize': 36,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#e6e6e6',
        'grid.linewidth': 0.8,
        'xtick.color': '#444',
        'ytick.color': '#444',
        'axes.edgecolor': '#444',
    })


def load(model: str, variant: str, base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep = pd.read_csv(os.path.join(base_dir, model, variant, 'results.csv'))
    base  = pd.read_csv(os.path.join(base_dir, model, 'baseline',  'results.csv'))
    return sweep, base


# Piecewise-Attention baseline numbers for the upper (attn-only) row.
# Each entry is {density: (mIoU, encoder_latency_relative_to_baseline)}.
# Latency is recorded but currently unused in the plot.
PIECEWISE_ATTN = {
    'DIS5K-VD': {
        0.25: (0.7679, 0.4603), 0.30: (0.7680, 0.6065),
        0.40: (0.7689, 0.6042), 0.50: (0.7838, 0.6069),
        0.60: (0.7839, 0.5967), 0.75: (0.7854, 0.5846),
    },
    'COIFT': {
        0.25: (0.9337, 0.6954), 0.30: (0.9345, 0.7179),
        0.40: (0.9353, 0.7133), 0.50: (0.9425, 0.6993),
        0.60: (0.9426, 0.6952), 0.75: (0.9451, 0.6757),
    },
    'ThinObject5K-TE': {
        0.25: (0.8774, 0.7010), 0.30: (0.8785, 0.7130),
        0.40: (0.8798, 0.7077), 0.50: (0.8926, 0.7018),
        0.60: (0.8926, 0.6935), 0.75: (0.8949, 0.6772),
    },
    'HRSOD': {
        0.25: (0.9218, 0.6985), 0.30: (0.9221, 0.7145),
        0.40: (0.9227, 0.7118), 0.50: (0.9273, 0.6998),
        0.60: (0.9273, 0.7034), 0.75: (0.9287, 0.6822),
    },
}


def _inject_algo(sw: pd.DataFrame,
                 algo: str,
                 data: dict[str, dict[float, tuple[float, float]]],
                 datasets) -> pd.DataFrame:
    """Insert a new algorithm's mIoU rows into the long-format `sw` frame
    (overwrite if rows for `algo` already exist)."""
    rows = []
    for ds, by_d in data.items():
        if ds not in datasets:
            continue
        for d, (miou, _lat) in by_d.items():
            rows.append({'algo': algo, 'dataset': ds,
                         'ratio': float(d), 'miou': float(miou),
                         'biou': 0.0, 'seed': 0})
    new_rows = pd.DataFrame(rows)
    mask = ((sw['algo'] == algo) & (sw['dataset'].isin(datasets)))
    return pd.concat([sw[~mask], new_rows], ignore_index=True)


def _draw_panel(ax, d_ds, b_mu) -> None:
    if b_mu is not None:
        ax.axhline(b_mu, ls=(0, (5, 3)), color='#475569', lw=1.4, alpha=0.9, zorder=1)
        ax.text(0.98, 0.04, f'baseline {b_mu:.3f}',
                va='bottom', ha='right', fontsize=32, color='#475569',
                transform=ax.transAxes, zorder=20,
                bbox=dict(facecolor='white', edgecolor='#cbd5e1',
                          alpha=1.0, pad=4))

    for algo in ALGO_ORDER:
        d_algo = d_ds[d_ds['algo'] == algo]
        if algo == 'tome':
            d_algo = d_algo[d_algo['ratio'] >= 0.5]
        d = (d_algo
             .groupby('ratio', as_index=False)
             .agg(miou=('miou', 'mean'),
                  sem=('miou', lambda s: s.std() / max(np.sqrt(len(s)), 1)))
             .sort_values('ratio'))
        if d.empty:
            continue
        x, y, e = d['ratio'].values, d['miou'].values, d['sem'].values
        c = COLORS[algo]
        ours = (algo == 'sparsesam')
        lw = 5.0 if ours else 4.0           # thicker overall
        ms = 520 if ours else 260           # bigger markers
        alpha_line = 1.0                    # full opacity for every algo
        z_base = 6 if ours else 3
        ax.fill_between(x, y - e, y + e, color=c,
                        alpha=0.18 if ours else 0.12, lw=0, zorder=z_base - 1)
        # A thin dark stroke under each line gives pale cyans a visible
        # silhouette without darkening the hue itself.
        ax.plot(x, y, color='#1f2937', lw=lw + 2.6,
                zorder=z_base - 1, alpha=0.35, solid_capstyle='round')
        ax.plot(x, y, color=c, lw=lw, zorder=z_base, alpha=alpha_line,
                solid_capstyle='round')
        ax.scatter(x, y, color=c, s=ms, marker=MARKERS[algo],
                   edgecolor='#1f2937',
                   linewidths=2.0 if ours else 1.6,
                   zorder=z_base + 1, label=LABELS[algo],
                   alpha=alpha_line)

    ax.set_xlim(0.20, 0.80)
    ymin = float(d_ds['miou'].min())
    ymax = float(b_mu) if b_mu is not None else float(d_ds['miou'].max())
    pad = max(0.02, 0.06 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)


def render(model: str, variant: str, base_dir: str, out_path: str) -> None:
    sweep, base = load(model, variant, base_dir)
    datasets = ['DIS5K-VD', 'ThinObject5K-TE', 'COIFT', 'HRSOD']
    datasets = [d for d in datasets if d in sweep['dataset'].unique()]

    style()
    n = len(datasets)
    ncols, nrows = n, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows),
                             dpi=160, squeeze=False)

    for ax, ds in zip(axes.flat, datasets):
        d_ds = sweep[sweep['dataset'] == ds]
        b_ds = base[base['dataset'] == ds]
        b_mu = float(b_ds['miou'].mean()) if not b_ds.empty else None
        _draw_panel(ax, d_ds, b_mu)
        ax.set_title(ds, loc='left', color='#222')
        ax.set_xlabel('density')
        ax.set_ylabel('mIoU')

    for ax in axes.flat[len(datasets):]:
        ax.axis('off')

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=len(labels),
               frameon=False, fontsize=13, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot → {out_path}")


def render_combined(model: str, base_dir: str, out_path: str) -> None:
    """Both variants stacked: top row = attn-only, bottom row = attn + MLP."""
    sw_attn,    base = load(model, 'attn_only',     base_dir)
    sw_attnmlp, _    = load(model, 'attn_plus_mlp', base_dir)
    datasets = ['DIS5K-VD', 'ThinObject5K-TE', 'COIFT', 'HRSOD']
    datasets = [d for d in datasets if d in sw_attn['dataset'].unique()]

    # Add Piecewise-Attention as an additional baseline on the upper row.
    sw_attn = _inject_algo(sw_attn, 'piecewise', PIECEWISE_ATTN, datasets)

    style()
    rows = [
        ('without MLP compressing', sw_attn),
        ('with MLP compressing',    sw_attnmlp),
    ]
    fig, axes = plt.subplots(len(rows), len(datasets),
                             figsize=(7.5 * len(datasets), 6.2 * len(rows)),
                             dpi=160, squeeze=False)

    for r, (row_label, sw) in enumerate(rows):
        for c, ds in enumerate(datasets):
            ax = axes[r, c]
            d_ds = sw[sw['dataset'] == ds]
            b_ds = base[base['dataset'] == ds]
            b_mu = float(b_ds['miou'].mean()) if not b_ds.empty else None
            _draw_panel(ax, d_ds, b_mu)
            if r == 0:
                ax.set_title(ds, loc='left', color='#222')
            if r == len(rows) - 1:
                ax.set_xlabel('density')
            else:
                ax.set_xlabel('')
            if c == 0:
                ax.set_ylabel(f'{row_label}\nmIoU', fontsize=28)
            else:
                ax.set_ylabel('')

    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Vertical colorbar on the right side of the panel grid, legend below.
    fig.tight_layout(rect=(0, 0.08, 0.93, 1))

    cax = fig.add_axes([0.945, 0.14, 0.014, 0.78])
    sm = ScalarMappable(norm=Normalize(_HUE_LO, _HUE_HI), cmap=_HUE)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='vertical')
    cb.set_label('average speedup', fontsize=24, labelpad=10)
    cb.ax.tick_params(labelsize=20)
    cb.ax.axhline(1.0, color='#475569', lw=1.2)   # mark "no speedup"

    fig.legend(handles, labels, loc='lower center', ncols=len(labels),
               frameon=False, fontsize=32, bbox_to_anchor=(0.46, 0.0),
               markerscale=2.2, handlelength=2.0,
               columnspacing=1.4, handletextpad=0.5)

    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='vit_l', choices=['vit_b', 'vit_l', 'vit_h'])
    p.add_argument('--variant', default='combined',
                   choices=['attn_only', 'attn_plus_mlp', 'combined'])
    p.add_argument('--base-dir', default='./.outputs/sam_hq44k')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    out = args.out or os.path.join(
        args.base_dir, args.model, 'plots',
        f'algo_miou_vs_ratio_{args.variant}.png',
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.variant == 'combined':
        render_combined(args.model, args.base_dir, out)
    else:
        render(args.model, args.variant, args.base_dir, out)


if __name__ == '__main__':
    main()
