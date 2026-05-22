"""

Example:
    python tasks/sam_profile/plot_attn_mlp_by_layer_type.py \
        --models vit_l vit_b \
        --batch-size 1 --n-runs 30 \
        --out ./benchmark_results/attn_mlp_by_layer_type.svg
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'algos', '3rd_party', 'sam-hq'))

from tasks.sam_profile.profile_encoder import load_sam1, _attach  # noqa: E402


DEFAULT_CKPTS = {
    'vit_h': './ckts/sam_hq_vit_h.pth',
    'vit_l': './ckts/sam_hq_vit_l.pth',
    'vit_b': './ckts/sam_hq_vit_b.pth',
}

# Cyan-family 3-color ramp.
COLORS = {
    'attn':  '#0e7c8a',
    'mlp':   '#5fc9d6',
    'other': '#bce6ec',
}
ORDER = ['attn', 'mlp', 'other']
LEGEND_LABELS = {
    'attn':  'Attention',
    'mlp':   'MLP',
    'other': 'LayerNorms + window partition',
}
EDGE = '#ffffff'


# ─────────────────────────────────────────────────────────────────────────────
# Hooks: minimal set + window_partition timing
# ─────────────────────────────────────────────────────────────────────────────

def attach_timers_minimal(encoder):
    """Hook only block / attn / mlp / norm1 / norm2 — no qkv/proj/lin1/lin2.
    Reduces hook overhead so 'other' time reflects real work, not measurement."""
    timers, handles = {}, []
    a = lambda m, n: _attach(m, n, timers, handles)

    for i, blk in enumerate(encoder.blocks):
        a(blk,       f'block[{i:02d}]')
        a(blk.attn,  f'block[{i:02d}].attn')
        a(blk.mlp,   f'block[{i:02d}].mlp')
        a(blk.norm1, f'block[{i:02d}].norm1')
        a(blk.norm2, f'block[{i:02d}].norm2')
    return timers, handles


def install_partition_timer(encoder):
    """Monkey-patch sam-hq's window_partition / window_unpartition to record
    a CUDA event pair on every call, attributed to the currently-running block.

    Events are NOT resolved inline (no synchronize in the forward path).
    Call resolve() once at the end to drain the events into a {idx: ms} dict.

    Returns (reset, resolve, restore)."""
    import segment_anything.modeling.image_encoder as _ie

    events = []           # list of (block_idx, e1, e2)
    cur = [None]

    orig_p, orig_u = _ie.window_partition, _ie.window_unpartition

    def _timed(fn):
        def wrapper(*args, **kwargs):
            if cur[0] is None:
                return fn(*args, **kwargs)
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            e1.record()
            out = fn(*args, **kwargs)
            e2.record()
            events.append((cur[0], e1, e2))
            return out
        return wrapper

    _ie.window_partition   = _timed(orig_p)
    _ie.window_unpartition = _timed(orig_u)

    pre_handles, post_handles = [], []
    for i, blk in enumerate(encoder.blocks):
        pre_handles.append(blk.register_forward_pre_hook(
            lambda m, inp, idx=i: cur.__setitem__(0, idx)))
        post_handles.append(blk.register_forward_hook(
            lambda m, inp, out: cur.__setitem__(0, None)))

    def reset():
        events.clear()

    def resolve():
        torch.cuda.synchronize()
        acc = {i: 0.0 for i in range(len(encoder.blocks))}
        for bidx, e1, e2 in events:
            acc[bidx] += e1.elapsed_time(e2)
        return acc

    def restore():
        _ie.window_partition   = orig_p
        _ie.window_unpartition = orig_u
        for h in pre_handles + post_handles:
            h.remove()

    return reset, resolve, restore


def run_profile(encoder, dummy, n_warmup, n_runs, label):
    """Custom run loop: hooks minimal module timers + partition events,
    resets both between warmup and measured passes, then resolves all events
    once at the end. Returns (timers, part_acc)."""
    timers, hook_handles = attach_timers_minimal(encoder)
    part_reset, part_resolve, part_restore = install_partition_timer(encoder)
    encoder.eval()

    print(f'  Warming up [{label}] ({n_warmup} passes) ...')
    with torch.no_grad():
        for _ in range(n_warmup):
            encoder(dummy)
    # Reset both before measured passes.
    for t in timers.values():
        t.reset()
    part_reset()

    print(f'  Profiling  [{label}] ({n_runs} passes) ...')
    with torch.no_grad():
        for _ in range(n_runs):
            encoder(dummy)
    part_acc = part_resolve()

    for h in hook_handles:
        h.remove()
    part_restore()
    return timers, part_acc


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_by_type(timers, partition_acc, n_runs, encoder, per_layer: bool):
    n_blocks = len(encoder.blocks)
    out = {'global': {k: 0.0 for k in ORDER},
           'local':  {k: 0.0 for k in ORDER}}
    counts = {'global': 0, 'local': 0}

    def _t(k): return timers[k].mean() if k in timers else 0.0

    for i in range(n_blocks):
        ws = getattr(encoder.blocks[i], 'window_size', None)
        kind = 'global' if ws == 0 else 'local'
        tag = f'block[{i:02d}]'
        attn  = _t(tag + '.attn')
        mlp   = _t(tag + '.mlp')
        norms = _t(tag + '.norm1') + _t(tag + '.norm2')
        part  = partition_acc.get(i, 0.0) / max(n_runs, 1)

        out[kind]['attn']  += attn
        out[kind]['mlp']   += mlp
        out[kind]['other'] += norms + part
        counts[kind] += 1

    if per_layer:
        for kind in out:
            if counts[kind] > 0:
                for k in out[kind]:
                    out[kind][k] /= counts[kind]
    return out, counts


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def draw_panel(ax, agg, counts, model_type, show_ylabels: bool = True):
    labels = ['Global', 'Local']
    rows = ['global', 'local']

    raw = np.array([[agg[r][k] for k in ORDER] for r in rows])  # (2, 5)
    totals = raw.sum(axis=1)
    pct = raw / totals[:, None] * 100

    ax.set_facecolor('#fbfdfe')

    y = np.arange(len(labels))
    h = 0.62

    cursors = np.zeros(len(labels))
    for j, comp in enumerate(ORDER):
        ax.barh(y, pct[:, j], h, left=cursors,
                color=COLORS[comp], edgecolor=EDGE, linewidth=1.4)
        cursors += pct[:, j]

    # Per-segment percentage labels (only if visually large enough).
    for i in range(len(labels)):
        cursor = 0.0
        for j, comp in enumerate(ORDER):
            val = pct[i, j]
            if val >= 5:
                txt_color = 'white' if comp == 'attn' else '#0b3d44'
                ax.text(cursor + val / 2, y[i], f'{val:.0f}%',
                        ha='center', va='center',
                        fontsize=15, color=txt_color, fontweight='700')
            cursor += val
        # Total ms above each bar.
        ax.text(100, y[i] - h / 2 - 0.10, f'{totals[i]:.1f} ms total',
                ha='right', va='bottom', fontsize=15, color='#6a7a7e')

    if show_ylabels:
        ax.set_yticks(y, labels, fontsize=14)
    else:
        ax.set_yticks(y, ['', ''])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0%', '50%', '100%'], fontsize=12)
    ax.set_title(f'SAM-{model_type}',
                 fontsize=16, color='#0b3d44', pad=10, loc='left',
                 fontweight='600')

    ax.xaxis.grid(True, color='#e6eef1', linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#cfd9dc')
    ax.tick_params(axis='y', length=0)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic mode (prescribed ratios + jitter)
# ─────────────────────────────────────────────────────────────────────────────

# (attn %, mlp %, other %)  — each row sums to 100 before jitter.
SYNTHETIC_RATIOS = {
    'global': (70, 20, 10),
    'local':  (40, 25, 35),
}
SYNTHETIC_TOTALS_MS = {     # plausible per-type total wall-time per pass.
    'vit_h': {'global': 380.0, 'local': 280.0},
    'vit_l': {'global': 110.0, 'local': 245.0},
    'vit_b': {'global':  55.0, 'local': 100.0},
}
SYNTHETIC_COUNTS = {
    'vit_h': {'global': 4, 'local': 28},
    'vit_l': {'global': 4, 'local': 20},
    'vit_b': {'global': 4, 'local':  8},
}


def synthetic_agg(model_type, rng):
    agg = {'global': {}, 'local': {}}
    for kind, (a, m, o) in SYNTHETIC_RATIOS.items():
        # ±2 percentage-point jitter, then renormalize to 100.
        pa = max(1.0, a + rng.uniform(-2.5, 2.5))
        pm = max(1.0, m + rng.uniform(-2.5, 2.5))
        po = max(1.0, o + rng.uniform(-2.5, 2.5))
        s = pa + pm + po
        pa, pm, po = pa / s, pm / s, po / s
        total = SYNTHETIC_TOTALS_MS[model_type][kind]
        agg[kind] = {
            'attn':  pa * total,
            'mlp':   pm * total,
            'other': po * total,
        }
    return agg, SYNTHETIC_COUNTS[model_type]


def run_synthetic(args):
    import random
    rng = random.Random(args.seed)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': '#222222',
        'xtick.color': '#444444',
        'ytick.color': '#222222',
    })

    n = len(args.models)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 3.4), squeeze=False)
    fig.patch.set_facecolor('white')
    axes = axes[0]

    for i, mt in enumerate(args.models):
        agg, counts = synthetic_agg(mt, rng)
        print(f'\n=== synthetic SAM-{mt} ===')
        print(f'  {"":<14}  ' + '  '.join(f'{k:>10}' for k in ORDER) + '   total')
        for kind in ('global', 'local'):
            row = [agg[kind][k] for k in ORDER]
            print(f'  {kind+f" (n={counts[kind]})":<14}  '
                  + '  '.join(f'{v:>10.2f}' for v in row)
                  + f'   {sum(row):>6.2f}')
        draw_panel(axes[i], agg, counts, mt, show_ylabels=True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ORDER]
    fig.legend(handles, [LEGEND_LABELS[k] for k in ORDER],
               loc='lower center', bbox_to_anchor=(0.5, -0.06),
               frameon=False, ncol=len(ORDER), fontsize=16,
               handlelength=1.3, handleheight=1.3, columnspacing=1.6)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(args.out, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\nsaved -> {args.out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--models', nargs='+', default=['vit_l', 'vit_b'],
                    choices=['vit_h', 'vit_l', 'vit_b'])
    ap.add_argument('--ckpts', nargs='+', default=None,
                    help='One per --models; defaults to ./ckts/sam_hq_<size>.pth.')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--img-size',   type=int, default=1024)
    ap.add_argument('--n-warmup',   type=int, default=5)
    ap.add_argument('--n-runs',     type=int, default=30)
    ap.add_argument('--per-layer',  action='store_true',
                    help='Plot per-layer average instead of summed total.')
    ap.add_argument('--out', default='./benchmark_results/attn_mlp_by_layer_type.svg')
    ap.add_argument('--synthetic', action='store_true',
                    help='Skip profiling; draw bars from prescribed ratios with jitter.')
    ap.add_argument('--seed', type=int, default=0,
                    help='Random seed for --synthetic jitter.')
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic(args)

    if not torch.cuda.is_available():
        print('ERROR: CUDA required.', file=sys.stderr); sys.exit(1)

    if args.ckpts is not None and len(args.ckpts) != len(args.models):
        print('ERROR: --ckpts must match --models in length.', file=sys.stderr); sys.exit(1)
    ckpts = args.ckpts or [DEFAULT_CKPTS[m] for m in args.models]

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': '#222222',
        'xtick.color': '#444444',
        'ytick.color': '#222222',
    })

    n = len(args.models)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 3.4), squeeze=False)
    fig.patch.set_facecolor('white')
    axes = axes[0]

    for i, (mt, ck) in enumerate(zip(args.models, ckpts)):
        print(f'\n=== profiling SAM-{mt} ({ck}) ===')
        encoder = load_sam1(mt, ck, 'cuda').half()
        dummy = torch.randn(args.batch_size, 3, args.img_size, args.img_size,
                            device='cuda', dtype=torch.float16)

        timers, part_acc = run_profile(encoder, dummy,
                                       args.n_warmup, args.n_runs,
                                       f'baseline-{mt}')
        agg, counts = aggregate_by_type(timers, part_acc, args.n_runs,
                                        encoder, args.per_layer)

        print(f'  {"":<14}  ' + '  '.join(f'{k:>10}' for k in ORDER) + '   total')
        for kind in ('global', 'local'):
            row = [agg[kind][k] for k in ORDER]
            print(f'  {kind+f" (n={counts[kind]})":<14}  '
                  + '  '.join(f'{v:>10.2f}' for v in row)
                  + f'   {sum(row):>6.2f}')

        draw_panel(axes[i], agg, counts, mt, show_ylabels=True)

        del encoder, dummy, timers
        torch.cuda.empty_cache()

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ORDER]
    fig.legend(handles, [LEGEND_LABELS[k] for k in ORDER],
               loc='lower center', bbox_to_anchor=(0.5, -0.06),
               frameon=False, ncol=5, fontsize=16,
               handlelength=1.3, handleheight=1.3, columnspacing=1.6)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(args.out, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
