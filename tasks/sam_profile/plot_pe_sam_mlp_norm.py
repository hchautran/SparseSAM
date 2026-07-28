#!/usr/bin/env python3
"""2x4 grid of per-token MLP update norm  ‖Δ_MLP‖₂.
Top row: PE-Core-L14-336 (24x24 patches).
Bottom row: SAM-HQ ViT-L (64x64 patches).
Same image, same 4 block indices.
"""

import argparse
import os
import sys
import types
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'algos', '3rd_party', 'sam-hq'))
sys.path.insert(0, os.path.join(_REPO, 'algos', '3rd_party', 'perception_models'))

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms
from segment_anything import sam_model_registry, SamPredictor


CYAN_SOLID = LinearSegmentedColormap.from_list('cyan_solid', [
    (0.00, '#f0f9fa'),
    (0.20, '#a8dee2'),
    (0.50, '#3aaab4'),
    (0.80, '#1a6c75'),
    (1.00, '#0a3a40'),
])


def _patch_pe_block(block, store):
    def patched(self, x, attn_mask=None):
        x = x + self.drop_path1(self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask)))
        x_attn = x.detach().clone()
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        store.append({'mlp_delta': (x - x_attn).detach().clone()})
        return x
    block._orig_forward = block.forward
    block.forward = types.MethodType(patched, block)


def _restore_pe_block(block):
    block.forward = block._orig_forward
    del block._orig_forward


def capture_pe_mlp_norms(model_name, image_path, block_indices, device):
    print(f'Loading {model_name} …')
    model = pe.CLIP.from_config(model_name, pretrained=True).to(device).eval()
    visual = model.visual
    grid = visual.image_size // visual.patch_size

    transform = pe_transforms.get_image_transform(visual.image_size)
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    captures = {i: [] for i in block_indices}
    for i in block_indices:
        _patch_pe_block(visual.transformer.resblocks[i], captures[i])

    with torch.no_grad():
        _ = model.encode_image(x)

    for i in block_indices:
        _restore_pe_block(visual.transformer.resblocks[i])

    norms = {}
    for i in block_indices:
        delta = captures[i][0]['mlp_delta'].float()
        if visual.use_cls_token:
            delta = delta[:, 1:, :]
        per_tok = delta.norm(dim=-1)[0].cpu().numpy()
        norms[i] = per_tok.reshape(grid, grid)
        print(f'  PE  block {i:2d}  ‖Δ_MLP‖ ∈ '
              f'[{per_tok.min():.2f}, {per_tok.max():.2f}]  mean={per_tok.mean():.2f}')

    del model
    torch.cuda.empty_cache()
    return norms


def capture_sam_mlp_norms(model_type, ckpt, image_path, block_indices, device):
    print(f'Loading SAM-HQ {model_type} from {ckpt} …')
    sam = sam_model_registry[model_type](checkpoint=ckpt).to(device).eval().half()

    captures = {i: {} for i in block_indices}
    handles = []
    for i in block_indices:
        blk = sam.image_encoder.blocks[i]

        def make_norm2_pre(idx):
            def pre(_mod, inputs):
                captures[idx]['x_attn'] = inputs[0].detach().clone()
            return pre

        def make_mlp_post(idx):
            def post(_mod, _inp, output):
                captures[idx]['mlp_delta'] = output.detach().clone()
            return post

        handles.append(blk.norm2.register_forward_pre_hook(make_norm2_pre(i)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_post(i)))

    img = np.array(Image.open(image_path).convert('RGB'))
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    for h in handles:
        h.remove()

    norms = {}
    for i in block_indices:
        mlp_delta = captures[i]['mlp_delta'].float()  # (1, H, W, C)
        per_tok = mlp_delta.norm(dim=-1)[0].cpu().numpy()  # (H, W)
        norms[i] = per_tok
        print(f'  SAM block {i:2d}  ‖Δ_MLP‖ ∈ '
              f'[{per_tok.min():.2f}, {per_tok.max():.2f}]  mean={per_tok.mean():.2f}')

    del sam
    torch.cuda.empty_cache()
    return norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='./input_imgs/butterfly.png')
    ap.add_argument('--block-indices', type=int, nargs=4, default=[0, 7, 15, 23])
    ap.add_argument('--pe-model', default='PE-Core-L14-336')
    ap.add_argument('--sam-type', default='vit_l')
    ap.add_argument('--sam-ckpt', default='./ckts/sam_hq_vit_l.pth')
    ap.add_argument('--out-dir', default='./benchmark_results')
    ap.add_argument('--out-name', default='pe_and_sam_mlp_norm_1x4')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    pe_norms = capture_pe_mlp_norms(args.pe_model, args.image,
                                     args.block_indices, args.device)
    sam_norms = capture_sam_mlp_norms(args.sam_type, args.sam_ckpt, args.image,
                                       args.block_indices, args.device)

    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'axes.titlesize':    16,
        'axes.titleweight':  'semibold',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })
    fig, axes = plt.subplots(2, 4, figsize=(3.4 * 4, 3.6 * 2), squeeze=False)

    def draw(ax, m, title):
        lo, hi = float(m.min()), float(m.max())
        norm = PowerNorm(gamma=0.5, vmin=lo, vmax=hi) if lo < hi else None
        im = ax.imshow(m, cmap=CYAN_SOLID, norm=norm, interpolation='nearest',
                       vmin=lo if norm is None else None,
                       vmax=hi if norm is None else None)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for k, i in enumerate(args.block_indices):
        draw(axes[0, k], pe_norms[i], f'block {i}' if k == 0 or True else '')
        draw(axes[1, k], sam_norms[i], '')

    axes[0, 0].set_ylabel('PE-L', fontsize=16, fontweight='semibold', labelpad=10)
    axes[1, 0].set_ylabel('SAM',  fontsize=16, fontweight='semibold', labelpad=10)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.spines['left'].set_visible(True)

    fig.suptitle(r'$\|\mathbf{\Delta}_{\mathrm{MLP}}\|_2$', fontsize=20, y=0.99)
    fig.text(0.5, 0.93,
             'per-token MLP update norm across encoder blocks',
             ha='center', va='top', fontsize=15, style='italic', color='#444')
    plt.tight_layout(rect=(0, 0, 1, 0.92))

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ('png', 'svg', 'pdf'):
        out = os.path.join(args.out_dir, f'{args.out_name}.{ext}')
        kwargs = {'dpi': 160} if ext == 'png' else {}
        fig.savefig(out, bbox_inches='tight', **kwargs)
        print(f'  → {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
