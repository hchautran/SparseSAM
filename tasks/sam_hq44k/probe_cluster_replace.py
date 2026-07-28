#!/usr/bin/env python3
"""Probe: replace each token in the encoder feature map with its k-means
centroid (per image), feed the surrogate features into a frozen mask
decoder, and measure mIoU vs k.

Tests the hypothesis: the decoder only needs ~log2(k) bits per token —
i.e. tokens within an object can be identical so long as inter-object
tokens differ.

Both `features` (the main 256-d ViT output) and `interm_features[0]`
(used by SAM-HQ's HQ branch) are clustered at the same k to avoid the
HQ branch leaking unclustered information.
"""

import os
import sys
import time
import argparse
import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam-hq"))

from segment_anything import SamPredictor, sam_model_registry
from utils.data_utils import get_default_datasets
from train.utils.dataloader import get_im_gt_name_dict, Resize
from utils.data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou


# ---------- k-means ----------

@torch.no_grad()
def kmeans_replace(x: torch.Tensor, k: int, mode: str = "centroid",
                   iters: int = 15, seed: int = 0) -> torch.Tensor:
    """Cluster N tokens into k codes, replace each token with a code.

    x: (N, D).
    mode: 'centroid' — replace with k-means centroid (semantic content + partition).
          'random'   — keep k-means partition, but replace each cluster's centroid
                       with a randomly chosen real token from x. In-distribution
                       (same manifold as real tokens) but uncorrelated with the
                       cluster's actual content — isolates the role of the partition.
    """
    n, d = x.shape
    if k >= n:
        return x
    in_dtype = x.dtype
    x32 = x.float()

    g = torch.Generator(device=x.device).manual_seed(seed)
    init_idx = torch.randperm(n, generator=g, device=x.device)[:k]
    centers = x32[init_idx].clone()

    for _ in range(iters):
        d2 = torch.cdist(x32, centers)
        labels = d2.argmin(dim=1)
        sums = torch.zeros(k, d, device=x.device, dtype=torch.float32)
        counts = torch.zeros(k, device=x.device, dtype=torch.float32)
        sums.index_add_(0, labels, x32)
        counts.index_add_(0, labels, torch.ones(n, device=x.device, dtype=torch.float32))
        empty = counts == 0
        new_centers = sums / counts.clamp(min=1).unsqueeze(1)
        new_centers[empty] = centers[empty]
        if torch.allclose(centers, new_centers, atol=1e-4, rtol=0):
            centers = new_centers
            break
        centers = new_centers

    d2 = torch.cdist(x32, centers)
    labels = d2.argmin(dim=1)

    if mode == "centroid":
        codebook = centers
    elif mode == "random":
        # Pick k random real tokens from x as the codebook. Different from the
        # k-means init (which became the centroids); resampled here so the
        # codebook is uncorrelated with the partition.
        rand_idx = torch.randperm(n, generator=g, device=x.device)[:k]
        codebook = x32[rand_idx]
    else:
        raise ValueError(f"unknown mode {mode}")

    out = codebook[labels]
    return out.to(in_dtype)


def cluster_feature_map(feat: torch.Tensor, k: int, layout: str,
                        mode: str = "centroid") -> torch.Tensor:
    """Cluster a per-image feature map at k codes, replace each spatial
    token with a code (centroid or random).

    layout: 'BCHW' for (1, C, H, W) or 'BHWC' for (1, H, W, C).
    """
    assert feat.shape[0] == 1, "per-image clustering"
    if layout == 'BCHW':
        _, C, H, W = feat.shape
        x = feat[0].permute(1, 2, 0).reshape(H * W, C)
        x = kmeans_replace(x, k, mode=mode)
        return x.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0).contiguous()
    else:
        _, H, W, C = feat.shape
        x = feat[0].reshape(H * W, C)
        x = kmeans_replace(x, k, mode=mode)
        return x.reshape(1, H, W, C).contiguous()


# ---------- data ----------

def custom_collate_fn(batch):
    ori_ims = [item['ori_im'] for item in batch]
    collated = {}
    for key in batch[0].keys():
        if key == 'ori_im':
            collated[key] = ori_ims
        elif key in ('ori_im_path', 'ori_gt_path'):
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]
    return collated


def build_dataloader(ds_cfg):
    valid_im_gt_list = get_im_gt_name_dict([ds_cfg], flag="valid")
    dataset = OnlineDataset(
        [valid_im_gt_list[0]],
        transform=transforms.Compose([Resize([1024, 1024])]),
        eval_ori_resolution=True,
    )
    return DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=True, collate_fn=custom_collate_fn,
    )


# ---------- probe ----------

@torch.no_grad()
def run_probe(predictor, dataloader, ks: List[int], num_samples: int,
              modes: List[str]) -> List[Dict]:
    device = predictor.device
    rows = []

    pbar = tqdm(total=min(num_samples, len(dataloader)), desc="probe")
    seen = 0
    for batch in dataloader:
        if seen >= num_samples:
            break
        images = batch['image'].to(device)
        labels_val = batch['label']
        labels_ori = batch['ori_label'].to(device)

        if labels_val.dim() == 4:
            boxes = misc.masks_to_boxes(labels_val[:, 0, :, :])
        else:
            boxes = misc.masks_to_boxes(labels_val[0:1, :, :])
        boxes = boxes.to(device, dtype=torch.float16)

        transformed = predictor.model.preprocess(images).half()
        features, interm_features = predictor.model.image_encoder(transformed)

        # baseline (once)
        configs: List = [("baseline", -1, None)]
        for mode in modes:
            for k in ks:
                configs.append((mode, k, mode))

        for k_label_prefix, k, mode in configs:
            if mode is None:
                feats_use = features
                interm_use = interm_features
                k_label = "baseline"
            else:
                feats_use = cluster_feature_map(features, k, layout='BCHW', mode=mode)
                interm_use = [cluster_feature_map(interm_features[0], k, layout='BHWC', mode=mode)] + list(interm_features[1:])
                k_label = f"{mode}_{k}"

            predictor.features = feats_use
            predictor.interm_features = interm_use
            predictor.original_size = (images.shape[2], images.shape[3])
            predictor.input_size = tuple(transformed.shape[-2:])
            predictor.is_image_set = True

            try:
                masks, _, _ = predictor.predict_torch(
                    point_coords=None, point_labels=None,
                    boxes=boxes, hq_token_only=True,
                )
                iou = compute_iou(masks, labels_ori).item()
                biou = compute_boundary_iou(masks, labels_ori).item()
            except Exception as e:
                print(f"  decode error k={k_label}: {e}")
                iou, biou = 0.0, 0.0

            rows.append({
                'image_idx': seen,
                'k': k if k > 0 else -1,
                'mode': mode if mode is not None else 'baseline',
                'k_label': k_label,
                'iou': iou,
                'boundary_iou': biou,
            })

        seen += 1
        pbar.update(1)
    pbar.close()
    return rows


# ---------- plot ----------

def plot(df: pd.DataFrame, out_path: str, dataset_name: str) -> None:
    import matplotlib.pyplot as plt
    agg = (df.groupby(['mode', 'k'], as_index=False)
             .agg(miou=('iou', 'mean'),
                  miou_std=('iou', 'std'),
                  biou=('boundary_iou', 'mean')))
    base = agg[agg['mode'] == 'baseline']

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))
    colors = {'centroid': 'C0', 'random': 'C3'}
    nice = {'centroid': 'centroid replacement (partition + semantics)',
            'random':   'random-token replacement (partition only)'}
    for mode in ['centroid', 'random']:
        d = agg[(agg['mode'] == mode) & (agg['k'] > 0)].sort_values('k')
        if d.empty:
            continue
        ax.errorbar(d['k'], d['miou'], yerr=d['miou_std'], marker='o',
                    capsize=3, color=colors[mode], label=nice[mode])
    if not base.empty:
        b = float(base['miou'].iloc[0])
        ax.axhline(b, ls='--', color='gray', label=f'baseline mIoU = {b:.3f}')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('k (codes per image, log scale)')
    ax.set_ylabel('mIoU')
    ax.set_title(f'Decoder mIoU vs codebook size — centroid vs random replacement\n({dataset_name}, n={df["image_idx"].nunique()} imgs)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Plot → {out_path}")


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-ckt', type=str, default='./ckts/sam_hq_vit_l.pth')
    p.add_argument('--model-type', type=str, default='vit_l',
                   choices=['vit_h', 'vit_l', 'vit_b'])
    p.add_argument('--dataset-idx', type=int, default=0,
                   help='index into get_default_datasets()')
    p.add_argument('--num-samples', type=int, default=50)
    p.add_argument('--ks', type=int, nargs='+',
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024])
    p.add_argument('--modes', type=str, nargs='+',
                   default=['centroid', 'random'],
                   choices=['centroid', 'random'])
    p.add_argument('--output-dir', type=str, default='./benchmark_results/cluster_probe')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading SAM {args.model_type} from {args.model_ckt} ...")
    sam = sam_model_registry[args.model_type](checkpoint=args.model_ckt).to('cuda').half()
    predictor = SamPredictor(sam)

    datasets = get_default_datasets()
    ds_cfg = datasets[args.dataset_idx]
    print(f"Dataset: {ds_cfg['name']}  n={args.num_samples}  ks={args.ks}")

    dl = build_dataloader(ds_cfg)

    t0 = time.perf_counter()
    rows = run_probe(predictor, dl, list(args.ks), args.num_samples, args.modes)
    print(f"Probe done in {time.perf_counter()-t0:.1f}s")

    df = pd.DataFrame(rows)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv = os.path.join(args.output_dir, f'cluster_probe_{ds_cfg["name"]}_{ts}.csv')
    df.to_csv(csv, index=False)
    print(f"CSV → {csv}")

    print("\nMean per k:")
    print(df.groupby('k_label')[['iou', 'boundary_iou']].mean().round(4).to_string())

    plot_path = os.path.join(args.output_dir, f'cluster_probe_{ds_cfg["name"]}_{ts}.png')
    plot(df, plot_path, ds_cfg['name'])


if __name__ == '__main__':
    main()
