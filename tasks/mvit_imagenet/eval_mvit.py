#!/usr/bin/env python3
"""ImageNet-1K classification eval for `timm` MViTv2 (and other supervised
timm vision models).

This is direct supervised classification — the model has a built-in
1000-way classifier head — so the eval is just `logits = model(images)`
followed by top-K accuracy. No text encoder, no zero-shot prompts.

Examples:
  python eval_mvit.py --model mvitv2_tiny.fb_in1k --batch-size 64

  python eval_mvit.py --model mvitv2_small.fb_in1k --batch-size 32 \
      --val-root ./tasks/pe_imagenet/data/imagenet/val
"""

import os
import sys
import csv
import json
import time
import argparse
from typing import Dict, List, Optional

import torch
import timm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

from algos.registry import (
    MVIT_REGISTRY, mvit_algo_choices,
    apply_mvit as _apply_mvit, remove_all_mvit as _remove_all_mvit,
)


@torch.no_grad()
def evaluate(model, loader, device, dtype) -> Dict[str, float]:
    seen, c1, c5 = 0, 0, 0
    bar = tqdm(loader, desc="imagenet1k")
    for images, target in bar:
        images = images.to(device=device, dtype=dtype, non_blocking=True)
        target = target.to(device)
        logits = model(images)
        top5 = logits.topk(5, dim=1).indices
        c1 += (top5[:, 0] == target).sum().item()
        c5 += (top5 == target.view(-1, 1)).any(dim=1).sum().item()
        seen += target.numel()
        bar.set_postfix(acc1=f"{c1/seen*100:.2f}",
                        acc5=f"{c5/seen*100:.2f}")
    return {"acc1": c1 / seen, "acc5": c5 / seen, "n": seen}


def main():
    p = argparse.ArgumentParser(description="timm ImageNet classification eval")
    p.add_argument("--model", default="mvitv2_tiny.fb_in1k",
                   help="timm model id (any classifier-head model)")
    p.add_argument("--val-root", default="./tasks/pe_imagenet/data/imagenet/val",
                   help="ImageNet val root with one class-subdir per WordNet ID")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--output-dir", default="./benchmark_results")
    p.add_argument("--algorithm", nargs="+", default=["none"],
                   choices=mvit_algo_choices())
    p.add_argument("--ratio", nargs="+", type=float, default=[0.5])
    p.add_argument("--partial-start-block", type=int, default=0,
                   help="First block index (flattened across stages) where "
                        "the partial patch kicks in.")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--mlp-merge", dest="mlp_merge", action="store_true", default=True)
    p.add_argument("--no-mlp-merge", dest="mlp_merge", action="store_false")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    print(f"Loading model: {args.model}")
    model = timm.create_model(args.model, pretrained=True).eval().to(args.device, dtype=dtype)
    cfg = model.default_cfg
    input_size = cfg.get("input_size", (3, 224, 224))[1:]
    print(f"  input_size={input_size}  num_classes={cfg.get('num_classes', 1000)}  "
          f"crop_pct={cfg.get('crop_pct', 0.9)}  interpolation={cfg.get('interpolation', 'bicubic')}")

    # timm provides the canonical preprocessing pipeline for each checkpoint.
    transform = timm.data.create_transform(
        input_size=cfg.get("input_size", (3, 224, 224)),
        is_training=False,
        mean=cfg.get("mean", (0.485, 0.456, 0.406)),
        std=cfg.get("std", (0.229, 0.224, 0.225)),
        interpolation=cfg.get("interpolation", "bicubic"),
        crop_pct=cfg.get("crop_pct", 0.9),
        crop_mode=cfg.get("crop_mode", "center"),
    )

    print(f"Indexing val: {args.val_root}")
    ds = ImageFolder(args.val_root, transform=transform)
    print(f"  {len(ds)} images, {len(ds.classes)} classes")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Sweep configs.
    configs = []
    for algo in args.algorithm:
        spec = MVIT_REGISTRY.get(algo)
        if algo == "none" or (spec is not None and not spec.accepts_ratio):
            configs.append((algo, None))
        else:
            for r in args.ratio:
                configs.append((algo, float(r)))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"timm_imagenet_{args.model.replace('/', '_')}.csv")
    rows = []
    for algo, ratio in configs:
        tag = f"{algo}" + (f"@r={ratio}" if ratio is not None else "")
        print(f"\n############ config: {tag} ############")
        _remove_all_mvit(model)
        if algo != "none":
            _apply_mvit(model, algo, args=args, ratio=ratio)

        t0 = time.perf_counter()
        metrics = evaluate(model, loader, args.device, dtype)
        elapsed = time.perf_counter() - t0
        print(f"  → {json.dumps(metrics, default=str)}  ({elapsed:.1f}s)")

        rows.append({
            "model": args.model, "algo": algo, "ratio": ratio,
            "dataset": "imagenet1k_val",
            "mlp_merge": getattr(args, "mlp_merge", True),
            "partial_start_block": getattr(args, "partial_start_block", 0),
            "group_size": getattr(args, "group_size", 4),
            "elapsed_s": elapsed,
            **metrics,
        })

    _remove_all_mvit(model)
    keys = sorted({k for r in rows for k in r.keys()})
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, newline="") as f:
            existing_keys = next(csv.reader(f), [])
        keys = list(existing_keys) if existing_keys else keys
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n{len(rows)} row(s) appended to {csv_path}")


if __name__ == "__main__":
    main()
