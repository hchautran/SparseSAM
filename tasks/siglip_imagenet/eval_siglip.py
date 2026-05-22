#!/usr/bin/env python3
"""Zero-shot ImageNet-1K classification with SigLIP / SigLIP 2 (HuggingFace).

Mirrors the PE-on-ImageNet eval (`eval_pe_clip.py`) but bypasses
`clip_benchmark.metrics.zeroshot_classification` so we can call SigLIP's
HF interface directly (`get_image_features`, `get_text_features`).
clip_benchmark is reused only to build the val dataset and to source
class names + prompt templates.

SigLIP's pretraining uses sigmoid (not softmax) loss, but for zero-shot
classification we still take argmax of the unnormalized logits — the
ranking is the same.

Examples:
  python eval_siglip.py --model google/siglip2-so400m-patch14-384 \
      --batch-size 32 --dtype bf16

  # Override dataset root
  python eval_siglip.py --model google/siglip-base-patch16-224 \
      --dataset-root ./tasks/pe_imagenet/data/imagenet --batch-size 64
"""

import os
import sys
import csv
import json
import time
import argparse
import datetime
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
PM_PE = os.path.join(_REPO, "algos", "3rd_party", "perception_models", "apps", "pe")
sys.path.insert(0, _REPO)
sys.path.insert(0, PM_PE)              # for `clip_benchmark.*`

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from clip_benchmark.datasets.builder import (
    build_dataset, get_dataset_collate_fn,
)


# ──────────────────────────────────────────────────────────────────────────
# Text feature bank: encode all class names × templates once, mean over
# templates, L2-normalize → (n_classes, D) classifier weights.
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_text_classifier(model, tokenizer, classnames: List[str],
                          templates: List[str], device: str, dtype) -> torch.Tensor:
    """Build (n_classes, D) classifier weights by averaging text features
    over templates. SigLIP 2's text encoder caps at 64 tokens
    (`text_config.max_position_embeddings`); the GemmaTokenizer it ships
    with reports `model_max_length=1e18` (sentinel), so we pin it to 64
    explicitly here rather than relying on the tokenizer default."""
    text_max = int(getattr(model.config.text_config,
                           "max_position_embeddings", 64))
    classifier = []
    for cn in tqdm(classnames, desc="text features"):
        prompts = [t.format(c=cn) for t in templates]
        toks = tokenizer(prompts, padding="max_length", truncation=True,
                         max_length=text_max, return_tensors="pt").to(device)
        out = model.get_text_features(**toks)
        # transformers v5+ returns BaseModelOutputWithPooling; older versions
        # may return a bare tensor. Handle both.
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = F.normalize(feats.float(), dim=-1)
        feats = feats.mean(dim=0)
        feats = F.normalize(feats, dim=-1)
        classifier.append(feats)
    return torch.stack(classifier, dim=0).to(device=device, dtype=dtype)   # (n_classes, D)


# ──────────────────────────────────────────────────────────────────────────
# Image features → top-K classification.
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_imagenet(model, loader, classifier, device, dtype,
                      desc: str = "imagenet1k") -> Dict[str, float]:
    classifier_t = classifier.t().contiguous()  # (D, n_classes)
    seen, c1, c5 = 0, 0, 0
    bar = tqdm(loader, desc=desc)
    for images, target in bar:
        images = images.to(device=device, dtype=dtype, non_blocking=True)
        target = target.to(device)
        out = model.get_image_features(pixel_values=images)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = F.normalize(feats.float(), dim=-1).to(dtype)
        logits = feats @ classifier_t                     # (B, n_classes)
        top5 = logits.topk(5, dim=1).indices
        c1 += (top5[:, 0] == target).sum().item()
        c5 += (top5 == target.view(-1, 1)).any(dim=1).sum().item()
        seen += target.numel()
        bar.set_postfix(acc1=f"{c1/seen*100:.2f}",
                        acc5=f"{c5/seen*100:.2f}")
    return {"acc1": c1 / seen, "acc5": c5 / seen, "n": seen}


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SigLIP / SigLIP 2 zero-shot ImageNet")
    p.add_argument("--model", default="google/siglip2-so400m-patch14-384",
                   help="HF model id (any Siglip / Siglip2 dual-encoder)")
    p.add_argument("--dataset-root", default="./tasks/pe_imagenet/data/imagenet",
                   help="ImageNet-1k root (passed to clip_benchmark)")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--output-dir", default="./benchmark_results")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(args.model).eval().to(args.device, dtype=dtype)
    tokenizer  = AutoTokenizer.from_pretrained(args.model)
    image_proc = AutoImageProcessor.from_pretrained(args.model)

    # Build a torchvision transform pipeline matching the HF processor.
    # SigLIP's `resample` is PIL's int (2=BILINEAR, 3=BICUBIC, 0=NEAREST);
    # using the wrong one costs accuracy on zero-shot ImageNet.
    from torchvision import transforms
    sd = image_proc.size
    target = int(getattr(sd, "height", None)
                 or getattr(sd, "shortest_edge", None) or 384)
    pil_to_tv = {
        0: transforms.InterpolationMode.NEAREST,
        2: transforms.InterpolationMode.BILINEAR,
        3: transforms.InterpolationMode.BICUBIC,
    }
    interp = pil_to_tv.get(int(getattr(image_proc, "resample", 2)),
                            transforms.InterpolationMode.BILINEAR)
    transform = transforms.Compose([
        transforms.Resize((target, target), interpolation=interp),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_proc.image_mean, std=image_proc.image_std),
    ])
    print(f"  preprocess: resize ({target},{target}) interp={interp.value}  "
          f"mean={image_proc.image_mean}  std={image_proc.image_std}")

    print(f"Indexing imagenet1k (val split): {args.dataset_root}")
    ds = build_dataset(
        dataset_name="imagenet1k", root=args.dataset_root, transform=transform,
        split=args.split, download=False, task="zeroshot_classification",
    )
    classnames = ds.classes
    templates  = ds.templates
    if not classnames or not templates:
        raise RuntimeError("imagenet1k dataset did not expose classnames/templates")
    print(f"  {len(ds)} images,  {len(classnames)} classes,  {len(templates)} templates,  "
          f"input={target}x{target}")

    collate = get_dataset_collate_fn("imagenet1k")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=collate)

    print("Building text classifier…")
    t0 = time.perf_counter()
    classifier = build_text_classifier(model, tokenizer, classnames, templates,
                                       args.device, dtype)
    t_text = time.perf_counter() - t0
    print(f"  text features: {tuple(classifier.shape)}  took {t_text:.1f}s")

    print("Running zero-shot ImageNet…")
    t0 = time.perf_counter()
    metrics = evaluate_imagenet(model, loader, classifier, args.device, dtype)
    t_eval = time.perf_counter() - t0
    print(f"\n→ {json.dumps(metrics, default=str)}  ({t_eval:.1f}s)")

    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.rsplit("/", 1)[-1]
    csv_path = os.path.join(args.output_dir, f"siglip_imagenet_{model_short}.csv")
    row = {
        "model": args.model, "algo": "none", "ratio": None,
        "dataset": "imagenet1k", "split": args.split,
        "elapsed_s": t_eval, "elapsed_text_s": t_text,
        **metrics,
    }
    keys = sorted(row.keys())
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, newline="") as f:
            existing_keys = next(csv.reader(f), [])
        keys = list(existing_keys) if existing_keys else keys
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"Result appended to {csv_path}")


if __name__ == "__main__":
    main()
