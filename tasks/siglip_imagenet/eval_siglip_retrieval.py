#!/usr/bin/env python3
"""SigLIP / SigLIP 2 zero-shot image-text retrieval on COCO (Karpathy 5K test).

5000 val2014 images × 5 captions each = ~25000 caption-image pairs.
Reports Recall@{1, 5, 10} for both directions:
  * Image→Text (i2t): for each image, rank all captions; "hit" if any of
    the 5 ground-truth captions for that image lands in the top-K.
  * Text→Image (t2i): for each caption, rank all images; "hit" if the
    one ground-truth image lands in the top-K.

Hand-rolled (no clip_benchmark dependency on the retrieval side) so we
can call SigLIP's HF interface (`get_image_features`,
`get_text_features`) directly.

Examples:
  python eval_siglip_retrieval.py --model google/siglip2-base-patch16-512 \
      --batch-size 32 --dtype bf16

  python eval_siglip_retrieval.py --model google/siglip2-so400m-patch14-384 \
      --coco-root ./data/coco
"""

import os
import sys
import csv
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

from algos.registry import (
    SIGLIP_REGISTRY, siglip_algo_choices,
    apply_siglip as _registry_apply_siglip,
    remove_all_siglip as _registry_remove_siglip,
)


# ──────────────────────────────────────────────────────────────────────────
# COCO Karpathy 5K test split
# ──────────────────────────────────────────────────────────────────────────

class CocoKarpathyTest(Dataset):
    """Loads images from `<coco_root>/val2014/` driven by the Karpathy test
    JSON. Returns (PIL_image, image_id). Caption metadata is exposed via
    `self.captions` and `self.image_id_to_idx`."""

    def __init__(self, coco_root: str, transform=None):
        karpathy = os.path.join(coco_root, "coco_test_karpathy.json")
        if not os.path.exists(karpathy):
            raise FileNotFoundError(f"Karpathy test JSON not found at {karpathy}")
        with open(karpathy) as f:
            d = json.load(f)

        self.image_dir = os.path.join(coco_root, "val2014")
        self.transform = transform

        # Stable ordering: sort by image id so indices are deterministic.
        images = sorted(d["images"], key=lambda x: x["id"])
        self.image_ids: List[int] = [im["id"] for im in images]
        self.file_names: List[str] = [im["file_name"] for im in images]
        self.image_id_to_idx: Dict[int, int] = {iid: i for i, iid in enumerate(self.image_ids)}

        captions: List[str] = []
        caption_image_idx: List[int] = []
        by_iid: Dict[int, List[str]] = {}
        for ann in d["annotations"]:
            by_iid.setdefault(ann["image_id"], []).append(ann["caption"])
        for iid in self.image_ids:
            for cap in by_iid.get(iid, []):
                captions.append(cap.strip())
                caption_image_idx.append(self.image_id_to_idx[iid])
        self.captions: List[str] = captions
        self.caption_image_idx: torch.Tensor = torch.tensor(caption_image_idx, dtype=torch.long)
        gt_caps: Dict[int, List[int]] = {i: [] for i in range(len(self.image_ids))}
        for cap_idx, img_idx in enumerate(caption_image_idx):
            gt_caps[img_idx].append(cap_idx)
        self.image_to_caption_idx: List[List[int]] = [gt_caps[i] for i in range(len(self.image_ids))]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.file_names[idx])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.image_ids[idx]


class Flickr30kHFTest(Dataset):
    """Wraps `lmms-lab/flickr30k` and exposes a deterministic test subset
    of `n_test` images (default 1000, sorted by filename). This is **not**
    the official Karpathy 1k split — that requires a separate split list.
    A consistent subset is fine for differential studies (algorithm A vs B)
    but absolute recall numbers won't match published Karpathy results."""

    def __init__(self, transform=None, n_test: int = 1000):
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/flickr30k", split="test")
        # Deterministic ordering by filename.
        order = sorted(range(len(ds)), key=lambda i: ds[i]["filename"])[:n_test]
        self._ds = ds
        self._order = order
        self.transform = transform

        captions: List[str] = []
        caption_image_idx: List[int] = []
        for new_idx, orig_idx in enumerate(order):
            row = ds[orig_idx]
            for cap in row["caption"]:
                captions.append(cap.strip())
                caption_image_idx.append(new_idx)
        self.captions = captions
        self.caption_image_idx = torch.tensor(caption_image_idx, dtype=torch.long)
        gt_caps: Dict[int, List[int]] = {i: [] for i in range(len(order))}
        for cap_idx, img_idx in enumerate(caption_image_idx):
            gt_caps[img_idx].append(cap_idx)
        self.image_to_caption_idx = [gt_caps[i] for i in range(len(order))]

    def __len__(self):
        return len(self._order)

    def __getitem__(self, idx):
        row = self._ds[self._order[idx]]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, idx


# ──────────────────────────────────────────────────────────────────────────
# Encoders — separate image and text passes so we can build the full
# similarity matrix once.
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_images(model, loader, device, dtype) -> torch.Tensor:
    feats = []
    for images, _ in tqdm(loader, desc="encode images"):
        images = images.to(device=device, dtype=dtype, non_blocking=True)
        out = model.get_image_features(pixel_values=images)
        f = out.pooler_output if hasattr(out, "pooler_output") else out
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)         # (N_img, D)


@torch.no_grad()
def encode_texts(model, tokenizer, captions: List[str],
                 device, dtype, batch: int = 256) -> torch.Tensor:
    text_max = int(getattr(model.config.text_config,
                           "max_position_embeddings", 64))
    feats = []
    for i in tqdm(range(0, len(captions), batch), desc="encode texts"):
        chunk = captions[i:i + batch]
        toks = tokenizer(chunk, padding="max_length", truncation=True,
                         max_length=text_max, return_tensors="pt").to(device)
        out = model.get_text_features(**toks)
        f = out.pooler_output if hasattr(out, "pooler_output") else out
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)         # (N_cap, D)


# ──────────────────────────────────────────────────────────────────────────
# Recall metrics
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def recall_at_k(image_feats: torch.Tensor, text_feats: torch.Tensor,
                caption_image_idx: torch.Tensor,
                image_to_caption_idx: List[List[int]],
                ks: List[int] = (1, 5, 10),
                device: str = "cuda") -> Dict[str, float]:
    """Compute i2t and t2i recall@k. Similarity computed in chunks to
    keep memory bounded for the (N_img, N_cap) matrix at large N."""
    img = image_feats.to(device)
    txt = text_feats.to(device)
    N_img, D = img.shape
    N_cap = txt.shape[0]
    K_max = max(ks)

    # ── i2t: for each image, top-K captions ───────────────────────────
    i2t_top = torch.empty(N_img, K_max, dtype=torch.long, device=device)
    chunk = 256
    for i in tqdm(range(0, N_img, chunk), desc="i2t topk"):
        sim = img[i:i + chunk] @ txt.t()           # (chunk, N_cap)
        i2t_top[i:i + chunk] = sim.topk(K_max, dim=1).indices

    i2t_hits = {k: 0 for k in ks}
    for i, gt_caps in enumerate(image_to_caption_idx):
        gt_set = set(gt_caps)
        top_row = i2t_top[i].tolist()
        for k in ks:
            if any(c in gt_set for c in top_row[:k]):
                i2t_hits[k] += 1

    # ── t2i: for each caption, top-K images ───────────────────────────
    t2i_top = torch.empty(N_cap, K_max, dtype=torch.long, device=device)
    for j in tqdm(range(0, N_cap, chunk), desc="t2i topk"):
        sim = txt[j:j + chunk] @ img.t()           # (chunk, N_img)
        t2i_top[j:j + chunk] = sim.topk(K_max, dim=1).indices

    cap_image_idx = caption_image_idx.to(device)
    t2i_hits = {k: 0 for k in ks}
    for k in ks:
        hit = (t2i_top[:, :k] == cap_image_idx.unsqueeze(1)).any(dim=1)
        t2i_hits[k] = hit.sum().item()

    return {
        **{f"i2t_R@{k}": i2t_hits[k] / N_img for k in ks},
        **{f"t2i_R@{k}": t2i_hits[k] / N_cap for k in ks},
        "n_images": N_img, "n_captions": N_cap,
    }


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SigLIP zero-shot image-text retrieval (COCO 5K)")
    p.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    p.add_argument("--dataset", default="coco",
                   choices=["coco", "flickr30k"],
                   help="Retrieval dataset (coco = Karpathy 5k test, "
                        "flickr30k = first 1000 of lmms-lab/flickr30k by "
                        "filename — not the official Karpathy 1k split).")
    p.add_argument("--coco-root", default="./data/coco",
                   help="Root containing val2014/ and coco_test_karpathy.json")
    p.add_argument("--flickr-n-test", type=int, default=1000,
                   help="Number of Flickr30K images to use (deterministic subset).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--output-dir", default="./benchmark_results")
    p.add_argument("--algorithm", nargs="+", default=["none"],
                   choices=siglip_algo_choices(),
                   help="Patches to evaluate (sweep). Each (algo, ratio) "
                        "config is run as a separate row in the output CSV.")
    p.add_argument("--ratio", nargs="+", type=float, default=[0.5],
                   help="Compression ratio sweep — keep-fraction in (0, 1]. "
                        "Lower = more compression. Ignored for `none`.")
    p.add_argument("--partial-start-block", type=int, default=0,
                   help="First layer index where the partial patch kicks in.")
    p.add_argument("--group-size", type=int, default=4,
                   help="(sparsesam) Z-group size. Must divide token count.")
    p.add_argument("--mlp-merge", dest="mlp_merge", action="store_true", default=True)
    p.add_argument("--no-mlp-merge", dest="mlp_merge", action="store_false")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(args.model).eval().to(args.device, dtype=dtype)
    tokenizer  = AutoTokenizer.from_pretrained(args.model)
    image_proc = AutoImageProcessor.from_pretrained(args.model)

    sd = image_proc.size
    target = int(getattr(sd, "height", None)
                 or getattr(sd, "shortest_edge", None) or 384)
    pil_to_tv = {0: transforms.InterpolationMode.NEAREST,
                 2: transforms.InterpolationMode.BILINEAR,
                 3: transforms.InterpolationMode.BICUBIC}
    interp = pil_to_tv.get(int(getattr(image_proc, "resample", 2)),
                            transforms.InterpolationMode.BILINEAR)
    transform = transforms.Compose([
        transforms.Resize((target, target), interpolation=interp),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_proc.image_mean, std=image_proc.image_std),
    ])

    if args.dataset == "coco":
        print(f"Indexing COCO Karpathy 5K: {args.coco_root}")
        ds = CocoKarpathyTest(args.coco_root, transform=transform)
        ds_tag = "coco_5k_karpathy"
    else:
        print(f"Indexing Flickr30K (lmms-lab/flickr30k, first {args.flickr_n_test} by filename)")
        ds = Flickr30kHFTest(transform=transform, n_test=args.flickr_n_test)
        ds_tag = f"flickr30k_{args.flickr_n_test}"
    print(f"  {len(ds)} images,  {len(ds.captions)} captions")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Encode captions once — text features are independent of the vision patch.
    print("Encoding captions…")
    t0 = time.perf_counter()
    text_feats = encode_texts(model, tokenizer, ds.captions, args.device, dtype)
    t_txt = time.perf_counter() - t0

    # Build the (algo, ratio) sweep.
    configs: List[Tuple[str, Optional[float]]] = []
    for algo in args.algorithm:
        spec = SIGLIP_REGISTRY.get(algo)
        if algo == "none" or (spec is not None and not spec.accepts_ratio):
            configs.append((algo, None))
        else:
            for r in args.ratio:
                configs.append((algo, float(r)))

    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.rsplit("/", 1)[-1]
    csv_path = os.path.join(args.output_dir, f"siglip_{ds_tag.split('_')[0]}_retrieval_{model_short}.csv")
    all_rows = []

    for algo, ratio in configs:
        tag = f"{algo}" + (f"@r={ratio}" if ratio is not None else "")
        print(f"\n############ config: {tag} ############")
        _registry_remove_siglip(model)
        if algo != "none":
            _registry_apply_siglip(model, algo, args=args, ratio=ratio)

        print("Encoding images…")
        t0 = time.perf_counter()
        image_feats = encode_images(model, loader, args.device, dtype)
        t_img = time.perf_counter() - t0

        print("Computing recall…")
        metrics = recall_at_k(image_feats, text_feats,
                              ds.caption_image_idx,
                              ds.image_to_caption_idx,
                              ks=[1, 5, 10], device=args.device)
        print(f"  → {json.dumps(metrics)}  (img={t_img:.1f}s)")

        all_rows.append({
            "model": args.model, "algo": algo, "ratio": ratio,
            "dataset": ds_tag,
            "mlp_merge": getattr(args, "mlp_merge", True),
            "partial_start_block": getattr(args, "partial_start_block", 0),
            "group_size": getattr(args, "group_size", 4),
            "elapsed_img_s": t_img, "elapsed_txt_s": t_txt,
            **metrics,
        })

    _registry_remove_siglip(model)

    keys = sorted({k for r in all_rows for k in r.keys()})
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, newline="") as f:
            existing_keys = next(csv.reader(f), [])
        keys = list(existing_keys) if existing_keys else keys
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"\n{len(all_rows)} row(s) appended to {csv_path}")


if __name__ == "__main__":
    main()
