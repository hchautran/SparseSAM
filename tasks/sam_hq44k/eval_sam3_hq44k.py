#!/usr/bin/env python3
"""SAM3 (HuggingFace `Sam3Model`) HQ44K sweep — baseline vs sparsesam-patched.

Mirrors `eval_sam2_hq44k.py` but uses SAM3's text+box-prompted inference.
Patches `model.vision_encoder.backbone` (Sam3ViTModel, 32 layers, RoPE)
via `algos.sparsesam.patch.sam3_vit::apply_patch`.

For each ratio, optionally enables:
  • partial-MLP  (`prune_mlp=True`)   — same scheme as SAM-HQ ViT.
  • sparse-attn  (`sparse_attn=True`) — flex_attention with keep-bar
                                         + diag block mask.

Usage:
    python tasks/sam_hq44k/eval_sam3_hq44k.py \\
        --ratios 0.5 0.3 \\
        --batch-sizes 1 \\
        --num-samples 100 \\
        --dataset-idx 0
"""

import os
import sys
import time
import argparse
import datetime
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam-hq"))

# Reuse SAM-HQ's data pipeline (HQ44K loader + IoU helpers).
from sam_engine import get_default_datasets
from train.utils.dataloader import get_im_gt_name_dict, Resize
from utils.data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou

from transformers.models.sam3.modeling_sam3 import Sam3Model
from transformers.models.sam3.processing_sam3 import Sam3Processor


# ─────────────────────────────────────────────────────────────────────────
# Dataloader (same as eval_sam2_hq44k)
# ─────────────────────────────────────────────────────────────────────────

def _custom_collate(batch):
    ori_ims = [item["ori_im"] for item in batch]
    collated = {}
    for key in batch[0].keys():
        if key == "ori_im":
            collated[key] = ori_ims
        elif key in ("ori_im_path", "ori_gt_path"):
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]
    return collated


def build_dataloader(datasets_config, dataset_idx, batch_size):
    valid_im_gt_list = get_im_gt_name_dict(
        [datasets_config[dataset_idx]], flag="valid",
    )
    dataset = OnlineDataset(
        [valid_im_gt_list[0]],
        transform=transforms.Compose([Resize([1024, 1024])]),
        eval_ori_resolution=True,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=True, collate_fn=_custom_collate,
    )


# ─────────────────────────────────────────────────────────────────────────
# Sparsesam patch helpers
# ─────────────────────────────────────────────────────────────────────────

def apply_sparsesam(model, ratio, group_size=4, prune_mlp=True,
                    sparse_attn=False, m_block=64, n_block=64):
    from algos.sparsesam.patch.sam3_vit import apply_patch
    apply_patch(
        model.vision_encoder.backbone,
        ratio=ratio, group_size=group_size,
        prune_mlp=prune_mlp, sparse_attn=sparse_attn,
        m_block=m_block, n_block=n_block,
    )


def remove_sparsesam(model):
    from algos.sparsesam.patch.sam3_vit import remove_patch
    vit = model.vision_encoder.backbone
    if hasattr(vit, "_tome_info"):
        remove_patch(vit)


# ─────────────────────────────────────────────────────────────────────────
# Inference helpers (mirror train_sam3_hq44k.py)
# ─────────────────────────────────────────────────────────────────────────

def _xyxy_to_cxcywh_norm(box_xyxy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """xyxy (pixel) → cxcywh (normalized to [0,1])."""
    x0, y0, x1, y1 = box_xyxy.unbind(-1)
    cx = (x0 + x1) / 2 / W
    cy = (y0 + y1) / 2 / H
    w  = (x1 - x0) / W
    h  = (y1 - y0) / H
    return torch.stack([cx, cy, w, h], dim=-1)


def _box_iou_xyxy(boxes_pred: torch.Tensor, box_target: torch.Tensor) -> torch.Tensor:
    """boxes_pred (Q, 4), box_target (4,) → IoU per query (Q,)."""
    if box_target.dim() == 2:
        box_target = box_target.squeeze(0)
    bx = box_target.to(boxes_pred.dtype).to(boxes_pred.device)
    x1 = torch.maximum(boxes_pred[:, 0], bx[0])
    y1 = torch.maximum(boxes_pred[:, 1], bx[1])
    x2 = torch.minimum(boxes_pred[:, 2], bx[2])
    y2 = torch.minimum(boxes_pred[:, 3], bx[3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a_p = (boxes_pred[:, 2] - boxes_pred[:, 0]).clamp(0) * (boxes_pred[:, 3] - boxes_pred[:, 1]).clamp(0)
    a_t = (bx[2] - bx[0]).clamp(0) * (bx[3] - bx[1]).clamp(0)
    return inter / (a_p + a_t - inter + 1e-9)


def _prepare_batch(processor, images_t, labels_val, text_prompt, device):
    B, _, H, W = images_t.shape
    pil_imgs, boxes_xyxy_pix, boxes_norm = [], [], []
    for i in range(B):
        img_np = images_t[i].permute(1, 2, 0).cpu().numpy()
        img_np = ((img_np * 255).astype(np.uint8)
                  if img_np.max() <= 1.0 else img_np.astype(np.uint8))
        pil_imgs.append(Image.fromarray(img_np))
        bx = misc.masks_to_boxes(labels_val[i, 0:1])[0].cpu()
        boxes_xyxy_pix.append(bx)
        boxes_norm.append(_xyxy_to_cxcywh_norm(bx, H, W).tolist())
    input_boxes = [[b] for b in boxes_norm]
    inputs = processor(
        images=pil_imgs, text=[text_prompt] * B,
        input_boxes=input_boxes, return_tensors="pt",
    )
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    return inputs, torch.stack(boxes_xyxy_pix).to(device)


def _select_query(pred_boxes_xyxy_at_1024, gt_box_xyxy_at_1024):
    return int(_box_iou_xyxy(pred_boxes_xyxy_at_1024, gt_box_xyxy_at_1024).argmax().item())


# ─────────────────────────────────────────────────────────────────────────
# Eval driver
# ─────────────────────────────────────────────────────────────────────────

def process_batch(model, processor, images, labels_val, labels_ori,
                  text_prompt, amp_dtype, device):
    images     = images.to(device)
    labels_val = labels_val.to(device)
    labels_ori = labels_ori.to(device)
    B = images.shape[0]

    inputs, gt_boxes = _prepare_batch(processor, images, labels_val, text_prompt, device)

    amp_ctx = (torch.autocast("cuda", dtype=amp_dtype)
               if amp_dtype is not None else nullcontext())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), amp_ctx:
        outputs = model(**inputs)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000.0

    pred_masks = outputs.pred_masks                    # (B, Q, h, w)
    pred_boxes = outputs.pred_boxes                    # (B, Q, 4) at processor target_size
    scale = images.shape[-1] / float(pred_masks.shape[-1])
    pred_boxes_at_1024 = pred_boxes * scale

    ious, bious = [], []
    for b in range(B):
        q = _select_query(pred_boxes_at_1024[b], gt_boxes[b])
        m = pred_masks[b, q].unsqueeze(0).unsqueeze(0)
        m = (m.sigmoid() > 0.5).float()
        m = F.interpolate(m, size=labels_ori[b:b+1].shape[-2:], mode="nearest")
        ious.append(compute_iou(m, labels_ori[b:b+1]))
        bious.append(compute_boundary_iou(m, labels_ori[b:b+1]))

    return {
        "fwd_ms":           fwd_ms,
        "fwd_per_image_ms": fwd_ms / B,
        "iou":              torch.mean(torch.stack(ious)).item(),
        "boundary_iou":     torch.mean(torch.stack(bious)).item(),
    }


def run_one(model, processor, batch_size, dataloader, num_samples,
            text_prompt, amp_dtype, device, label="", warmup_batches=2):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    warm_iter = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            data = next(warm_iter)
        except StopIteration:
            break
        try:
            process_batch(model, processor, data["image"], data["label"],
                          data["ori_label"], text_prompt, amp_dtype, device)
        except Exception as e:
            print(f"  warmup error: {e}")
    torch.cuda.synchronize(); del warm_iter

    fwd_times, ious, bious = [], [], []
    total_images = 0
    run_all = num_samples is None or num_samples <= 0
    pbar_total = (len(dataloader) if run_all
                  else min(len(dataloader), max(1, num_samples // batch_size)))
    pbar = tqdm(total=pbar_total, desc=label or f"bs={batch_size}", leave=False)
    t0 = time.perf_counter()

    for data in dataloader:
        if not run_all and total_images >= num_samples:
            break
        try:
            m = process_batch(model, processor, data["image"], data["label"],
                              data["ori_label"], text_prompt, amp_dtype, device)
            fwd_times.append(m["fwd_per_image_ms"])
            ious.append(m["iou"]); bious.append(m["boundary_iou"])
        except Exception as e:
            print(f"  batch error: {e}")
        total_images += data["image"].shape[0]
        pbar.update(1)
    pbar.close()
    elapsed = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)
               if torch.cuda.is_available() else 0.0)

    return {
        "throughput_imgs_per_sec":     total_images / elapsed if elapsed > 0 else 0.0,
        "fwd_per_image_mean_ms":       float(np.mean(fwd_times)) if fwd_times else 0.0,
        "miou":                        float(np.mean(ious)) if ious else 0.0,
        "boundary_iou":                float(np.mean(bious)) if bious else 0.0,
        "peak_memory_allocated_mb":    float(peak_mb),
        "n_images":                    total_images,
    }


# ─────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────

def run_sweep(model, processor, ratios, batch_sizes, num_samples,
              datasets_config, dataset_indices, text_prompt, amp_dtype,
              device, prune_mlp=True, sparse_attn=False) -> List[Dict]:
    runs = [("none", 1.0)] + [("sparsesam", r) for r in ratios if r < 1.0]
    seen = set(); deduped = []
    for r in runs:
        if r not in seen:
            seen.add(r); deduped.append(r)
    runs = deduped

    print(f"\n{'='*70}")
    print(f"SAM3 sweep: {len(runs)} algo/ratio × {len(batch_sizes)} bs × "
          f"{len(dataset_indices)} dataset(s)  →  "
          f"{len(runs)*len(batch_sizes)*len(dataset_indices)} runs")
    print(f"{'='*70}\n")

    all_results = []
    for algo, ratio in runs:
        remove_sparsesam(model)
        if algo == "sparsesam":
            apply_sparsesam(model, ratio=ratio, prune_mlp=prune_mlp,
                            sparse_attn=sparse_attn)

        for bs in batch_sizes:
            for ds_idx in dataset_indices:
                ds_name = datasets_config[ds_idx]["name"]
                label = f"{algo} r={ratio:.2f} bs={bs} ds={ds_name}"
                print(f"\n── {label} ──")
                dl = build_dataloader(datasets_config, ds_idx, bs)
                m = run_one(model, processor, bs, dl, num_samples,
                            text_prompt, amp_dtype, device, label=label)
                del dl
                m.update({
                    "algo":        algo,
                    "ratio":       ratio,
                    "dataset":     ds_name,
                    "batch_size":  bs,
                    "prune_mlp":   prune_mlp if algo == "sparsesam" else None,
                    "sparse_attn": sparse_attn if algo == "sparsesam" else None,
                })
                all_results.append(m)
                print(
                    f"  [{ds_name}] thru={m['throughput_imgs_per_sec']:.2f} img/s  "
                    f"fwd/img={m['fwd_per_image_mean_ms']:.1f}ms  "
                    f"mIoU={m['miou']:.4f}  bIoU={m['boundary_iou']:.4f}  "
                    f"mem={m['peak_memory_allocated_mb']:.0f}MB"
                )

    remove_sparsesam(model)
    return all_results


def print_summary(rows: List[Dict]) -> None:
    if not rows:
        print("no rows."); return
    cols = ["algo", "ratio", "dataset", "batch_size", "miou", "boundary_iou",
            "fwd_per_image_mean_ms", "throughput_imgs_per_sec",
            "peak_memory_allocated_mb"]
    df = pd.DataFrame(rows)[cols].sort_values(
        ["dataset", "ratio"], ascending=[True, False],
    )
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 160)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",       default="facebook/sam3")
    p.add_argument("--processor",   default=None,
                   help="Processor id; defaults to --model.")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to a fine-tuned trainable-only checkpoint "
                        "(produced by `train_sam3_hq44k.py:save_checkpoint` — "
                        "key 'trainable_state_dict'). Loaded on top of "
                        "the base `--model` weights with strict=False.")
    p.add_argument("--text-prompt", default="object")
    p.add_argument("--ratios",      type=float, nargs="+", default=[0.5, 0.3])
    p.add_argument("--batch-sizes", type=int,   nargs="+", default=[1])
    p.add_argument("--num-samples", type=int,   default=100,
                   help="Per dataset (0 = full val set).")
    p.add_argument("--dataset-idx", type=int,   nargs="*", default=None)
    p.add_argument("--no-mlp-merge",  action="store_true")
    p.add_argument("--sparse-attn",   action="store_true",
                   help="Enable flex_attention sparse-attn (off by default).")
    p.add_argument("--m-block",     type=int, default=64)
    p.add_argument("--n-block",     type=int, default=64)
    p.add_argument("--amp-dtype",   default="float16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--output-dir",  default="./benchmark_results")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    amp_dtype = (torch.bfloat16 if args.amp_dtype == "bfloat16"
                 else torch.float16 if args.amp_dtype == "float16"
                 else None)
    device = "cuda"

    print(f"Loading SAM3 model + processor from {args.model} …")
    processor = Sam3Processor.from_pretrained(args.processor or args.model)
    model = Sam3Model.from_pretrained(args.model).to(device).eval()

    if args.checkpoint:
        print(f"Loading fine-tuned trainable params from {args.checkpoint} …")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("trainable_state_dict", ckpt)
        miss, unex = model.load_state_dict(sd, strict=False)
        n_loaded = sum(1 for _ in sd.keys())
        print(f"  loaded {n_loaded} trainable tensors  (missing={len(miss)} "
              f"unexpected={len(unex)})")
        if unex:
            print(f"  ⚠ first unexpected keys: {list(unex)[:3]}…")

    datasets = get_default_datasets()
    dataset_indices = (args.dataset_idx if args.dataset_idx is not None
                       else list(range(len(datasets))))

    results = run_sweep(
        model=model, processor=processor,
        ratios=args.ratios, batch_sizes=args.batch_sizes,
        num_samples=args.num_samples,
        datasets_config=datasets, dataset_indices=dataset_indices,
        text_prompt=args.text_prompt,
        amp_dtype=amp_dtype, device=device,
        prune_mlp=not args.no_mlp_merge, sparse_attn=args.sparse_attn,
    )

    print_summary(results)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = os.path.join(args.output_dir, f"sam3_hq44k_{ts}.csv")
    pd.DataFrame(results).to_csv(csv, index=False)
    print(f"\n→ wrote {csv}")


if __name__ == "__main__":
    main()
