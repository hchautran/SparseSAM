#!/usr/bin/env python3
"""SAM2 (Hiera trunk) HQ44K sweep — baseline vs sparsesam-patched encoder.

Mirrors `tasks/sam_hq44k/eval_hq44k.py` (SAM-HQ ViT path) but loads
SAM2 via `build_sam2` and patches `model.image_encoder.trunk` (Hiera)
with `algos.sparsesam.patch.sam2_hiera::apply_patch`.

Usage:
    python tasks/sam_hq44k/eval_sam2_hq44k.py \\
        --model-cfg ./sam2_configs/sam2.1/sam2.1_hiera_b+.yaml \\
        --checkpoint ./ckts/sam2.1_hiera_base_plus.pt \\
        --ratios 0.5 0.7 \\
        --batch-sizes 4 \\
        --num-samples 100
"""

import os
import sys
import time
import argparse
import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam-hq"))
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam-hq", "sam-hq2"))

# SAM-HQ's eval pipeline already provides a robust dataloader + IoU helpers;
# reuse them so the SAM2 vs SAM-HQ comparison is apples-to-apples.
from sam_engine import get_default_datasets
from train.utils.dataloader import get_im_gt_name_dict, Resize
from utils.data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou


# ─────────────────────────────────────────────────────────────────────────
# Dataloader (mirrors eval_hq44k.py:43-74)
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
        dataset,
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=True,
        collate_fn=_custom_collate,
    )


# ─────────────────────────────────────────────────────────────────────────
# SAM2 loader + patch
# ─────────────────────────────────────────────────────────────────────────

def load_sam2(config_file, ckpt_path, device="cuda"):
    """Load SAM2 from a hydra config name + ckpt. Mirrors profile_sam2.py."""
    from sam2.build_sam import build_sam2
    config_name = os.path.basename(config_file)
    if not config_name.startswith("configs/"):
        config_name = (
            f"configs/sam2.1/{config_name}" if "sam2.1" in config_name
            else f"configs/{config_name}"
        )
    print(f"Loading SAM2 config={config_name} from {ckpt_path} …")
    model = build_sam2(
        config_file=config_name, ckpt_path=ckpt_path, device=device,
        apply_postprocessing=False,
    )
    return model


def apply_sparsesam(model, ratio, group_size=4, prune_mlp=True, start_block=0):
    from algos.sparsesam.patch.sam2_hiera import apply_patch
    apply_patch(
        model.image_encoder.trunk,
        ratio=ratio, group_size=group_size,
        prune_mlp=prune_mlp,
        start_block=start_block,
    )


def remove_sparsesam(model):
    from algos.sparsesam.patch.sam2_hiera import remove_patch
    if hasattr(model.image_encoder.trunk, "_tome_info"):
        remove_patch(model.image_encoder.trunk)


# ─────────────────────────────────────────────────────────────────────────
# Per-batch eval — encoder forward + per-image decode with box prompts
# ─────────────────────────────────────────────────────────────────────────

def _set_features_from_encoder(predictor, vision_features, vision_pos_embed, batch_size):
    """Mimic predictor.set_image_batch's bookkeeping after we've done the
    encoder forward manually so we can time it. Mirrors the relevant bits
    of `SAM2ImagePredictor.set_image_batch` (sam2_image_predictor.py:134-176)."""
    feats = [
        feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
        for feat, feat_size in zip(
            vision_features[::-1], predictor._bb_feat_sizes[::-1],
        )
    ][::-1]
    predictor._features = {
        "image_embed":    feats[-1],
        "high_res_feats": feats[:-1],
    }
    predictor._is_image_set = True
    predictor._is_batch     = True


def process_batch(predictor, images, boxes, labels_ori, amp_dtype):
    """One encoder forward (timed) + per-image box-prompted decode.
    Returns timing + IoU metrics."""
    device = predictor.device
    batch_size = images.shape[0]

    images       = images.to(device)
    boxes        = boxes.to(device)
    labels_ori   = labels_ori.to(device)

    # Tell the predictor what image size we ran on (for prompt scaling).
    predictor._orig_hw = [tuple(images.shape[2:]) for _ in range(batch_size)]

    img_pp = predictor._transforms.forward_batch(
        [images[i].permute(1, 2, 0).cpu().numpy() for i in range(batch_size)]
    ).to(device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    amp_ctx = (torch.autocast("cuda", dtype=amp_dtype)
               if amp_dtype is not None else torch.autocast("cuda", enabled=False))
    with torch.no_grad(), amp_ctx:
        backbone_out = predictor.model.forward_image(img_pp)
        _, vision_feats, _, _ = predictor.model._prepare_backbone_features(
            backbone_out
        )
        if predictor.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed
        _set_features_from_encoder(
            predictor, vision_feats, None, batch_size,
        )
    torch.cuda.synchronize()
    encoder_ms = (time.perf_counter() - t0) * 1000.0

    ious, b_ious = [], []
    for i in range(batch_size):
        try:
            with torch.no_grad(), amp_ctx:
                masks, iou_pred, _ = predictor._predict(
                    point_coords=None, point_labels=None,
                    boxes=boxes[i:i+1].float(),
                    mask_input=None,
                    multimask_output=False,
                    return_logits=False,
                    img_idx=i,
                )
            # masks: (1, 1, H, W) torch.bool — match SAM-HQ's shape contract.
            ious.append(compute_iou(masks, labels_ori[i:i+1]))
            b_ious.append(compute_boundary_iou(masks, labels_ori[i:i+1]))
        except Exception as e:
            print(f"  decode error image {i}: {e}")
            ious.append(torch.tensor(0.0, device=device))
            b_ious.append(torch.tensor(0.0, device=device))

    return {
        "encoder_ms":           encoder_ms,
        "encoder_per_image_ms": encoder_ms / batch_size,
        "iou":                  torch.mean(torch.stack(ious)).item(),
        "boundary_iou":         torch.mean(torch.stack(b_ious)).item(),
    }


def run_one(predictor, batch_size, dataloader, num_samples,
            amp_dtype, label="", warmup_batches=2) -> Dict:
    """Bench one (config × batch_size × dataset)."""
    device = predictor.device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    warm_iter = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            data_val = next(warm_iter)
        except StopIteration:
            break
        images = data_val["image"].to(device)
        labels = data_val["label"]
        if labels.dim() == 4:
            boxes = misc.masks_to_boxes(labels[:, 0, :, :])
        else:
            boxes = misc.masks_to_boxes(labels[0:1, :, :])
        try:
            process_batch(predictor, images, boxes,
                          data_val["ori_label"].to(device), amp_dtype)
        except Exception as e:
            print(f"  warmup failed: {e}")
    torch.cuda.synchronize()
    del warm_iter

    enc_times, enc_per_img, ious, b_ious = [], [], [], []
    total_images = 0
    run_all = num_samples is None or num_samples <= 0
    pbar_total = (
        len(dataloader) if run_all
        else min(len(dataloader), max(1, num_samples // batch_size))
    )
    pbar = tqdm(total=pbar_total, desc=label or f"bs={batch_size}", leave=False)
    t0 = time.perf_counter()

    for data_val in dataloader:
        if not run_all and total_images >= num_samples:
            break
        images     = data_val["image"]
        labels_val = data_val["label"]
        labels_ori = data_val["ori_label"]
        if labels_val.dim() == 4:
            boxes = misc.masks_to_boxes(labels_val[:, 0, :, :])
        else:
            boxes = misc.masks_to_boxes(labels_val[0:1, :, :])
        try:
            m = process_batch(predictor, images, boxes, labels_ori, amp_dtype)
            enc_times.append(m["encoder_ms"])
            enc_per_img.append(m["encoder_per_image_ms"])
            ious.append(m["iou"]); b_ious.append(m["boundary_iou"])
        except Exception as e:
            print(f"  batch error: {e}")
        total_images += images.shape[0]
        pbar.update(1)
    pbar.close()
    elapsed = time.perf_counter() - t0

    peak_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)
               if torch.cuda.is_available() else 0.0)
    return {
        "throughput_imgs_per_sec":     total_images / elapsed if elapsed > 0 else 0.0,
        "encoder_batch_mean_ms":       float(np.mean(enc_times)) if enc_times else 0.0,
        "encoder_per_image_mean_ms":   float(np.mean(enc_per_img)) if enc_per_img else 0.0,
        "miou":                        float(np.mean(ious)) if ious else 0.0,
        "boundary_iou":                float(np.mean(b_ious)) if b_ious else 0.0,
        "peak_memory_allocated_mb":    float(peak_mb),
        "n_images":                    total_images,
    }


# ─────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────

def run_sweep(predictor, ratios, batch_sizes, num_samples,
              datasets_config, dataset_indices, amp_dtype,
              prune_mlp: bool = True, start_block: int = 0) -> List[Dict]:
    runs = [("none", 1.0)] + [("sparsesam", r) for r in ratios if r < 1.0]
    seen = set(); deduped = []
    for r in runs:
        if r not in seen:
            seen.add(r); deduped.append(r)
    runs = deduped

    print(f"\n{'='*70}")
    print(f"SAM2 sweep: {len(runs)} algo/ratio × {len(batch_sizes)} bs "
          f"× {len(dataset_indices)} dataset(s)  =  "
          f"{len(runs)*len(batch_sizes)*len(dataset_indices)} runs")
    print(f"{'='*70}\n")

    all_results = []
    for algo, ratio in runs:
        # Reset patch state.
        remove_sparsesam(predictor.model)
        if algo == "sparsesam":
            apply_sparsesam(predictor.model, ratio=ratio,
                            prune_mlp=prune_mlp, start_block=start_block)

        for bs in batch_sizes:
            for ds_idx in dataset_indices:
                ds_name = datasets_config[ds_idx]["name"]
                label = f"{algo} r={ratio:.2f} bs={bs} ds={ds_name}"
                print(f"\n── {label} ──")
                dl = build_dataloader(datasets_config, ds_idx, bs)
                m = run_one(predictor, bs, dl, num_samples, amp_dtype, label=label)
                del dl
                m.update({
                    "algo":        algo,
                    "ratio":       ratio,
                    "dataset":     ds_name,
                    "batch_size":  bs,
                    "prune_mlp":   prune_mlp if algo == "sparsesam" else None,
                    "start_block": start_block if algo == "sparsesam" else None,
                })
                all_results.append(m)
                print(
                    f"  [{ds_name}] thru={m['throughput_imgs_per_sec']:.2f} img/s  "
                    f"enc/img={m['encoder_per_image_mean_ms']:.1f}ms  "
                    f"mIoU={m['miou']:.4f}  bIoU={m['boundary_iou']:.4f}  "
                    f"mem={m['peak_memory_allocated_mb']:.0f}MB"
                )

    remove_sparsesam(predictor.model)
    return all_results


def print_summary(rows: List[Dict]) -> None:
    if not rows:
        print("no rows."); return
    cols = ["algo", "ratio", "dataset", "batch_size",
            "miou", "boundary_iou",
            "encoder_per_image_mean_ms", "throughput_imgs_per_sec",
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
    p.add_argument("--model-cfg", required=True,
                   help="Path to SAM2 hydra config (e.g. "
                        "./sam2_configs/sam2.1/sam2.1_hiera_b+.yaml)")
    p.add_argument("--checkpoint", required=True,
                   help="Path to SAM2 checkpoint .pt")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.7],
                   help="Keep ratios for sparsesam.")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--num-samples", type=int, default=100,
                   help="Images per dataset (0 = full val set).")
    p.add_argument("--dataset-idx", type=int, nargs="*", default=None,
                   help="Subset of HQ44K datasets to eval; default all.")
    p.add_argument("--no-mlp-merge", action="store_true",
                   help="Disable partial-MLP path inside sparsesam (debug).")
    p.add_argument("--start-block", type=int, default=0,
                   help="Only apply sparsesam to blocks with index >= this. "
                        "Useful for ablations like 'patch only blocks after the "
                        "first global' (set to first_global_idx + 1, e.g. 24 "
                        "for HQ-Hiera-L whose first global is at block 23).")
    p.add_argument("--amp-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--output-dir", default="./benchmark_results")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    amp_dtype = (torch.bfloat16 if args.amp_dtype == "bfloat16"
                 else torch.float16 if args.amp_dtype == "float16"
                 else None)

    model = load_sam2(args.model_cfg, args.checkpoint, device="cuda")
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(model)

    datasets = get_default_datasets()
    dataset_indices = (args.dataset_idx if args.dataset_idx is not None
                       else list(range(len(datasets))))

    results = run_sweep(
        predictor=predictor,
        ratios=args.ratios,
        batch_sizes=args.batch_sizes,
        num_samples=args.num_samples,
        datasets_config=datasets,
        dataset_indices=dataset_indices,
        amp_dtype=amp_dtype,
        prune_mlp=not args.no_mlp_merge,
        start_block=args.start_block,
    )

    print_summary(results)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = os.path.join(args.output_dir, f"sam2_hq44k_{ts}.csv")
    pd.DataFrame(results).to_csv(csv, index=False)
    print(f"\n→ wrote {csv}")


if __name__ == "__main__":
    main()
