#!/usr/bin/env python3
"""mIoU / Boundary-IoU for SAM-HQ on L4, baseline vs sparsesam, box-prompted.

Reuses the repo's dataloader, GT->box derivation (misc.masks_to_boxes), and IoU
metrics, but runs the image encoder in fp16 (required by the sparsesam CUTE kernel)
and the prompt-encoder / mask-decoder in fp32. Reason: this sam-hq HEAD is not
half-clean (prompt-encoder forces coords to fp32 then matmuls a half buffer), so a
full .half() model — as in eval_hq44k.process_batch — throws inside decode and,
because that path swallows exceptions, would silently report IoU=0 for every image.
A small _pe_encoding patch + the fp32 decoder avoids that. fp32 decode is if anything
higher-precision, so baseline mIoU is faithful.
"""
import os, sys, argparse
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam-hq"))

from segment_anything import sam_model_registry, SamPredictor
from algos.registry import apply_sam, remove_all_sam
from train.utils.dataloader import get_im_gt_name_dict, Resize
from utils.data_utils import OnlineDataset
import torch.nn.functional as F
import train.utils.misc as misc

# Inlined from train.train (which pulls in the whole training stack); these only
# use misc. preds: BxCxHxW logits/masks; target: Bx1xHxW.
def compute_iou(preds, target):
    if preds.shape[2:] != target.shape[2:]:
        preds = F.interpolate(preds, size=target.shape[2:], mode="bilinear", align_corners=False)
    return sum(misc.mask_iou(preds[i], target[i]) for i in range(len(preds))) / len(preds)

def compute_boundary_iou(preds, target):
    if preds.shape[2:] != target.shape[2:]:
        preds = F.interpolate(preds, size=target.shape[2:], mode="bilinear", align_corners=False)
    return sum(misc.boundary_iou(target[i], preds[i]) for i in range(len(preds))) / len(preds)

# fp16/fp32 dtype fix for the random-gaussian positional encoding.
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom
def _pe_encoding(self, coords):
    coords = 2 * coords - 1
    coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
    coords = coords @ self.positional_encoding_gaussian_matrix
    coords = 2 * np.pi * coords
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
PositionEmbeddingRandom._pe_encoding = _pe_encoding

COIFT = {
    "name": "COIFT",
    "im_dir": "./data/thin_object_detection/COIFT/images",
    "gt_dir": "./data/thin_object_detection/COIFT/masks",
    "im_ext": ".jpg", "gt_ext": ".png",
}


def collate(batch):
    out = {}
    for k in batch[0]:
        if k in ("ori_im", "ori_im_path", "ori_gt_path", "imidx"):
            out[k] = [b[k] for b in batch]
        else:
            try: out[k] = torch.stack([b[k] for b in batch])
            except Exception: out[k] = [b[k] for b in batch]
    return out


def build_loader(cfg):
    lst = get_im_gt_name_dict([cfg], flag="valid")
    ds = OnlineDataset([lst[0]], transform=transforms.Compose([Resize([1024, 1024])]),
                       eval_ori_resolution=True)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                      collate_fn=collate)


@torch.no_grad()
def evaluate(predictor, loader, num_samples):
    dev = predictor.device
    ious, bious = [], []
    for data in loader:
        if len(ious) >= num_samples:
            break
        images = data["image"].to(dev)
        labels = data["label"]                 # 1024-frame mask
        labels_ori = data["ori_label"].to(dev)
        boxes = misc.masks_to_boxes(labels[:, 0, :, :] if labels.dim() == 4 else labels[0:1])
        boxes = boxes.to(dev, dtype=torch.float16)

        transformed = predictor.model.preprocess(images).half()
        feats, interm = predictor.model.image_encoder(transformed)
        feats = feats.float(); interm = [f.float() for f in interm]

        predictor.features = feats
        predictor.interm_features = interm
        predictor.original_size = (images.shape[2], images.shape[3])
        predictor.input_size = tuple(transformed.shape[-2:])
        predictor.is_image_set = True

        masks, _, _ = predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=boxes.float(), hq_token_only=True, multimask_output=False)
        ious.append(compute_iou(masks, labels_ori).item())
        bious.append(compute_boundary_iou(masks, labels_ori).item())
    return float(np.mean(ious)), float(np.mean(bious)), len(ious)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", default="vit_l")
    ap.add_argument("--model-ckt", default="./ckts/sam_hq_vit_l.pth")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--num-samples", type=int, default=280)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  model={args.model_type}  ckpt={args.model_ckt}")
    sam = sam_model_registry[args.model_type](checkpoint=args.model_ckt).to("cuda")
    sam.image_encoder.half()                       # encoder fp16, decoder fp32
    predictor = SamPredictor(sam)
    loader = build_loader(COIFT)

    print(f"\nCOIFT box-prompted  (n={args.num_samples})")
    print(f"{'algo':16s} {'mIoU':>8s} {'B-IoU':>8s} {'n':>5s}")
    print("-" * 40)

    remove_all_sam(sam.image_encoder)
    miou, biou, n = evaluate(predictor, loader, args.num_samples)
    base = miou
    print(f"{'baseline':16s} {miou:>8.4f} {biou:>8.4f} {n:>5d}")

    for r in args.ratios:
        remove_all_sam(sam.image_encoder)
        apply_sam(sam.image_encoder, "sparsesam", ratio=r)
        miou, biou, n = evaluate(predictor, loader, args.num_samples)
        print(f"{'sparsesam '+f'{r:.2f}':16s} {miou:>8.4f} {biou:>8.4f} {n:>5d}   "
              f"(Δ mIoU {miou-base:+.4f})")
    remove_all_sam(sam.image_encoder)


if __name__ == "__main__":
    main()
