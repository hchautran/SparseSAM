#!/usr/bin/env python3
"""Full end-to-end SAM-HQ latency with a per-stage breakdown.

Unlike bench_e2e_l4.py (which times only image_encoder + prompt_encoder +
mask_decoder on an already-resident GPU tensor), this walks the whole pipeline a
real application pays for, exactly as SamPredictor.set_image / .predict do:

  load       cv2.imread + BGR->RGB                      (disk + CPU, optional)
  preprocess ResizeLongestSide (PIL) -> HWC->CHW -> H2D
             -> Sam.preprocess (normalize + pad to 1024) -> fp16
  encoder    image_encoder                              (fp16, sparsified)
  prompt     transform.apply_boxes (CPU) + prompt_encoder
  decoder    mask_decoder (hq_token_only)
  postproc   Sam.postprocess_masks (2x interpolate to original size) + threshold
  d2h        masks/iou -> CPU numpy

Stage timings sync the GPU at every boundary; that sync overhead is not free, so
a second sync-free pass measures the honest wall-clock e2e. Both are reported.

Random weights by default (latency doesn't depend on trained weights); pass
--checkpoint to use real ones. Encoder runs fp16, prompt/decoder/postproc fp32,
mirroring tasks/sam_hq44k/eval_hq44k.py.
"""
import os, sys, time, glob, argparse, statistics
from collections import defaultdict

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam-hq"))

import cv2
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from algos.registry import apply_sam, remove_all_sam

STAGES = ["load", "preprocess", "encoder", "prompt", "decoder", "postproc", "d2h"]


class Timer:
    """Accumulates per-stage ms. sync=False turns every stage into a no-op so the
    same pipeline code can be run sync-free for the wall-clock measurement."""

    def __init__(self, sync=True):
        self.sync = sync
        self.acc = defaultdict(float)

    def __call__(self, name):
        return _Span(self, name)


class _Span:
    def __init__(self, timer, name):
        self.t, self.name = timer, name

    def __enter__(self):
        if self.t.sync:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.t.sync:
            torch.cuda.synchronize()
            self.t.acc[self.name] += (time.perf_counter() - self.t0) * 1000
        return False


@torch.no_grad()
def run_image(sam, transform, item, tm, dev, include_load):
    """One image through the whole pipeline. `item` is (path, image_or_None, boxes)."""
    path, cached_img, boxes_orig = item

    with tm("load"):
        if include_load:
            bgr = cv2.imread(path)
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            image = cached_img
    original_size = image.shape[:2]

    with tm("preprocess"):
        resized = transform.apply_image(image)                       # CPU, PIL bilinear
        t = torch.as_tensor(resized, device=dev)
        t = t.permute(2, 0, 1).contiguous()[None, :, :, :].float()   # 1x3xHxW, H2D done
        input_size = tuple(t.shape[-2:])
        x = sam.preprocess(t).half()                                 # normalize + pad + fp16

    with tm("encoder"):
        features, interm = sam.image_encoder(x)
    # fp16 encoder -> fp32 decoder bridge (this sam-hq HEAD is not half-clean).
    # Charged to the encoder stage: it exists only because the encoder ran fp16.
    with tm("encoder"):
        features = features.float()
        interm = [f.float() for f in interm]

    with tm("prompt"):
        boxes = transform.apply_boxes(boxes_orig, original_size)     # CPU, numpy
        boxes = torch.as_tensor(boxes, dtype=torch.float, device=dev)
        sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=boxes, masks=None)
        image_pe = sam.prompt_encoder.get_dense_pe()

    with tm("decoder"):
        low_res_masks, iou_pred = sam.mask_decoder(
            image_embeddings=features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            hq_token_only=True,
            interm_embeddings=interm,
        )

    with tm("postproc"):
        masks = sam.postprocess_masks(low_res_masks, input_size, original_size)
        masks = masks > sam.mask_threshold

    with tm("d2h"):
        masks_np = masks.detach().cpu().numpy()
        iou_np = iou_pred.detach().cpu().numpy()

    return masks_np, iou_np


def bench(sam, transform, items, dev, iters, warmup, include_load):
    """Returns (per-stage median ms per image, sync-free wall ms per image)."""
    n = len(items)
    for _ in range(warmup):
        tm = Timer(sync=False)
        for it in items:
            run_image(sam, transform, it, tm, dev, include_load)

    # staged pass (syncs at every boundary)
    per_iter = []
    for _ in range(iters):
        tm = Timer(sync=True)
        for it in items:
            run_image(sam, transform, it, tm, dev, include_load)
        per_iter.append(dict(tm.acc))
    stages = {s: statistics.median([p.get(s, 0.0) for p in per_iter]) / n for s in STAGES}

    # sync-free pass (honest wall clock)
    walls = []
    for _ in range(iters):
        tm = Timer(sync=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for it in items:
            run_image(sam, transform, it, tm, dev, include_load)
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1000 / n)
    return stages, statistics.median(walls)


def rand_boxes(h, w, P, seed):
    """P random XYXY boxes in the original image frame, as a user would supply."""
    rng = np.random.RandomState(seed)
    x0 = rng.rand(P) * w * 0.5
    y0 = rng.rand(P) * h * 0.5
    x1 = np.minimum(x0 + rng.rand(P) * w * 0.4 + 20, w - 1)
    y1 = np.minimum(y0 + rng.rand(P) * h * 0.4 + 20, h - 1)
    return np.stack([x0, y0, x1, y1], axis=1).astype(np.float32)


def print_table(rows, base):
    head = (f"{'algo':14s} " + " ".join(f"{s:>10s}" for s in STAGES)
            + f" {'sum':>9s} {'wall':>9s} {'spd':>6s} {'img/s':>6s}")
    print(head)
    print("-" * len(head))
    for name, stages, wall in rows:
        tot = sum(stages.values())
        print(f"{name:14s} " + " ".join(f"{stages[s]:>10.2f}" for s in STAGES)
              + f" {tot:>9.2f} {wall:>9.2f} {base/wall:>5.2f}x {1000/wall:>6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", default="vit_l")
    ap.add_argument("--checkpoint", default=None, help="optional .pth; random weights if omitted")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--images", default=os.path.join(_REPO, "input_imgs", "*"),
                    help="glob of real images; preprocessing/postprocessing cost scales "
                         "with their native resolution")
    ap.add_argument("--synthetic", type=int, nargs=2, metavar=("H", "W"), default=None,
                    help="use one synthetic HxW image instead of --images")
    ap.add_argument("--prompts", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--include-load", action="store_true",
                    help="re-read images from disk every iteration (adds decode + I/O; "
                         "off by default so results aren't dominated by page cache)")
    args = ap.parse_args()

    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} cc{torch.cuda.get_device_capability(0)}")

    if args.synthetic:
        h, w = args.synthetic
        imgs = [("<synthetic>", np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))]
    else:
        paths = [p for p in sorted(glob.glob(args.images))
                 if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        if not paths:
            raise SystemExit(f"no images matched {args.images}")
        imgs = []
        for p in paths:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            imgs.append((p, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    sizes = sorted({im.shape[:2] for _, im in imgs})
    print(f"model={args.model_type} images={len(imgs)} sizes={sizes}")
    print(f"iters={args.iters} warmup={args.warmup} include_load={args.include_load}\n")

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(dev)
    sam.eval()
    sam.image_encoder.half()
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    for P in args.prompts:
        items = [(p, im, rand_boxes(im.shape[0], im.shape[1], P, seed=i))
                 for i, (p, im) in enumerate(imgs)]
        print(f"=== P={P} prompt(s)/image — ms per image ===")

        remove_all_sam(sam.image_encoder)
        b_stages, b_wall = bench(sam, transform, items, dev, args.iters, args.warmup,
                                 args.include_load)
        rows = [("baseline", b_stages, b_wall)]

        for r in args.ratios:
            remove_all_sam(sam.image_encoder)
            apply_sam(sam.image_encoder, "sparsesam", ratio=r)
            s, w = bench(sam, transform, items, dev, args.iters, args.warmup, args.include_load)
            rows.append((f"sparsesam {r:.2f}", s, w))
        remove_all_sam(sam.image_encoder)

        print_table(rows, b_wall)
        non_enc = b_wall - b_stages["encoder"]
        print(f"\nbaseline non-encoder floor: {non_enc:.2f} ms/image "
              f"({100*non_enc/b_wall:.1f}% of e2e) — Amdahl ceiling "
              f"{b_wall/non_enc:.2f}x however fast the encoder gets\n")


if __name__ == "__main__":
    main()
