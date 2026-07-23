#!/usr/bin/env python3
"""End-to-end SAM-HQ timing on L4: image_encoder + prompt_encoder + mask_decoder
(+ mask upscaling), baseline vs sparsesam. Random weights + random box prompts
(no checkpoint/data needed — timing doesn't depend on trained weights).

Mirrors tasks/sam_hq44k/eval_hq44k.py process_batch: one encoder forward per image,
then predict_torch(boxes, hq_token_only=True) per image. Reports encoder-only vs
end-to-end speedup at P prompts/image so you can see how the (unaccelerated)
decoder overhead eats into the encoder speedup (Amdahl).
"""
import os, sys, time, argparse, statistics
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam-hq"))

from segment_anything import sam_model_registry, SamPredictor
from algos.registry import apply_sam, remove_all_sam

# Fix a dtype bug that surfaces under .half(): forward_with_coords forces coords to
# float32 but the gaussian matrix is half -> matmul dtype mismatch. Cast to match.
import numpy as np
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom
def _pe_encoding(self, coords):
    coords = 2 * coords - 1
    coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
    coords = coords @ self.positional_encoding_gaussian_matrix
    coords = 2 * np.pi * coords
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
PositionEmbeddingRandom._pe_encoding = _pe_encoding


@torch.no_grad()
def run_once(predictor, x, boxes_list, img_size):
    """One full forward over a batch: encoder then per-image decode. Returns
    (encoder_ms, decode_ms)."""
    model = predictor.model
    B = x.shape[0]

    torch.cuda.synchronize(); t0 = time.perf_counter()
    features, interm = model.image_encoder(x)      # encoder runs in fp16
    torch.cuda.synchronize(); enc_ms = (time.perf_counter() - t0) * 1000

    # decoder runs in fp32 (this sam-hq HEAD isn't half-clean); bridge at boundary
    features = features.float(); interm = [f.float() for f in interm]

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(B):
        predictor.features        = features[i:i+1]
        predictor.interm_features = [f[i:i+1] for f in interm]
        predictor.original_size   = (img_size, img_size)
        predictor.input_size      = (img_size, img_size)
        predictor.is_image_set    = True
        predictor.predict_torch(point_coords=None, point_labels=None,
                                boxes=boxes_list[i], hq_token_only=True)
    torch.cuda.synchronize(); dec_ms = (time.perf_counter() - t0) * 1000
    return enc_ms, dec_ms


def bench(predictor, x, boxes_list, img_size, iters, warmup):
    for _ in range(warmup):
        run_once(predictor, x, boxes_list, img_size)
    encs, decs = [], []
    for _ in range(iters):
        e, d = run_once(predictor, x, boxes_list, img_size)
        encs.append(e); decs.append(d)
    return statistics.median(encs), statistics.median(decs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", default="vit_l")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompts", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=1024)
    args = ap.parse_args()

    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} cc{torch.cuda.get_device_capability(0)}")
    print(f"model={args.model_type} batch={args.batch_size} img={args.img_size}\n")

    sam = sam_model_registry[args.model_type](checkpoint=None).to(dev)  # fp32 model
    sam.image_encoder.half()                                            # encoder fp16 only
    predictor = SamPredictor(sam)
    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=dev, dtype=torch.float16)

    def rand_boxes(P):
        xy = torch.rand(P, 2, device=dev) * (args.img_size * 0.5)
        wh = torch.rand(P, 2, device=dev) * (args.img_size * 0.4) + 20
        return torch.cat([xy, xy + wh], dim=1).to(torch.float16)

    for P in args.prompts:
        boxes_list = [rand_boxes(P) for _ in range(args.batch_size)]

        # baseline
        remove_all_sam(sam.image_encoder)
        be, bd = bench(predictor, x, boxes_list, args.img_size, args.iters, args.warmup)
        base_enc, base_e2e = be, be + bd
        print(f"--- P={P} prompts/image ---")
        print(f"{'algo':10s} {'enc_ms':>8s} {'dec_ms':>8s} {'e2e_ms':>8s} "
              f"{'enc_spd':>7s} {'e2e_spd':>7s}")
        print(f"{'baseline':10s} {be:>8.1f} {bd:>8.1f} {base_e2e:>8.1f} "
              f"{1.0:>7.2f} {1.0:>7.2f}")

        for r in args.ratios:
            remove_all_sam(sam.image_encoder)
            apply_sam(sam.image_encoder, "sparsesam", ratio=r)
            e, d = bench(predictor, x, boxes_list, args.img_size, args.iters, args.warmup)
            e2e = e + d
            print(f"{'sparse'+f'{r:.2f}':10s} {e:>8.1f} {d:>8.1f} {e2e:>8.1f} "
                  f"{base_enc/e:>7.2f} {base_e2e/e2e:>7.2f}")
        remove_all_sam(sam.image_encoder)
        print()


if __name__ == "__main__":
    main()
