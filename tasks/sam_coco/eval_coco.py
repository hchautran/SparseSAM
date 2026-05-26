#!/usr/bin/env python3
"""SAM-HQ COCO sweep over (algo x ratio x batch_size).

Reports encoder latency, peak memory, and COCO segm AP metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "algos" / "3rd_party" / "sam-hq"))

from utils.coco_mmdet import default_detector_config, setup_reference_paths
from utils.sam_precision import enable_sam_dtype_for_predictor
from algos.registry import apply_sam, remove_all_sam, sam_algo_choices

VALID_ALGOS = sam_algo_choices()


def reset_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()



class ImageEncoderCudaProfiler:
    """Measure SAM image-encoder latency without including detector/mask decode."""

    def __init__(self) -> None:
        self._event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._wrapped = False

    def wrap_forward(self, module: torch.nn.Module) -> None:
        if self._wrapped or not torch.cuda.is_available():
            return
        original_forward = module.forward

        def wrapped_forward(*args, **kwargs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = original_forward(*args, **kwargs)
            end_event.record()
            self._event_pairs.append((start_event, end_event))
            return output

        module.forward = wrapped_forward
        self._wrapped = True

    def summarize(self) -> Dict[str, float]:
        if not self._event_pairs:
            return {
                "encoder_batch_mean_ms": 0.0,
                "encoder_batch_std_ms": 0.0,
                "encoder_per_image_mean_ms": 0.0,
                "encoder_per_image_std_ms": 0.0,
            }
        torch.cuda.synchronize()
        elapsed_ms = np.array([start.elapsed_time(end) for start, end in self._event_pairs], dtype=np.float64)
        return {
            "encoder_batch_mean_ms": float(np.mean(elapsed_ms)),
            "encoder_batch_std_ms": float(np.std(elapsed_ms)),
            "encoder_per_image_mean_ms": float(np.mean(elapsed_ms)),
            "encoder_per_image_std_ms": float(np.std(elapsed_ms)),
        }

def _set_cfg_data_root(cfg, data_root: str | None) -> None:
    if not data_root or not hasattr(cfg, "data"):
        return
    root = data_root.rstrip("/") + "/"
    for split in ("train", "val", "test"):
        if split in cfg.data and isinstance(cfg.data[split], dict):
            real_split = "val" if split == "test" else split
            cfg.data[split]["ann_file"] = root + f"annotations/instances_{real_split}2017.json"
            cfg.data[split]["img_prefix"] = root + f"{real_split}2017/"


def _limit_dataset(dataset, num_samples: int | None) -> None:
    if not num_samples or num_samples <= 0:
        return
    if hasattr(dataset, "data_infos"):
        dataset.data_infos = dataset.data_infos[:num_samples]
    if hasattr(dataset, "img_ids"):
        dataset.img_ids = dataset.img_ids[:num_samples]


def _prepare_mmdet(args):
    sam_quant_root = setup_reference_paths(args.sam_quant_root)

    from mmcv import Config
    from mmcv.cnn import fuse_conv_bn
    from mmcv.runner import get_dist_info, init_dist, wrap_fp16_model
    from mmdet.apis import multi_gpu_test, single_gpu_test
    from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
    from mmdet.models import build_detector
    from mmdet.utils import build_ddp, build_dp, compat_cfg, get_device, replace_cfg_vals, setup_multi_processes, update_data_root

    from tasks.sam_coco.configs.det_observer_instance_sam_ import DetObserverInstanceSAM  # noqa: F401

    config = args.config or default_detector_config(args.detector, args.model_type, sam_quant_root)
    cfg = Config.fromfile(config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if not hasattr(cfg, "total_epochs"):
        cfg.total_epochs = cfg.get("runner", {}).get("max_epochs", 1)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if "pretrained" in cfg.model:
        cfg.model.pretrained = None
    elif cfg.model.get("backbone", None) is not None and "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    cfg.model.model_type = args.model_type
    cfg.model.sam_checkpoint = None if str(args.det_sam_ckt).lower() in {"none", "null", ""} else args.det_sam_ckt
    if args.det_checkpoint is not None:
        if "det_model_ckpt" in cfg.model:
            cfg.model.det_model_ckpt = args.det_checkpoint
        if "det_wrapper_cfg" in cfg.model and "det_weight" in cfg.model.det_wrapper_cfg:
            cfg.model.det_wrapper_cfg.det_weight = args.det_checkpoint
    if "det_wrapper_cfg" in cfg.model and "det_config" in cfg.model.det_wrapper_cfg:
        local_yolox = _REPO / "tasks" / "sam_coco" / "configs" / "yolox" / "yolox_l_8x8_300e_coco.py"
        if args.detector == "yolox" and local_yolox.exists():
            cfg.model.det_wrapper_cfg.det_config = str(local_yolox)
    cfg.model.show_image = args.show_images
    if args.show_dir is not None:
        cfg.model.result_coco_path = args.show_dir
    _set_cfg_data_root(cfg, args.data_root)

    cfg.gpu_ids = args.gpu_ids[0:1] if args.gpu_ids is not None else [args.gpu_id]
    cfg.device = get_device()
    distributed = args.launcher != "none"
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        "samples_per_gpu": args.batch_size,
        "workers_per_gpu": args.workers_per_gpu,
        "dist": distributed,
        "shuffle": False,
        **cfg.data.get("test_dataloader", {}),
    }
    test_loader_cfg["samples_per_gpu"] = args.batch_size
    test_loader_cfg["workers_per_gpu"] = args.workers_per_gpu

    dataset = build_dataset(cfg.data.test)
    _limit_dataset(dataset, args.num_samples)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model.CLASSES = dataset.CLASSES

    return {
        "cfg": cfg,
        "config": config,
        "dataset": dataset,
        "data_loader": data_loader,
        "model": model,
        "distributed": distributed,
        "single_gpu_test": single_gpu_test,
        "multi_gpu_test": multi_gpu_test,
        "build_dp": build_dp,
        "build_ddp": build_ddp,
        "get_dist_info": get_dist_info,
    }





def _warmup_image_encoder(predictor, warmup_batches: int = 3) -> None:
    if warmup_batches <= 0 or not torch.cuda.is_available():
        return
    encoder_dtype = next(predictor.model.image_encoder.parameters()).dtype
    image_size = predictor.model.image_encoder.img_size
    device = predictor.device
    with torch.no_grad():
        for _ in range(warmup_batches):
            image = torch.zeros((1, 3, image_size, image_size), device=device, dtype=encoder_dtype)
            input_image = predictor.model.preprocess(image).to(dtype=encoder_dtype)
            predictor.model.image_encoder(input_image)
    torch.cuda.synchronize()

def _build_predictor(args, algo: str, ratio: float):
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[args.model_type](checkpoint=args.model_ckt).to("cuda")
    predictor = SamPredictor(sam)
    sam_dtype = torch.float16
    enable_sam_dtype_for_predictor(predictor, dtype=sam_dtype)
    remove_all_sam(predictor.model.image_encoder, mask_decoder=predictor.model.mask_decoder)
    if algo != "none" and ratio < 1.0:
        apply_sam(
            predictor.model.image_encoder,
            algo,
            args=args,
            ratio=ratio,
            margin=args.margin,
            mlp_merge=args.mlp_merge,
        )
    return predictor


def _evaluate_one(args, algo: str, ratio: float, batch_size: int) -> Dict[str, Any]:
    reset_memory()
    args.batch_size = batch_size
    env = _prepare_mmdet(args)
    cfg = env["cfg"]
    model = env["model"]
    dataset = env["dataset"]
    data_loader = env["data_loader"]

    predictor = _build_predictor(args, algo, ratio)
    _warmup_image_encoder(predictor, args.encoder_warmup_batches)
    profiler = ImageEncoderCudaProfiler()
    profiler.wrap_forward(predictor.model.image_encoder)

    model.det_model.to(cfg.device)
    model.replace_quant_sam(predictor)
    model.predictor.model.to(cfg.device)
    model.to(cfg.device)

    rank, _ = env["get_dist_info"]()
    if not env["distributed"]:
        wrapped = env["build_dp"](model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = env["single_gpu_test"](
            wrapped,
            data_loader,
            args.show,
            args.show_dir,
            args.show_score_thr,
            gt=(args.show_dir is not None and "gt" in args.show_dir),
        )
    else:
        wrapped = env["build_ddp"](
            model,
            cfg.device,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            broadcast_buffers=False,
        )
        outputs = env["multi_gpu_test"](wrapped, data_loader, args.tmpdir, args.gpu_collect or cfg.evaluation.get("gpu_collect", False))

    metrics: Dict[str, Any] = {}
    if rank == 0:
        if args.out:
            import mmcv

            out_path = Path(args.out)
            if len(args.algos) > 1 or len(args.ratios) > 1 or len(args.batch_sizes) > 1:
                out_path = out_path.with_name(f"{out_path.stem}_{algo}_r{ratio:.2f}_bs{batch_size}{out_path.suffix}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mmcv.dump(outputs, str(out_path))

        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals", "metric"]:
                eval_kwargs.pop(key, None)
            if args.eval_options is not None:
                eval_kwargs.update(args.eval_options)
            metrics = dataset.evaluate(outputs, metric=args.eval, **eval_kwargs)
            print(metrics)

    metrics.update(profiler.summarize())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        metrics.update({
            "peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_memory_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        })

    metrics.update({
        "algo": algo,
        "ratio": ratio,
        "batch_size": batch_size,
        "num_images": len(dataset),
        "sam_dtype": str(next(predictor.model.parameters()).dtype).replace("torch.", ""),
        "detector": args.detector,
        "model_type": args.model_type,
        "config": env["config"],
        "timestamp": dt.datetime.now().isoformat(),
    })
    return metrics


def _runs(algos: Iterable[str], ratios: Iterable[float]) -> List[tuple[str, float]]:
    runs: List[tuple[str, float]] = []
    for algo in algos:
        if algo == "none":
            runs.append(("none", 1.0))
        else:
            runs.extend((algo, float(r)) for r in ratios if float(r) < 1.0)
    deduped = []
    seen = set()
    for run in runs:
        if run not in seen:
            seen.add(run)
            deduped.append(run)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SparseSAM algorithms on COCO through PTQ4SAM/MMDet")
    parser.add_argument("--sam-quant-root", default=os.environ.get("SAM_QUANT_ROOT", "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization"))
    parser.add_argument("--config", default=None, help="MMDet detector config. Defaults from --detector and --model-type.")
    parser.add_argument("--detector", default="dino", choices=["yolox", "dino", "hdetr"])
    parser.add_argument("--det-checkpoint", default=None, help="Optional override for cfg.model.det_model_ckpt")
    parser.add_argument("--data-root", default=None, help="COCO root containing val2017/ and annotations/.")
    parser.add_argument("--model-ckt", default="./ckts/sam_hq_vit_l.pth", help="SAM-HQ checkpoint used by the injected predictor under test.")
    parser.add_argument("--det-sam-ckt", default="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/ckts/sam_vit_l_0b3195.pth", help="Original SAM checkpoint used only to build the MMDet wrapper before predictor injection.")
    parser.add_argument("--model-type", default="vit_l", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--algos", nargs="+", default=["none", "sparsesam", "tome", "gradtome", "sparge", "piecewise"], choices=VALID_ALGOS)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.75])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--num-samples", type=int, default=None, help="Optional quick-run limit on COCO images.")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--mlp-merge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--piecewise-block-size", type=int, default=64)
    parser.add_argument("--output-dir", default="./benchmark_results/sam_coco")
    parser.add_argument("--out", default="./demo/coco/results.pkl")
    parser.add_argument("--eval", nargs="+", default=["segm"])
    parser.add_argument("--format-only", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--show-dir", default=None)
    parser.add_argument("--show-images", type=int, default=0)
    parser.add_argument("--show-score-thr", type=float, default=0.3)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-ids", type=int, nargs="+")
    parser.add_argument("--gpu-collect", action="store_true")
    parser.add_argument("--tmpdir", default=None)
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--fuse-conv-bn", action="store_true")
    parser.add_argument("--cfg-options", nargs="+")
    parser.add_argument("--eval-options", nargs="+")
    parser.add_argument("--encoder-warmup-batches", type=int, default=3)
    return parser.parse_args()


def run_sweep(args: argparse.Namespace) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    runs = _runs(args.algos, args.ratios)
    print(f"COCO sweep: {len(runs)} algo/ratio configs x {len(args.batch_sizes)} batch size(s)")
    for algo, ratio in runs:
        for batch_size in args.batch_sizes:
            print(f"\n--- {algo} r={ratio:.2f} bs={batch_size} ---")
            result = _evaluate_one(args, algo, ratio, batch_size)
            results.append(result)
            print(
                f"segm_mAP={result.get('segm_mAP', float('nan')):.3f} "
                f"segm_mAP_50={result.get('segm_mAP_50', float('nan')):.3f} "
                f"enc/img={result.get('encoder_per_image_mean_ms', float('nan')):.1f} ms "
                f"peak_mem={result.get('peak_memory_allocated_mb', float('nan')):.0f} MB"
            )
    return results


def main() -> List[Dict[str, Any]]:
    args = parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    setup_reference_paths(args.sam_quant_root)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results = run_sweep(args)
    csv = Path(args.output_dir) / f"sam_coco_eval_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(results).to_csv(csv, index=False)
    print(f"\nResults saved -> {csv}")
    return results


if __name__ == "__main__":
    main()
