"""Small COCO/MMDet helpers for the SparseSAM COCO task.

These helpers intentionally avoid the old quantization/calibration argument
surface. They only keep the MMDet test arguments required by PTQ4SAM's detector
wrapper and SparseSAM's SAM algorithm sweep.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from mmcv import DictAction
except Exception:  # pragma: no cover - imported only in the COCO env
    DictAction = None


DEFAULT_SAM_QUANT_ROOT = Path("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization")


def setup_reference_paths(sam_quant_root: str | os.PathLike | None = None) -> Path:
    """Add PTQ4SAM and its CUDA ops from the legacy environment checkout."""
    root = Path(sam_quant_root or os.environ.get("SAM_QUANT_ROOT", DEFAULT_SAM_QUANT_ROOT)).resolve()
    ptq4sam = root / "PTQ4SAM"
    ops = ptq4sam / "projects" / "instance_segment_anything" / "ops"
    for path in (ptq4sam, ops):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)
    return root


def default_detector_config(detector: str, model_type: str, sam_quant_root: str | os.PathLike | None = None) -> str:
    del sam_quant_root
    config_model = model_type.replace("_", "-")
    repo_root = Path(__file__).resolve().parents[2]
    config_root = repo_root / "tasks" / "sam_coco" / "configs"
    table = {
        "yolox": config_root / "yolox" / f"yolo_l-sam-{config_model}.py",
        "dino": config_root / "focalnet_dino" / f"focalnet-l-dino_sam-{config_model}.py",
        "hdetr": config_root / "hdetr" / f"r50-hdetr_sam-{config_model}.py",
    }
    return str(table[detector])


def parse_argsptq4sam(argv=None):
    """Compatibility parser for the PTQ4SAM/MMDet test flags used by COCO.

    This is deliberately minimal: quantization, calibration, DUO training, and
    DiffDUO training options are not part of the refactored COCO task.
    """
    parser = argparse.ArgumentParser(description="MMDet COCO eval with SAM predictor")
    parser.add_argument("--config", default=None, help="MMDet config file path")
    parser.add_argument("--work-dir", default="result/tmp")
    parser.add_argument("--out", default="./demo/coco/results.pkl")
    parser.add_argument("--fuse-conv-bn", action="store_true")
    parser.add_argument("--gpu-ids", type=int, nargs="+")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--format-only", action="store_true")
    parser.add_argument("--eval", type=str, nargs="+", default=["segm"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--show-dir")
    parser.add_argument("--show-score-thr", type=float, default=0.3)
    parser.add_argument("--gpu-collect", action="store_true")
    parser.add_argument("--tmpdir")
    if DictAction is not None:
        parser.add_argument("--cfg-options", nargs="+", action=DictAction)
        parser.add_argument("--eval-options", nargs="+", action=DictAction)
    else:
        parser.add_argument("--cfg-options", nargs="+")
        parser.add_argument("--eval-options", nargs="+")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--detector", type=str, default="dino", choices=["yolox", "dino", "hdetr"])
    args, unknown = parser.parse_known_args(argv)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args, unknown
