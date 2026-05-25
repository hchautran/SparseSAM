#!/usr/bin/env python3
"""Launch SAM3's official SA-Co/Gold eval (cgF1) locally, optionally with the
SparseSAM MLP-merge patch applied to the image-encoder ViT.

This is a thin wrapper around the vendored Meta SAM3 eval harness:
  * Hydra-composes a `gold_image_evals/*.yaml` config from the submodule
    (`algos/3rd_party/sam3/sam3/train/configs/`).
  * Overrides the placeholder paths in `eval_base.yaml` from CLI args, so
    the vendored YAML can stay untouched.
  * Optionally monkey-patches `sam3.model_builder.build_sam3_image_model`
    so the returned model has `algos.sparsesam.sam3.apply_patch` applied
    to its vision trunk — the trainer instantiates the model via Hydra,
    so intercepting the builder is the cleanest injection point.
  * Runs single-node, single-process (no submitit / SLURM).

Prerequisites (you must do these once before running):
  1. HF auth with an accepted SAM3 license:    hf auth login
  2. SA-Co/Gold annotations + images on disk   (see SAM3 README scripts/eval/gold)
  3. Optional: a local SAM3 checkpoint .pt     (omit to auto-download from HF)

Example:
  python tasks/sam3_saco_gold/eval_saco_gold.py \\
      --subset metaclip_nps \\
      --base-annotation-path /data/SACo-Gold/annotations \\
      --metaclip-img-path    /data/SACo-Gold/images_metaclip \\
      --base-experiment-log-dir ./sam3_logs \\
      --ratio 0.5
"""

import argparse
import os
import random
import sys
import types


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algos", "3rd_party", "sam3"))


# `sam3.train.loss.loss_fns` does an unconditional `import torchmetrics`, and
# `torchmetrics` eagerly imports its text/bert submodule, which pulls in
# `transformers`, which (in the installed 4.37) refuses to load against
# huggingface-hub>=1.0. We run in val mode with `DummyLoss`, so the single
# real torchmetrics call (`f1_score` in a training-loss method) never fires.
# Install a stub before any sam3.train.* import resolves the chain.
if "torchmetrics" not in sys.modules:
    _stub = types.ModuleType("torchmetrics")
    _stub_fn = types.ModuleType("torchmetrics.functional")

    def _stub_unreachable(*_a, **_k):
        raise RuntimeError(
            "torchmetrics.functional called from val-mode eval; "
            "this stub should be unreachable. Install the real torchmetrics "
            "if you reach this in a training context."
        )

    _stub_fn.f1_score = _stub_unreachable
    _stub.functional = _stub_fn
    sys.modules["torchmetrics"] = _stub
    sys.modules["torchmetrics.functional"] = _stub_fn


# Default BPE shipped with the vendored sam3 package.
_DEFAULT_BPE = os.path.join(
    _REPO, "algos", "3rd_party", "sam3",
    "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz",
)

# Map our --subset short names → the vendored config file names.
_SUBSET_CONFIGS = {
    "metaclip_nps": "configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml",
    "sa1b_nps":     "configs/gold_image_evals/sam3_gold_image_sa1b_nps.yaml",
    "attributes":   "configs/gold_image_evals/sam3_gold_image_attributes.yaml",
    "crowded":      "configs/gold_image_evals/sam3_gold_image_crowded.yaml",
    "wiki_common":  "configs/gold_image_evals/sam3_gold_image_wiki_common.yaml",
    "fg_food":      "configs/gold_image_evals/sam3_gold_image_fg_food.yaml",
    "fg_sports":    "configs/gold_image_evals/sam3_gold_image_fg_sports.yaml",
}


def _install_sparsesam_hook(ratio: float, mlp_merge: bool) -> None:
    """Monkey-patch `sam3.model_builder.build_sam3_image_model` so that every
    model the Hydra-driven trainer instantiates has the SparseSAM patch
    applied to its ViT trunk. No-op when `ratio >= 1.0`."""
    if ratio >= 1.0:
        return

    import sam3.model_builder as _mb
    from algos.sparsesam.sam3 import apply_patch

    _orig = _mb.build_sam3_image_model

    def _patched(*args, **kwargs):
        model = _orig(*args, **kwargs)
        apply_patch(model, ratio=float(ratio), mlp_merge=bool(mlp_merge))
        return model

    _mb.build_sam3_image_model = _patched
    print(
        f"[eval_saco_gold] sparsesam hook installed: ratio={ratio}, "
        f"mlp_merge={mlp_merge}"
    )


def _build_overrides(args) -> list[str]:
    """Hydra-style `key=value` strings to override `paths.*` and the cluster
    flag without touching the vendored eval_base.yaml."""
    ov = [
        f"paths.base_experiment_log_dir={args.base_experiment_log_dir}",
        f"paths.base_annotation_path={args.base_annotation_path}",
        f"paths.bpe_path={args.bpe_path}",
        "submitit.use_cluster=False",
        "launcher.num_nodes=1",
        f"launcher.gpus_per_node={args.num_gpus}",
    ]
    if args.base_annotation_path_silver is not None:
        ov.append(f"paths.base_annotation_path_silver={args.base_annotation_path_silver}")
    if args.metaclip_img_path is not None:
        ov.append(f"paths.metaclip_img_path={args.metaclip_img_path}")
    if args.sa1b_img_path is not None:
        ov.append(f"paths.sa1b_img_path={args.sa1b_img_path}")
    if args.silver_img_path is not None:
        ov.append(f"paths.silver_img_path={args.silver_img_path}")
    if args.checkpoint_path is not None:
        ov.append(f"paths.checkpoint_path={args.checkpoint_path}")
    return ov


def main() -> None:
    p = argparse.ArgumentParser(
        description="SAM3 SA-Co/Gold eval (cgF1) with optional SparseSAM patch",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--subset",
        type=str,
        default="metaclip_nps",
        choices=list(_SUBSET_CONFIGS.keys()),
        help="SA-Co/Gold subset to evaluate. The corresponding vendored "
        "yaml under sam3/train/configs/gold_image_evals/ is used.",
    )

    # ── paths (required by eval_base.yaml; supplied as Hydra overrides) ───
    p.add_argument("--base-experiment-log-dir", type=str, required=True,
                   help="Directory under which prediction dumps + cgF1 results go.")
    p.add_argument("--base-annotation-path", type=str, required=True,
                   help="Folder containing SA-Co/Gold annotation JSONs.")
    p.add_argument("--base-annotation-path-silver", type=str, default=None,
                   help="Silver annotation folder; only needed for silver evals.")
    p.add_argument("--metaclip-img-path", type=str, default=None,
                   help="Folder of MetaCLIP images. Required for 6/7 subsets.")
    p.add_argument("--sa1b-img-path", type=str, default=None,
                   help="Folder of SA-1B images. Required only for sa1b_nps.")
    p.add_argument("--silver-img-path", type=str, default=None,
                   help="Folder of silver images; only for silver evals.")
    p.add_argument("--bpe-path", type=str, default=_DEFAULT_BPE,
                   help=f"BPE vocab file. Default: vendored {_DEFAULT_BPE}.")
    p.add_argument("--checkpoint-path", type=str, default=None,
                   help="Local SAM3 ckpt .pt. Omit → auto-download from HF.")

    # ── runner ────────────────────────────────────────────────────────────
    p.add_argument("--num-gpus", type=int, default=1,
                   help="GPUs per node (single-node local).")

    # ── sparsesam patch ───────────────────────────────────────────────────
    p.add_argument("--ratio", type=float, default=1.0,
                   help="SparseSAM MLP-merge ratio (0,1]. 1.0 = no patch.")
    p.add_argument("--no-mlp-merge", action="store_true",
                   help="Use the patched Block class but run the full MLP "
                        "(measures patch overhead alone).")

    args = p.parse_args()

    # Sanity: subset-specific image-path requirement
    needs_metaclip = args.subset != "sa1b_nps"
    if needs_metaclip and args.metaclip_img_path is None:
        p.error(f"--metaclip-img-path is required for subset={args.subset}")
    if args.subset == "sa1b_nps" and args.sa1b_img_path is None:
        p.error("--sa1b-img-path is required for subset=sa1b_nps")

    # 1) Install the SparseSAM patch hook BEFORE Hydra instantiates the model.
    _install_sparsesam_hook(args.ratio, mlp_merge=not args.no_mlp_merge)

    # 2) Compose the vendored Hydra config with our overrides.
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf
    from sam3.train.train import single_node_runner
    from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers

    os.environ["HYDRA_FULL_ERROR"] = "1"
    register_omegaconf_resolvers()
    initialize_config_module("sam3.train", version_base="1.2")

    config_name = _SUBSET_CONFIGS[args.subset]
    overrides = _build_overrides(args)
    print(f"[eval_saco_gold] config={config_name}")
    print(f"[eval_saco_gold] overrides=\n  " + "\n  ".join(overrides))

    cfg = compose(config_name=config_name, overrides=overrides)

    # Mirror sam3.train.train.main()'s log-dir defaulting.
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam3_logs", config_name
        )
    makedir(cfg.launcher.experiment_log_dir)
    print(f"[eval_saco_gold] experiment_log_dir={cfg.launcher.experiment_log_dir}")
    print("[eval_saco_gold] resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # 3) Run the trainer in-process (single-node, single-proc when num_gpus=1).
    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config block"
    main_port = random.randint(
        submitit_conf.port_range[0], submitit_conf.port_range[1]
    )
    single_node_runner(cfg, main_port)


if __name__ == "__main__":
    main()
