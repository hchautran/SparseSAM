"""Unified algorithm registry for the SparseSAM + baselines workflow.

One `AlgoSpec` dataclass, one `REGISTRY` keyed by `(backbone, name)`, and a
single `register()` entry point. Per-backbone helpers (`apply_pe` / `apply_sam`
+ matching `remove_all_*`) are thin wrappers that exist because each backbone
has a different stock-class set that `remove_all_*` walks; the *apply* side
is fully generic.

Adding a new algorithm: write the patch, then add one `register(AlgoSpec(...))`
call inside `_register_builtins()` below. See `docs/ADDING_ALGORITHMS.md`.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class AlgoSpec:
    name: str
    backbone: str                              # "sam" | "pe"
    apply: Callable[..., Any]
    kwargs_from_args: Optional[Callable[[Any, Optional[float]], dict]] = None
    accepts_ratio: bool = True
    category: str = "compress"                 # "compress" | "partial" | "attention"
    description: str = ""
    # SAM-only: classes used by `remove_all_sam` to revert subclassed modules.
    block_class: Optional[type] = None
    attn_class: Optional[type] = None


REGISTRY: Dict[Tuple[str, str], AlgoSpec] = {}


def register(spec: AlgoSpec) -> AlgoSpec:
    key = (spec.backbone, spec.name)
    if key in REGISTRY:
        raise ValueError(
            f"{spec.backbone} algorithm {spec.name!r} already registered"
        )
    REGISTRY[key] = spec
    return spec


def specs_for(backbone: str) -> Dict[str, AlgoSpec]:
    """All registered specs for `backbone`, keyed by algo name."""
    return {n: s for (bb, n), s in REGISTRY.items() if bb == backbone}


def choices(backbone: str) -> List[str]:
    """Valid algo names for `backbone`, including the no-op 'none' baseline."""
    return ["none"] + sorted(specs_for(backbone).keys())


def spec_of(backbone: str, name: str) -> Optional[AlgoSpec]:
    return REGISTRY.get((backbone, name))


def _apply(model, backbone: str, name: str, args=None,
           ratio: Optional[float] = None, **extra):
    if name == "none":
        return None
    if backbone == "sam" and ratio is not None and ratio >= 1.0:
        return None  # ratio>=1.0 ⇒ keep all tokens, nothing to merge

    spec = spec_of(backbone, name)
    if spec is None:
        raise KeyError(
            f"Unknown {backbone} algorithm {name!r}. "
            f"Valid choices: {choices(backbone)}"
        )

    if spec.kwargs_from_args is not None:
        kwargs = spec.kwargs_from_args(args, ratio)
    elif backbone == "sam":
        kwargs = dict(ratio=ratio)
    else:
        kwargs = {}
    kwargs.update(extra)
    return spec.apply(model, **kwargs)


def apply_pe(model, name, args=None, ratio=None):
    return _apply(model, "pe", name, args, ratio)

def apply_sam(encoder, name, args=None, ratio=None, **extra):
    return _apply(encoder, "sam", name, args, ratio, **extra)


def algo_choices()     -> List[str]: return choices("pe")
def sam_algo_choices() -> List[str]: return choices("sam")


def _revert_subclasses(model, stock_classes: Tuple[type, ...]) -> int:
    n = 0
    for module in model.modules():
        cls = type(module)
        for stock in stock_classes:
            if cls is not stock and issubclass(cls, stock):
                module.__class__ = stock
                n += 1
                break
    return n


def _safe_delattr(obj, *attrs):
    for a in attrs:
        if hasattr(obj, a):
            try:
                delattr(obj, a)
            except AttributeError:
                pass


def remove_all_pe(model) -> int:
    """Revert every PE patch and clear per-module patch state. Idempotent."""
    try:
        from core.vision_encoder.pe import SelfAttention, ResidualAttentionBlock
    except Exception:
        return 0

    n = _revert_subclasses(model, (SelfAttention, ResidualAttentionBlock))
    for module in model.modules():
        for hook in ("_pe_compress_pre_hook", "_pe_compress_post_hook",
                     "_pe_partial_pre_hook", "_pe_partial_post_hook"):
            handle = getattr(module, hook, None)
            if handle is not None:
                try: handle.remove()
                except Exception: pass
                delattr(module, hook)
        _safe_delattr(module, "_flash_cos", "_flash_sin",
                      "_flash_cos_sin_key", "_tome_info")
    _safe_delattr(model, "_tome_info")
    return n


def remove_all_sam(encoder, mask_decoder=None) -> int:
    """Idempotently undo every SAM patch: revert subclassed Block/Attention
    modules and clear encoder.tome_info + any instance-level forward overrides.
    `mask_decoder` accepted for API symmetry (currently unused)."""
    from segment_anything.modeling.image_encoder import Block, Attention

    sam_specs = list(specs_for("sam").values())
    block_classes = tuple({s.block_class for s in sam_specs if s.block_class})
    attn_classes  = tuple({s.attn_class  for s in sam_specs if s.attn_class})

    n = 0
    for module in encoder.modules():
        t = type(module)
        if block_classes and t in block_classes:
            module.__class__ = Block; n += 1
        elif attn_classes and t in attn_classes:
            module.__class__ = Attention; n += 1

    _safe_delattr(encoder, "tome_info")
    encoder.__dict__.pop("forward", None)
    for blk in getattr(encoder, "blocks", []):
        blk.__dict__.pop("forward", None)
    return n


def update_sam_ratio(encoder, ratio: float):
    """Update merge ratio without re-patching (works if already patched)."""
    if hasattr(encoder, "tome_info"):
        encoder.tome_info["ratio"] = ratio


# kwargs_from_args helpers translate (argparse.Namespace, ratio) into the
# kwarg dict each patch's apply_* expects. Names follow `_kw_<backbone>_<flavor>`.

def _kw_pe_compress(args, ratio):
    cab = getattr(args, "compress_at_blocks", None)
    return dict(
        ratio=float(ratio if ratio is not None else args.ratio[0]),
        num_stages=int(args.num_stages),
        group_size=int(args.group_size),
        use_flash_rope=bool(getattr(args, "use_flash_rope", False)),
        compress_at_blocks=list(cab) if cab else None,
    )


def _kw_pe_partial_basic(args, ratio):
    return dict(
        ratio=float(ratio if ratio is not None else args.ratio[0]),
        start_block=int(getattr(args, "partial_start_block", 0)),
        mlp_merge=bool(getattr(args, "mlp_merge", True)),
    )


def _kw_pe_partial_sparsesam(args, ratio):
    return dict(
        ratio=float(ratio if ratio is not None else args.ratio[0]),
        group_size=int(getattr(args, "group_size", 4)),
        sparse_ratio=getattr(args, "sparse_ratio", None),
        start_block=int(getattr(args, "partial_start_block", 0)),
        mlp_merge=bool(getattr(args, "mlp_merge", True)),
    )


def _kw_pe_flash_rope(args, ratio):
    return {}


def _kw_sparge_ratio(args, ratio):
    return dict(
        ratio=float(ratio if ratio is not None
                    else (args.ratio[0] if hasattr(args, "ratio") else 1.0)),
    )


def _kw_sam_basic(args, ratio):
    return dict(
        ratio=float(ratio if ratio is not None else args.ratios[0]),
        margin=float(getattr(args, "margin", 0.5)),
        mlp_merge=bool(getattr(args, "mlp_merge", True)),
        piecewise_block_size=int(getattr(args, "piecewise_block_size", 64)),
    )


def _bake_sam_apply(apply_fn, internal_algo: str):
    """Wrap a SAM patch's `apply_patch` with `algo=internal_algo` pre-bound.
    Optional kwargs are forwarded only when accepted by the patch signature,
    so older patches without `mlp_merge` (etc.) keep working unchanged."""
    accepted = set(inspect.signature(apply_fn).parameters)

    def _apply_baked(encoder, ratio, margin=0.5, **extra):
        kw = dict(algo=internal_algo, ratio=ratio, margin=margin)
        for k, v in extra.items():
            if k in accepted:
                kw[k] = v
        return apply_fn(encoder, **kw)
    return _apply_baked


# Per-algo registration errors (keyed by "backbone:name"); inspect after
# import to see what got skipped when an optional dep is missing.
REGISTRATION_ERRORS: Dict[str, Exception] = {}


def _safe_register(backbone, name, import_and_make_spec):
    try:
        spec = import_and_make_spec()
        register(spec)
    except Exception as e:
        REGISTRATION_ERRORS[f"{backbone}:{name}"] = e


def _register_pe():
    def mk_tome():
        from .tome.pe_compress import apply_pe_tome_patch
        return AlgoSpec("tome", "pe", apply_pe_tome_patch,
            kwargs_from_args=_kw_pe_compress, category="compress",
            description="Bipartite-soft-matching merge at every stage boundary.")
    def mk_gradtome():
        from .gradtome.pe_compress import apply_pe_gradtome_patch
        return AlgoSpec("gradtome", "pe", apply_pe_gradtome_patch,
            kwargs_from_args=_kw_pe_compress, category="compress",
            description="Spatial-gradient-aware bipartite matching on (H,W) grid.")
    def mk_sparsesam():
        from .sparsesam.pe_compress import apply_pe_sparsesam_patch
        return AlgoSpec("sparsesam", "pe", apply_pe_sparsesam_patch,
            kwargs_from_args=_kw_pe_compress, category="compress",
            description="SparseSAM Z-group merge: keep top-K groups, average the rest.")
    def mk_flash_rope():
        from .sparsesam.pe_compress import apply_pe_flash_rope_patch
        return AlgoSpec("flash_rope", "pe", apply_pe_flash_rope_patch,
            kwargs_from_args=_kw_pe_flash_rope, accepts_ratio=False, category="attention",
            description="Standalone fused FA2 + 2D-axial RoPE cute kernel.")
    def mk_sparge():
        from .sparge.pe import apply_pe_sparge_patch
        return AlgoSpec("sparge", "pe", apply_pe_sparge_patch,
            kwargs_from_args=_kw_sparge_ratio, category="attention",
            description="SpargeAttn top-k attention swap. `ratio` is forwarded as `topk`.")
    def mk_tome_partial():
        from .tome.pe_partial import apply_pe_tome_partial_patch
        return AlgoSpec("tome_partial", "pe", apply_pe_tome_partial_patch,
            kwargs_from_args=_kw_pe_partial_basic, category="partial",
            description="Full-Q + ToMe-merged-K/V SDPA + (optional) merge/MLP/unmerge.")
    def mk_gradtome_partial():
        from .gradtome.pe_partial import apply_pe_gradtome_partial_patch
        return AlgoSpec("gradtome_partial", "pe", apply_pe_gradtome_partial_patch,
            kwargs_from_args=_kw_pe_partial_basic, category="partial",
            description="Same as tome_partial but with grad-bipartite matching.")
    def mk_sparsesam_partial():
        from .sparsesam.pe_partial import apply_pe_sparsesam_partial_patch
        return AlgoSpec("sparsesam_partial", "pe", apply_pe_sparsesam_partial_patch,
            kwargs_from_args=_kw_pe_partial_sparsesam, category="partial",
            description="Block-sparse cute-kernel attention (perm + keep-bar mask).")

    for name, mk in (("tome", mk_tome), ("gradtome", mk_gradtome),
                     ("sparsesam", mk_sparsesam), ("flash_rope", mk_flash_rope),
                     ("sparge", mk_sparge), ("tome_partial", mk_tome_partial),
                     ("gradtome_partial", mk_gradtome_partial),
                     ("sparsesam_partial", mk_sparsesam_partial)):
        _safe_register("pe", name, mk)


def _register_sam():
    def _from_tome():
        from .tome.sam import (apply_patch as fn,
            ToMeSAMBlock as B, ToMeSAMAttention as A)
        return fn, B, A
    def _from_sparsesam():
        from .sparsesam.sam import (apply_patch as fn,
            ToMeSAMBlock as B, ToMeSAMAttention as A)
        return fn, B, A
    def _from_sparsesam_random():
        from .sparsesam.sam_random import (apply_patch as fn,
            ToMeSAMBlockRandom as B, ToMeSAMAttentionRandom as A)
        return fn, B, A
    def _from_gradtome():
        from .gradtome.sam import (apply_patch as fn,
            ToMeSAMBlock as B, ToMeSAMAttention as A)
        return fn, B, A
    def _from_gradtome_hilbert():
        from .gradtome.sam_hilbert import (apply_patch as fn,
            ToMeSAMBlock as B, ToMeSAMAttention as A)
        return fn, B, A
    def _from_sparge():
        from .sparge.sam import apply_patch as fn, SpargeSAMAttention as A
        return fn, None, A
    def _from_piecewise():
        from .piecewise.sam import apply_patch as fn, PiecewiseSAMAttention as A
        return fn, None, A

    # (registered_name, importer, internal_algo, description)
    table = [
        ("tome",             _from_tome,             "tome",
            "Bipartite token merge per block on the SAM-HQ encoder."),
        ("pitome",           _from_tome,             "pitome",
            "PiToMe (energy-margin) variant of ToMe."),
        ("sparsesam",        _from_sparsesam,        "tome",
            "SparseSAM Z-group merge on the SAM-HQ encoder."),
        ("sparsesam_pitome", _from_sparsesam,        "pitome",
            "SparseSAM with PiToMe energy-margin matching."),
        ("sparsesam_random", _from_sparsesam_random, "sparsesam_random",
            "SparseSAM with random keep-set selection (sanity baseline)."),
        ("gradtome",         _from_gradtome,         "tome",
            "Gradient-aware bipartite matching on the spatial grid."),
        ("gradtome_pitome",  _from_gradtome,         "pitome",
            "GradToMe with PiToMe energy-margin matching."),
        ("gradtome_hilbert", _from_gradtome_hilbert, "tome",
            "GradToMe with Hilbert-curve token ordering."),
        ("sparge",           _from_sparge,           "sparge",
            "SpargeAttn top-k sparse attention. Drop-in SDPA swap; "
            "decomposed rel-pos bias preserved on Ampere."),
        ("piecewise",        _from_piecewise,        "piecewise",
            "Piecewise Sparse Attention (PISA) drop-in attention swap; "
            "decomposed rel-pos bias is preserved."),
    ]
    for name, importer, internal_algo, desc in table:
        def mk(_imp=importer, _ia=internal_algo, _n=name, _d=desc):
            fn, B, A = _imp()
            return AlgoSpec(_n, "sam", _bake_sam_apply(fn, _ia),
                kwargs_from_args=_kw_sam_basic,
                block_class=B, attn_class=A, description=_d)
        _safe_register("sam", name, mk)


def _register_builtins():
    _register_pe()
    _register_sam()


_register_builtins()


# Back-compat per-backbone registry views (live snapshots taken at import time).
PE_REGISTRY  = specs_for("pe")
SAM_REGISTRY = specs_for("sam")


__all__ = [
    "AlgoSpec", "REGISTRY", "REGISTRATION_ERRORS",
    "register", "specs_for", "spec_of", "choices",
    "apply_pe", "apply_sam",
    "remove_all_pe", "remove_all_sam",
    "update_sam_ratio",
    "algo_choices", "sam_algo_choices",
    "PE_REGISTRY", "SAM_REGISTRY",
]
