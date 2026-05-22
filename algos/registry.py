"""Unified algorithm registry for the SparseSAM + baselines workflow.

One `AlgoSpec` dataclass, one `REGISTRY` keyed by `(backbone, name)`, and a
single `register()` entry point. Per-backbone helpers (`apply_pe`,
`apply_sam`, `apply_siglip`, `apply_mvit` + matching `remove_all_*`) are
thin wrappers that exist because each backbone has a different stock-class
set that `remove_all_*` walks; the *apply* side is fully generic.

Adding a new algorithm: write the patch, then add one `register(AlgoSpec(...))`
call inside `_register_builtins()` below. See `docs/ADDING_ALGORITHMS.md`.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Spec + registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlgoSpec:
    name: str
    backbone: str                              # "pe" | "sam" | "siglip" | "mvit"
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


# ─────────────────────────────────────────────────────────────────────────────
# Generic apply / per-backbone wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _apply(model, backbone: str, name: str, args=None,
           ratio: Optional[float] = None, **extra):
    if name == "none":
        return None
    # SAM treats ratio>=1.0 as a no-op (full token count = nothing to merge).
    if backbone == "sam" and ratio is not None and ratio >= 1.0:
        return None

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

def apply_siglip(model, name, args=None, ratio=None):
    return _apply(model, "siglip", name, args, ratio)

def apply_mvit(model, name, args=None, ratio=None):
    return _apply(model, "mvit", name, args, ratio)


def algo_choices()        -> List[str]: return choices("pe")
def sam_algo_choices()    -> List[str]: return choices("sam")
def siglip_algo_choices() -> List[str]: return choices("siglip")
def mvit_algo_choices()   -> List[str]: return choices("mvit")


# ─────────────────────────────────────────────────────────────────────────────
# Per-backbone remove_all_*: revert subclassed modules to their stock classes
# and clear patch state. Each backbone has a different stock-class set, so
# they can't share a single implementation cleanly.
# ─────────────────────────────────────────────────────────────────────────────

def _revert_subclasses(model, stock_classes: Tuple[type, ...]) -> int:
    """For each module that is a strict subclass of any stock class, reassign
    `module.__class__` back to that stock class. Returns revert count."""
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
    """Idempotently undo every PE patch. Reverts PE subclasses to
    `SelfAttention` / `ResidualAttentionBlock`, removes registered hooks,
    and clears per-module caches. Safe to call before each sweep config."""
    try:
        from core.vision_encoder.pe import SelfAttention, ResidualAttentionBlock
    except Exception:
        return 0  # PE not importable -- nothing to revert.

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


def remove_all_siglip(model) -> int:
    """Idempotently revert subclassed SiglipAttention / SiglipEncoderLayer
    modules and clear patch state."""
    try:
        from transformers.models.siglip.modeling_siglip import (
            SiglipAttention, SiglipEncoderLayer,
        )
    except Exception:
        return 0

    n = _revert_subclasses(model, (SiglipAttention, SiglipEncoderLayer))
    for module in model.modules():
        _safe_delattr(module, "_tome_info")
    _safe_delattr(model, "_tome_info")
    return n


def remove_all_mvit(model) -> int:
    from .sparsesam.mvit import remove_mvit_sparsesam_partial_patch
    return remove_mvit_sparsesam_partial_patch(model)


# ─────────────────────────────────────────────────────────────────────────────
# kwargs_from_args helpers — translate argparse Namespace + sweep `ratio`
# into the kwarg dict each patch's `apply_*` function expects.
# ─────────────────────────────────────────────────────────────────────────────

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


def _kw_siglip_sparge(args, ratio):
    return dict(
        ratio=float(ratio if ratio is not None
                    else (args.ratio[0] if hasattr(args, "ratio") else 1.0)),
        start_block=int(getattr(args, "partial_start_block", 0)),
    )


def _kw_sam_basic(args, ratio):
    """SAM patches share an (algo, ratio, margin, mlp_merge) shape; the `algo`
    string is baked in at registration time by `_bake_sam_apply`."""
    return dict(
        ratio=float(ratio if ratio is not None else args.ratios[0]),
        margin=float(getattr(args, "margin", 0.5)),
        mlp_merge=bool(getattr(args, "mlp_merge", True)),
    )


def _kw_mvit_partial_sparsesam(args, ratio):
    return dict(
        ratio=float(ratio if ratio is not None else args.ratio[0]),
        group_size=int(getattr(args, "group_size", 4)),
        start_block=int(getattr(args, "partial_start_block", 0)),
        mlp_merge=bool(getattr(args, "mlp_merge", True)),
    )


def _bake_sam_apply(apply_fn, internal_algo: str):
    """Wrap a SAM patch's `apply_patch(encoder, algo=, ratio=, margin=, ...)`
    with `algo` baked in. Optional kwargs (e.g. `mlp_merge`) are forwarded
    only to patches whose signature accepts them — legacy patches without
    the option keep working unchanged."""
    accepted = set(inspect.signature(apply_fn).parameters)

    def _apply_baked(encoder, ratio, margin=0.5, **extra):
        kw = dict(algo=internal_algo, ratio=ratio, margin=margin)
        for k, v in extra.items():
            if k in accepted:
                kw[k] = v
        return apply_fn(encoder, **kw)
    return _apply_baked


# ─────────────────────────────────────────────────────────────────────────────
# Built-in algorithm registration.
# ─────────────────────────────────────────────────────────────────────────────

# Per-algo registration errors (missing dep, etc.). Keyed by f"{backbone}:{name}".
# Inspect after import to see what got skipped on a given host.
REGISTRATION_ERRORS: Dict[str, Exception] = {}


def _safe_register(backbone, name, import_and_make_spec):
    """Import the patch module and build+register an AlgoSpec. Tolerates
    per-algo ImportError so other algos in the same backbone keep working
    when an optional dep is missing on this host."""
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
    # Each (module-importer, registration-emitter) pair is wrapped in its own
    # try/except so one missing dep doesn't tank the whole backbone.
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
    ]
    for name, importer, internal_algo, desc in table:
        def mk(_imp=importer, _ia=internal_algo, _n=name, _d=desc):
            fn, B, A = _imp()
            return AlgoSpec(_n, "sam", _bake_sam_apply(fn, _ia),
                kwargs_from_args=_kw_sam_basic,
                block_class=B, attn_class=A, description=_d)
        _safe_register("sam", name, mk)


def _register_siglip():
    def mk_sparsesam_partial():
        from .sparsesam.siglip import apply_siglip_sparsesam_partial_patch
        return AlgoSpec("sparsesam_partial", "siglip",
            apply_siglip_sparsesam_partial_patch,
            kwargs_from_args=_kw_pe_partial_sparsesam, category="partial",
            description="Cute block-sparse FA2 (no RoPE) + uniform-stride z-order perm.")
    def mk_tome_partial():
        from .tome.siglip import apply_siglip_tome_partial_patch
        return AlgoSpec("tome_partial", "siglip",
            apply_siglip_tome_partial_patch,
            kwargs_from_args=_kw_pe_partial_basic, category="partial",
            description="Full-Q + ToMe-merged-K/V SDPA + optional merge/MLP/unmerge.")
    def mk_gradtome_partial():
        from .gradtome.siglip import apply_siglip_gradtome_partial_patch
        return AlgoSpec("gradtome_partial", "siglip",
            apply_siglip_gradtome_partial_patch,
            kwargs_from_args=_kw_pe_partial_basic, category="partial",
            description="Spatial-gradient bipartite matching on the patch grid.")
    def mk_sparge():
        from .sparge.siglip import apply_siglip_sparge_patch
        return AlgoSpec("sparge", "siglip",
            apply_siglip_sparge_patch,
            kwargs_from_args=_kw_siglip_sparge, category="attention",
            description="SpargeAttn top-k sparse attention swap. No RoPE step.")

    for name, mk in (("sparsesam_partial", mk_sparsesam_partial),
                     ("tome_partial",      mk_tome_partial),
                     ("gradtome_partial",  mk_gradtome_partial),
                     ("sparge",            mk_sparge)):
        _safe_register("siglip", name, mk)


def _register_mvit():
    def mk():
        from .sparsesam.mvit import apply_mvit_sparsesam_partial_patch
        return AlgoSpec("sparsesam_partial", "mvit",
            apply_mvit_sparsesam_partial_patch,
            kwargs_from_args=_kw_mvit_partial_sparsesam, category="partial",
            description="MLP-side broadcast merge → MLP → unmerge on post-attention "
                        "residual stream.")
    _safe_register("mvit", "sparsesam_partial", mk)


def _register_builtins():
    """Register each backbone. Per-algorithm failures (missing dep, etc.)
    are caught inside `_safe_register` and stashed in `REGISTRATION_ERRORS`
    so the rest of the registry keeps working."""
    _register_pe()
    _register_sam()
    _register_siglip()
    _register_mvit()


_register_builtins()


# ─────────────────────────────────────────────────────────────────────────────
# Back-compat aliases (kept so callers that import per-backbone registries by
# name keep working). These are live views computed once at import time.
# ─────────────────────────────────────────────────────────────────────────────

PE_REGISTRY     = specs_for("pe")
SAM_REGISTRY    = specs_for("sam")
SIGLIP_REGISTRY = specs_for("siglip")
MVIT_REGISTRY   = specs_for("mvit")


__all__ = [
    "AlgoSpec", "REGISTRY", "REGISTRATION_ERRORS",
    "register", "specs_for", "spec_of", "choices",
    "apply_pe", "apply_sam", "apply_siglip", "apply_mvit",
    "remove_all_pe", "remove_all_sam", "remove_all_siglip", "remove_all_mvit",
    "update_sam_ratio",
    "algo_choices", "sam_algo_choices", "siglip_algo_choices", "mvit_algo_choices",
    "PE_REGISTRY", "SAM_REGISTRY", "SIGLIP_REGISTRY", "MVIT_REGISTRY",
]
