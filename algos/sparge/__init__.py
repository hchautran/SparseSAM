"""SpargeAttn — drop-in sparse attention kernel from thu-ml/SpargeAttn.

Plug-and-play replacement for `F.scaled_dot_product_attention`. Unlike
`tome` / `pitome` / `sparsesam` this algorithm performs *no* token merging:
it sparsifies the QK matrix at runtime via top-k attention-mass masking
inside the cute kernel. The `ratio` knob is forwarded as `topk` (lower
ratio → sparser attention → faster but less accurate).

The SAM patch uses the `_pos` kernel variant so SAM-HQ's decomposed
relative-position bias is folded into the attention matrix inside the
kernel — no rel-pos drop. The `pos` argument is honored on Ampere
(sm80/sm86/sm87) only; on Ada/Hopper the kernel silently ignores it and
the patch warns at apply time. PE uses no rel-pos (RoPE is applied to
q/k before the kernel) so the PE patch is a faithful SDPA swap.

Constraints:
- head_dim ∈ {64, 128}. SAM-HQ ViT-B/L (head_dim=64) ✓; ViT-H (80) ✗.
- sequence length ≥ 128.
"""

from .sam import SpargeSAMAttention, apply_patch  # noqa: F401
from .pe import SpargePEAttention, apply_pe_sparge_patch, remove_pe_sparge_patch  # noqa: F401

__all__ = [
    "SpargeSAMAttention", "apply_patch",
    "SpargePEAttention", "apply_pe_sparge_patch", "remove_pe_sparge_patch",
]
