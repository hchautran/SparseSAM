"""Shared SigLIP / SigLIP 2 patching primitives.

SigLIP (and SigLIP 2 standard fixed-resolution checkpoints) shares one
HF class hierarchy:
  * `SiglipVisionModel` → `SiglipEncoder` → list of `SiglipEncoderLayer`
  * `SiglipEncoderLayer`: `layer_norm1 / self_attn / layer_norm2 / mlp`
    (no LayerScale, no DropPath — plain pre-norm).
  * `SiglipAttention`: separate `q_proj / k_proj / v_proj / out_proj`.
  * **No CLS, no register tokens, no RoPE.** Position info is added once
    at the input via `embeddings.position_embedding`.

Pooling is via `SiglipMultiheadAttentionPoolingHead` (reads the full
encoder output sequence), so token-count-preserving (`_partial`) patches
are the natural fit; full stage-compression would break the head.
"""

from __future__ import annotations
from typing import Optional

import torch.nn as nn

from transformers.models.siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipModel,
    SiglipVisionModel,
)


def _find_siglip_vision_encoder(model: nn.Module) -> Optional[SiglipEncoder]:
    """Return the inner `SiglipEncoder` (the module holding `.layers`)."""
    for m in model.modules():
        if isinstance(m, SiglipVisionModel):
            for sub in m.modules():
                if isinstance(sub, SiglipEncoder):
                    return sub
    return None


__all__ = [
    "SiglipAttention",
    "SiglipEncoder",
    "SiglipEncoderLayer",
    "SiglipModel",
    "SiglipVisionModel",
    "_find_siglip_vision_encoder",
]
