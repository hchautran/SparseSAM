# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    ratio:float=1.0,    
    class_token: bool = False,
) -> Tuple[Callable, Callable]:
    
    
    protected = 0
    if class_token:
        protected += 1
    if len(metric.shape) == 2:
        metric = metric[None,...]

    # We can only reduce by a maximum of 50% tokens
    T = metric.shape[1]

    if ratio < 1.0:
        r = math.floor(T- T*ratio)
        # Bipartite ToMe splits metric into a (even idx) of size t1=ceil(T/2)
        # and b (odd idx) of size T-t1. unm_idx is sliced from edge_idx of
        # size t1, so r > t1 makes merge() expand to (n, t1-r<0, c) → crash.
        # For ratio < 0.5 the requested reduction exceeds the bipartite
        # cap; clamp to t1 (effective compression caps at ~0.5).
        t1 = (T + 1) // 2 - protected
        r = min(r, t1)
    else:
        return do_nothing, do_nothing


    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        indices = torch.arange(T).to(metric.device)
        a_idx = indices[::2].unsqueeze(0).unsqueeze(-1)  # (1, N/2, 1)
        b_idx = indices[1::2].unsqueeze(0).unsqueeze(-1)  # (1, N/2, 1)

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if len(x.shape) == 2:
            x.unsqueeze_(0)
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        if mode is not None:
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        unm_absolute_indices = torch.gather(
            a_idx.expand(n, a.shape[1], 1), dim=1,
            index=unm_idx.expand(n, -1, 1),
        ).squeeze(-1)
        # (B*num_heads, N_dst)
        dst_absolute_indices = b_idx.squeeze(-1).expand(n, -1)
        absolute_indices = torch.cat([unm_absolute_indices, dst_absolute_indices], dim=1)

        return torch.cat([unm, dst], dim=1), absolute_indices
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


