"""Mixture-of-Experts: router + sparse expert dispatch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, TanhGELU
from .module_utils import DEFAULT_INIT_STD, InitModule, factory_kwargs


class MoERouter(InitModule):
    """Top-k softmax router.

    JAX reference: norm(x) * (rsqrt(features) * router_scale) -> linear -> softmax -> topk -> renorm.
    ``router_scale`` is a *per-feature* learned parameter (init ones).
    """

    def __init__(
            self,
            features: int,
            num_experts: int,
            top_k: int,
            init_std: float = DEFAULT_INIT_STD,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.features = features
        self.top_k = top_k
        self.num_experts = num_experts
        self.init_std = init_std
        dd = factory_kwargs(device, dtype)
        self.norm = RMSNorm(features, with_scale=False, **dd)
        self.gate = nn.Linear(features, num_experts, bias=False, **dd)
        # Per-feature learned router scale (init ones), times rsqrt(features)
        self.router_scale = nn.Parameter(torch.ones(features, **dd))
        self.root_size = self._build_root_size()

    def _build_root_size(self) -> float:
        return self.features ** -0.5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (weights, expert_indices) both ``[B, L, top_k]``."""
        h = self.norm(x) * self.root_size * self.router_scale
        logits = self.gate(h).float()                    # [B, L, E]
        probs = F.softmax(logits, dim=-1)                # [B, L, E]
        topk_weights, topk_idx = probs.topk(self.top_k, dim=-1)
        # Renormalise: divide by sum of selected expert probabilities
        denom = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        topk_weights = topk_weights / denom
        return topk_weights.to(x.dtype), topk_idx

    def _init_weights(self, ctx) -> None:
        nn.init.normal_(self.gate.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)
        nn.init.ones_(self.router_scale)


class MoEExperts(InitModule):
    """Batched expert GatedMLPs.

    Stores gate_up and down weights as ``[E, ...]`` tensors and dispatches
    tokens to the selected experts.  Includes per-expert learned scale.
    """

    def __init__(
            self,
            num_experts: int,
            features: int,
            expert_dim: int,
            init_std: float = DEFAULT_INIT_STD,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.features = features
        self.expert_dim = expert_dim
        self.init_std = init_std
        self.residual_init_std = init_std if residual_init_std is None else residual_init_std
        # Each expert is a GatedMLP: gate_up [features -> 2*expert_dim], down [expert_dim -> features]
        dd = factory_kwargs(device, dtype)
        self.gate_up = nn.Parameter(torch.empty(num_experts, features, 2 * expert_dim, **dd))
        self.down = nn.Parameter(torch.empty(num_experts, expert_dim, features, **dd))
        # Per-expert scale (init ones)
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts, **dd))
        self.act = TanhGELU(**dd)

    def _init_weights(self, ctx):
        nn.init.normal_(self.gate_up, mean=0.0, std=self.init_std, generator=ctx.generator)
        nn.init.normal_(self.down, mean=0.0, std=self.residual_init_std, generator=ctx.generator)
        nn.init.ones_(self.per_expert_scale)

    def forward(
            self,
            x: torch.Tensor,
            weights: torch.Tensor,
            expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch each token to its top-k experts.

        Tokens are grouped by expert so each expert runs one dense GEMM over its
        assigned rows.  Gathering the *weights* per token instead (``gate_up[ids]``)
        would materialise a ``[tokens, D, 2*expert_dim]`` tensor — tens of GB at
        26B-A4B dimensions for a normal prompt.

        Args:
            x: ``[B, L, D]``
            weights: ``[B, L, K]`` — normalised routing weights
            expert_indices: ``[B, L, K]`` — expert indices (int64)

        Returns:
            ``[B, L, D]``
        """
        B, L, D = x.shape
        K = weights.shape[-1]

        x_flat = x.reshape(-1, D)                            # [T, D]
        flat_idx = expert_indices.reshape(-1)                # [T*K]
        num_slots = flat_idx.shape[0]

        # Sort assignments by expert; `order` maps sorted slot -> original slot.
        order = torch.argsort(flat_idx, stable=True)
        token_of_slot = torch.div(order, K, rounding_mode="floor")  # [T*K]

        y_sorted = x_flat.new_empty(num_slots, D)
        scale_sorted = x_flat.new_empty(num_slots)

        experts, counts = torch.unique_consecutive(flat_idx[order], return_counts=True)
        start = 0
        for expert_id, count in zip(experts.tolist(), counts.tolist()):
            span = slice(start, start + count)
            start += count

            rows = x_flat[token_of_slot[span]]                       # [count, D]
            gate, up = (rows @ self.gate_up[expert_id]).chunk(2, dim=-1)
            y_sorted[span] = (self.act(gate) * up) @ self.down[expert_id]
            scale_sorted[span] = self.per_expert_scale[expert_id]

        # Undo the sort, then combine the K expert outputs for each token.
        y = torch.empty_like(y_sorted)
        y[order] = y_sorted
        scale = torch.empty_like(scale_sorted)
        scale[order] = scale_sorted

        combine = weights.reshape(-1, K) * scale.view(-1, K)         # [T, K]
        out = (y.view(-1, K, D) * combine.unsqueeze(-1)).sum(dim=1)
        return out.reshape(B, L, D)


class MoELayer(InitModule):
    """MoE layer: router -> expert dispatch (no dense branch).

    The dense branch lives in ``TransformerBlock`` alongside its own norms,
    matching the JAX reference structure.
    """

    def __init__(
            self,
            features: int,
            num_experts: int,
            top_k: int,
            expert_dim: int,
            init_std: float = DEFAULT_INIT_STD,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.router = MoERouter(features, num_experts, top_k, init_std=init_std, **dd)
        self.experts = MoEExperts(
            num_experts,
            features,
            expert_dim,
            init_std=init_std,
            residual_init_std=residual_init_std,
            **dd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices = self.router(x)
        return self.experts(x, weights, indices)
