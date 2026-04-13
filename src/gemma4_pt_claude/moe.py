"""Mixture-of-Experts: router + sparse expert dispatch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, TanhGELU
from .module_utils import InitModule, factory_kwargs


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
            init_std: float,
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
            init_std: float,
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
        """
        Args:
            x: ``[B, L, D]``
            weights: ``[B, L, K]`` — normalised routing weights
            expert_indices: ``[B, L, K]`` — expert indices (int64)

        Returns:
            ``[B, L, D]``
        """
        B, L, D = x.shape
        K = weights.shape[-1]

        # Flatten to [B*L, D]
        x_flat = x.reshape(-1, D)
        wi_flat = expert_indices.reshape(-1, K)  # [B*L, K]
        ww_flat = weights.reshape(-1, K)          # [B*L, K]

        # Accumulate expert outputs
        out = torch.zeros_like(x_flat)  # [B*L, D]

        for k_idx in range(K):
            expert_ids = wi_flat[:, k_idx]         # [B*L]
            w = ww_flat[:, k_idx].unsqueeze(-1)    # [B*L, 1]

            # Gather per-token expert weights
            gu = self.gate_up[expert_ids]   # [B*L, D, 2*H]
            dw = self.down[expert_ids]       # [B*L, H, D]

            # gate_up projection
            h = torch.bmm(x_flat.unsqueeze(1), gu).squeeze(1)  # [B*L, 2*H]
            gate, up = h.chunk(2, dim=-1)
            act = self.act(gate) * up  # [B*L, H]

            # down projection
            y = torch.bmm(act.unsqueeze(1), dw).squeeze(1)  # [B*L, D]

            # Per-expert scale
            es = self.per_expert_scale[expert_ids].unsqueeze(-1)  # [B*L, 1]
            out = out + w * es * y

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
            init_std: float,
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
