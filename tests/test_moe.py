"""Unit tests for MoE components."""

import torch
import pytest

from gemma4_pt_claude.layers import TanhGELU
from gemma4_pt_claude.moe import MoERouter, MoEExperts, MoELayer


class TestMoERouter:
    def test_output_shapes(self):
        router = MoERouter(features=64, num_experts=8, top_k=2)
        x = torch.randn(1, 4, 64)
        weights, indices = router(x)
        assert weights.shape == (1, 4, 2)
        assert indices.shape == (1, 4, 2)

    def test_weights_sum_to_one(self):
        router = MoERouter(features=32, num_experts=8, top_k=4)
        x = torch.randn(1, 4, 32)
        weights, _ = router(x)
        sums = weights.float().sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_indices_in_range(self):
        router = MoERouter(features=32, num_experts=16, top_k=4)
        x = torch.randn(2, 8, 32)
        _, indices = router(x)
        assert (indices >= 0).all()
        assert (indices < 16).all()


class TestMoEExperts:
    def test_output_shape(self):
        experts = MoEExperts(num_experts=8, features=32, expert_dim=16)
        x = torch.randn(1, 4, 32)
        weights = torch.ones(1, 4, 2) * 0.5
        indices = torch.randint(0, 8, (1, 4, 2))
        out = experts(x, weights, indices)
        assert out.shape == (1, 4, 32)

    def test_uses_tanh_gelu_module(self):
        experts = MoEExperts(num_experts=8, features=32, expert_dim=16)
        assert isinstance(experts.act, TanhGELU)


class TestMoEDispatchEquivalence:
    """The grouped dispatch must match a naive per-token reference."""

    @staticmethod
    def _reference(experts: MoEExperts, x, weights, indices):
        B, L, D = x.shape
        K = weights.shape[-1]
        x_flat = x.reshape(-1, D)
        out = torch.zeros_like(x_flat)
        for k_idx in range(K):
            ids = indices.reshape(-1, K)[:, k_idx]
            w = weights.reshape(-1, K)[:, k_idx].unsqueeze(-1)
            gate, up = torch.bmm(
                x_flat.unsqueeze(1), experts.gate_up[ids]
            ).squeeze(1).chunk(2, dim=-1)
            act = experts.act(gate) * up
            y = torch.bmm(act.unsqueeze(1), experts.down[ids]).squeeze(1)
            out = out + w * experts.per_expert_scale[ids].unsqueeze(-1) * y
        return out.reshape(B, L, D)

    @pytest.mark.parametrize("shape", [(1, 1, 32), (2, 5, 32), (1, 16, 32)])
    def test_matches_naive_reference(self, shape):
        torch.manual_seed(0)
        experts = MoEExperts(num_experts=8, features=32, expert_dim=16)
        experts.init_weights()
        B, L, D = shape
        x = torch.randn(B, L, D)
        weights = torch.rand(B, L, 3)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        indices = torch.randint(0, 8, (B, L, 3))
        assert torch.allclose(
            experts(x, weights, indices),
            self._reference(experts, x, weights, indices),
            atol=1e-5,
        )

    def test_unused_experts_are_skipped(self):
        """Routing everything to one expert must still be correct."""
        torch.manual_seed(0)
        experts = MoEExperts(num_experts=8, features=32, expert_dim=16)
        experts.init_weights()
        x = torch.randn(1, 4, 32)
        weights = torch.full((1, 4, 2), 0.5)
        indices = torch.full((1, 4, 2), 3)
        assert torch.allclose(
            experts(x, weights, indices),
            self._reference(experts, x, weights, indices),
            atol=1e-5,
        )

    def test_gradients_flow_to_selected_experts(self):
        torch.manual_seed(0)
        experts = MoEExperts(num_experts=4, features=16, expert_dim=8)
        experts.init_weights()
        x = torch.randn(1, 3, 16)
        weights = torch.full((1, 3, 1), 1.0)
        indices = torch.tensor([[[0], [2], [0]]])
        experts(x, weights, indices).square().mean().backward()
        grad = experts.gate_up.grad
        assert grad[0].abs().max() > 0 and grad[2].abs().max() > 0
        assert grad[1].abs().max() == 0 and grad[3].abs().max() == 0


class TestMoELayer:
    def test_end_to_end(self):
        layer = MoELayer(features=64, num_experts=8, top_k=2, expert_dim=32)
        x = torch.randn(1, 4, 64)
        out = layer(x)
        assert out.shape == (1, 4, 64)
