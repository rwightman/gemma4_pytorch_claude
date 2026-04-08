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


class TestMoELayer:
    def test_end_to_end(self):
        layer = MoELayer(features=64, num_experts=8, top_k=2, expert_dim=32)
        x = torch.randn(1, 4, 64)
        out = layer(x)
        assert out.shape == (1, 4, 64)
