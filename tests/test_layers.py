"""Unit tests for core layer primitives."""

import torch
import pytest

from gemma4_pt_claude.layers import (
    GatedMLP,
    RMSNorm,
    TanhGELU,
    apply_multidimensional_rope,
    apply_rope,
)


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_scale_plus_one_default(self):
        """scale_plus_one=True: param init 0 → effective scale 1 → output ≈ normed input."""
        norm = RMSNorm(32, scale_plus_one=True)
        x = torch.randn(1, 4, 32)
        out = norm(x)
        # With scale=0, effective scale=1, so should be close to plain RMS norm
        rms = x.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        expected = (x.float() * rms).to(x.dtype)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_scale_plus_one_false(self):
        """scale_plus_one=False: param init 1 → effective scale 1."""
        norm = RMSNorm(32, scale_plus_one=False)
        x = torch.randn(1, 4, 32)
        out = norm(x)
        rms = x.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        expected = (x.float() * rms).to(x.dtype)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_no_scale(self):
        norm = RMSNorm(16, with_scale=False)
        assert norm.scale is None
        x = torch.randn(2, 4, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_zero_init(self):
        norm = RMSNorm(16, zero_init=True)
        # zero_init means scale param is all zeros
        assert (norm.scale == 0).all()

    def test_dtype_preservation(self):
        norm = RMSNorm(32, scale_plus_one=False).to(torch.bfloat16)
        x = torch.randn(1, 4, 32, dtype=torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16


class TestGatedMLP:
    def test_output_shape(self):
        mlp = GatedMLP(64, 256)
        x = torch.randn(2, 8, 64)
        out = mlp(x)
        assert out.shape == x.shape

    def test_gate_up_proj_size(self):
        mlp = GatedMLP(32, 128)
        assert mlp.gate_up_proj.weight.shape == (256, 32)  # 2*hidden, features
        assert mlp.down_proj.weight.shape == (32, 128)

    def test_uses_tanh_gelu_module(self):
        mlp = GatedMLP(32, 128)
        assert isinstance(mlp.act, TanhGELU)


class TestRoPE:
    def test_output_shape(self):
        x = torch.randn(2, 8, 4, 64)
        pos = torch.arange(8).unsqueeze(0).expand(2, -1)
        out = apply_rope(x, pos)
        assert out.shape == x.shape

    def test_zero_position_identity(self):
        """At position 0, sin=0 cos=1, so RoPE should be near identity."""
        x = torch.randn(1, 1, 1, 64)
        pos = torch.zeros(1, 1, dtype=torch.long)
        out = apply_rope(x, pos)
        assert torch.allclose(out, x, atol=1e-5)

    def test_rope_proportion(self):
        """With proportion < 1, only part of the dims rotate."""
        x = torch.randn(1, 4, 1, 64)
        pos = torch.arange(4).unsqueeze(0)
        out_full = apply_rope(x, pos, rope_proportion=1.0)
        out_half = apply_rope(x, pos, rope_proportion=0.5)
        # They should differ since different proportions rotate
        assert not torch.allclose(out_full, out_half)

    def test_scale_factor(self):
        x = torch.randn(1, 4, 1, 64)
        pos = torch.arange(4).unsqueeze(0)
        out_1 = apply_rope(x, pos, scale_factor=1.0)
        out_8 = apply_rope(x, pos, scale_factor=8.0)
        # Different scale factors should give different results
        assert not torch.allclose(out_1, out_8)

    def test_multidimensional(self):
        x = torch.randn(1, 16, 4, 64)
        pos_h = torch.arange(16).unsqueeze(0)
        pos_w = torch.arange(16).unsqueeze(0)
        out = apply_multidimensional_rope(x, pos_h, pos_w)
        assert out.shape == x.shape
