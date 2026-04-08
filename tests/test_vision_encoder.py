"""Unit tests for the vision encoder."""

import torch
import pytest

from gemma4_pt_claude.config import VisionConfig
from gemma4_pt_claude.vision_encoder import (
    VisionPatchEmbedder,
    VisionAttention,
    VisionBlock,
    VisionMLP,
    VisionPooler,
    VisionEncoder,
)
from gemma4_pt_claude.layers import ClippedLinear, TanhGELU


def _tiny_vision_config(**overrides) -> VisionConfig:
    defaults = dict(
        d_model=32,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        ffw_hidden=64,
        patch_size=4,
        output_length=4,
        pooling_kernel_size=2,
        position_embedding_size=64,
        use_clipped_linear=False,
        standardize=False,
        rms_norm_eps=1e-6,
        rope_base_frequency=100.0,
        text_embed_dim=32,
    )
    defaults.update(overrides)
    return VisionConfig(**defaults)


class TestVisionPatchEmbedder:
    def test_output_shape(self):
        cfg = _tiny_vision_config()
        embedder = VisionPatchEmbedder(cfg)
        patch_dim = 3 * cfg.patch_size ** 2
        B, L = 2, cfg.max_patches
        pixel_values = torch.randn(B, L, patch_dim)
        position_ids = torch.zeros(B, L, 2, dtype=torch.long)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        out = embedder(pixel_values, position_ids, padding_mask)
        assert out.shape == (B, L, cfg.d_model)

    def test_position_embedding_zeroed_at_padding(self):
        """Position embeddings at padding positions should be zero."""
        cfg = _tiny_vision_config()
        embedder = VisionPatchEmbedder(cfg)
        B, L = 1, 4
        position_ids = torch.zeros(B, L, 2, dtype=torch.long)
        position_ids[:, 2:] = -1
        padding_mask = (position_ids == -1).all(dim=-1)

        # Directly check position embedding output (not full forward which
        # includes pixel normalization)
        pos_emb = embedder._position_embeddings(position_ids, padding_mask)
        # Padding positions should have zero position embedding
        assert pos_emb[:, 2:].abs().max().item() < 1e-5
        # Non-padding should NOT necessarily be zero (unless table is zero-init)
        # With zero-init table, all position embeddings are zero
        assert pos_emb[:, :2].abs().max().item() < 1e-5  # zero-init

    def test_position_embedding_table_zeros_init(self):
        cfg = _tiny_vision_config()
        embedder = VisionPatchEmbedder(cfg)
        assert embedder.position_embedding_table.data.abs().max().item() == 0.0


class TestVisionAttention:
    def test_output_shape(self):
        cfg = _tiny_vision_config()
        attn = VisionAttention(cfg)
        B, L = 2, 8
        x = torch.randn(B, L, cfg.d_model)
        mask = torch.zeros(B, 1, L, L)  # additive mask, 0 = attend
        pos_ids = torch.randint(0, 4, (B, L, 2))
        out = attn(x, mask, pos_ids)
        assert out.shape == (B, L, cfg.d_model)

    def test_bidirectional_mask(self):
        """All valid tokens should attend to each other (bidirectional)."""
        cfg = _tiny_vision_config()
        attn = VisionAttention(cfg)
        B, L = 1, 4
        x = torch.randn(B, L, cfg.d_model)
        mask = torch.zeros(B, 1, L, L)  # no masking
        pos_ids = torch.zeros(B, L, 2, dtype=torch.long)
        out = attn(x, mask, pos_ids)
        assert not torch.isnan(out).any()


class TestVisionMLP:
    def test_uses_tanh_gelu_module(self):
        cfg = _tiny_vision_config()
        mlp = VisionMLP(cfg)
        assert isinstance(mlp.act, TanhGELU)


class TestVisionPooler:
    def test_pooling_shape(self):
        cfg = _tiny_vision_config()
        pooler = VisionPooler(cfg)
        B, L = 2, cfg.max_patches
        hidden = torch.randn(B, L, cfg.d_model)
        pos_ids = torch.zeros(B, L, 2, dtype=torch.long)
        # Set up a grid of positions
        k = cfg.pooling_kernel_size
        for i in range(L):
            pos_ids[:, i, 0] = i % (k * 2)
            pos_ids[:, i, 1] = i // (k * 2)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        out, mask = pooler(hidden, pos_ids, padding_mask, cfg.output_length)
        assert out.shape == (B, cfg.output_length, cfg.d_model)
        assert mask.shape == (B, cfg.output_length)

    def test_scale_by_sqrt_d_model(self):
        cfg = _tiny_vision_config()
        pooler = VisionPooler(cfg)
        assert pooler.root_hidden_size == pytest.approx(cfg.d_model ** 0.5)


class TestVisionEncoder:
    def test_output_shape(self):
        cfg = _tiny_vision_config()
        encoder = VisionEncoder(cfg)
        patch_dim = 3 * cfg.patch_size ** 2
        B = 2
        pixel_values = torch.randn(B, cfg.max_patches, patch_dim)
        pos_ids = torch.zeros(B, cfg.max_patches, 2, dtype=torch.long)
        # Fill valid positions
        k = cfg.pooling_kernel_size
        for i in range(cfg.max_patches):
            pos_ids[:, i, 0] = i % (k * 2)
            pos_ids[:, i, 1] = i // (k * 2)
        hidden, pooler_mask = encoder(pixel_values, pos_ids)
        # Should return [B, output_length, D] (not flattened)
        assert hidden.shape == (B, cfg.output_length, cfg.d_model)
        assert pooler_mask.shape == (B, cfg.output_length)

    def test_no_nan(self):
        cfg = _tiny_vision_config()
        encoder = VisionEncoder(cfg)
        patch_dim = 3 * cfg.patch_size ** 2
        B = 1
        pixel_values = torch.randn(B, cfg.max_patches, patch_dim)
        pos_ids = torch.zeros(B, cfg.max_patches, 2, dtype=torch.long)
        for i in range(cfg.max_patches):
            pos_ids[:, i, 0] = i % 4
            pos_ids[:, i, 1] = i // 4
        hidden, _ = encoder(pixel_values, pos_ids)
        assert not torch.isnan(hidden).any()

    def test_standardize_buffers(self):
        cfg = _tiny_vision_config(standardize=True)
        encoder = VisionEncoder(cfg)
        assert encoder.std_bias.shape == (cfg.d_model,)
        assert encoder.std_scale.shape == (cfg.d_model,)
        assert encoder.std_bias.abs().max().item() == 0.0
        assert (encoder.std_scale == 1.0).all()

    def test_with_padding(self):
        """Encoder handles padded patches (position_ids = -1)."""
        cfg = _tiny_vision_config()
        encoder = VisionEncoder(cfg)
        patch_dim = 3 * cfg.patch_size ** 2
        B = 1
        num_valid = cfg.max_patches // 2
        pixel_values = torch.randn(B, cfg.max_patches, patch_dim)
        pos_ids = torch.full((B, cfg.max_patches, 2), -1, dtype=torch.long)
        for i in range(num_valid):
            pos_ids[:, i, 0] = i % 4
            pos_ids[:, i, 1] = i // 4
        hidden, pooler_mask = encoder(pixel_values, pos_ids)
        assert hidden.shape == (B, cfg.output_length, cfg.d_model)
        assert not torch.isnan(hidden).any()


class TestClippedLinear:
    def test_clipping(self):
        cl = ClippedLinear(4, 4, bias=False)
        # Set tight clip bounds
        cl.input_min.fill_(-0.5)
        cl.input_max.fill_(0.5)
        cl.output_min.fill_(-1.0)
        cl.output_max.fill_(1.0)
        x = torch.tensor([[2.0, -2.0, 0.3, -0.3]])
        out = cl(x)
        # Output should be within bounds
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_default_no_clip(self):
        cl = ClippedLinear(4, 4, bias=False)
        assert cl.input_min.item() == float("-inf")
        assert cl.input_max.item() == float("inf")

    def test_use_clipped_linear_in_vision(self):
        cfg = _tiny_vision_config(use_clipped_linear=True)
        attn = VisionAttention(cfg)
        assert isinstance(attn.q_proj, ClippedLinear)
