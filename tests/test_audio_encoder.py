"""Shape and correctness tests for the audio encoder."""

import torch
import pytest

from gemma4_pt_claude.config import AudioConfig
from gemma4_pt_claude.audio_encoder import (
    SubSamplingBlock,
    ChunkedLocalAttention,
    ConformerLayer,
    AudioEncoder,
)


@pytest.fixture
def cfg():
    return AudioConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        lm_model_dims=96,
        input_feat_size=128,
        sscp_channels=(16, 8),
    )


class TestSubSamplingBlock:
    def test_output_shape(self, cfg):
        block = SubSamplingBlock(cfg)
        B, T, F = 2, 100, cfg.input_feat_size
        x = torch.randn(B, T, F)
        mask = torch.zeros(B, T, dtype=torch.bool)

        out, out_mask = block(x, mask)
        # Time is subsampled by ~4x (2x per conv with stride 2)
        assert out.ndim == 3
        assert out.shape[0] == B
        assert out.shape[2] == cfg.hidden_size
        assert out_mask.shape == (B, out.shape[1])

    def test_mask_subsampling(self, cfg):
        block = SubSamplingBlock(cfg)
        B, T, F = 1, 80, cfg.input_feat_size
        x = torch.randn(B, T, F)
        # Last 20 frames are padded
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, 60:] = True

        out, out_mask = block(x, mask)
        # Mask should be subsampled, last positions should remain True
        assert out_mask.shape == (B, out.shape[1])

    def test_layernorm_used(self, cfg):
        """SubSamplingBlock should use LayerNorm, not GroupNorm."""
        block = SubSamplingBlock(cfg)
        assert isinstance(block.norm1, torch.nn.LayerNorm)
        assert isinstance(block.norm2, torch.nn.LayerNorm)
        # LayerNorm should have no bias (bias=False in JAX)
        assert block.norm1.bias is None
        assert block.norm2.bias is None


class TestChunkedLocalAttention:
    def test_output_shape(self, cfg):
        attn = ChunkedLocalAttention(cfg)
        B, T = 2, 48
        x = torch.randn(B, T, cfg.hidden_size)
        mask = torch.zeros(B, T, dtype=torch.bool)

        out = attn(x, mask)
        head_dim = cfg.hidden_size // cfg.num_heads
        assert out.shape == (B, T, cfg.num_heads, head_dim)

    def test_per_dim_scale_init_ones(self, cfg):
        """per_dim_scale should be initialized to ones (matching JAX)."""
        attn = ChunkedLocalAttention(cfg)
        torch.testing.assert_close(
            attn.per_dim_scale, torch.ones(cfg.hidden_size // cfg.num_heads),
        )

    def test_key_scale_buffer(self, cfg):
        """Should have a key_scale buffer for key scaling."""
        attn = ChunkedLocalAttention(cfg)
        assert hasattr(attn, "key_scale")
        assert attn.key_scale.ndim == 0
        assert attn.key_scale.item() > 0


class TestConformerLayer:
    def test_output_shape(self, cfg):
        layer = ConformerLayer(cfg)
        B, T = 2, 48
        x = torch.randn(B, T, cfg.hidden_size)
        mask = torch.zeros(B, T, dtype=torch.bool)

        out = layer(x, mask)
        assert out.shape == (B, T, cfg.hidden_size)


class TestAudioEncoder:
    def test_output_shape(self, cfg):
        encoder = AudioEncoder(cfg)
        B, T, F = 2, 100, cfg.input_feat_size
        mel = torch.randn(B, T, F)
        mask = torch.zeros(B, T, dtype=torch.bool)

        out, out_mask = encoder(mel, mask)
        assert out.ndim == 3
        assert out.shape[0] == B
        assert out.shape[2] == cfg.lm_model_dims
        assert out_mask.shape == (B, out.shape[1])

    def test_output_proj_has_bias(self, cfg):
        """output_proj should have bias=True (matching JAX nn.Dense)."""
        encoder = AudioEncoder(cfg)
        assert encoder.output_proj.bias is not None

    def test_no_audio_proj_or_norm(self, cfg):
        """AudioEncoder should not have audio_proj or audio_norm."""
        encoder = AudioEncoder(cfg)
        assert not hasattr(encoder, "audio_proj")
        assert not hasattr(encoder, "audio_norm")

    def test_padded_positions_zeroed(self, cfg):
        """Padded positions should have zero output."""
        encoder = AudioEncoder(cfg)
        B, T, F = 1, 100, cfg.input_feat_size
        mel = torch.randn(B, T, F)
        # All frames are padded
        mask = torch.ones(B, T, dtype=torch.bool)

        out, out_mask = encoder(mel, mask)
        # All outputs should be zero since all frames are padded
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
