"""Unit tests for transformer blocks and text decoder."""

import torch
import pytest

from gemma4_pt_claude.config import AttentionType, TextConfig
from gemma4_pt_claude.layers import TanhGELU
from gemma4_pt_claude.transformer import Embedder, PerLayerMapping, TransformerBlock, TextDecoder


def _small_text_config(**overrides) -> TextConfig:
    defaults = dict(
        vocab_size=128,
        embed_dim=64,
        hidden_dim=128,
        num_heads=4,
        head_dim=16,
        num_kv_heads=2,
        num_layers=4,
        sliding_window_size=32,
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        use_qk_norm=True,
        use_value_norm=False,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        per_layer_input_dim=0,
    )
    defaults.update(overrides)
    return TextConfig(**defaults)


class TestEmbedder:
    def test_encode_shape(self):
        cfg = _small_text_config()
        emb = Embedder(cfg)
        tokens = torch.randint(0, 128, (2, 8))
        out = emb.encode(tokens)
        assert out.shape == (2, 8, 64)

    def test_encode_scaling(self):
        """Output should be scaled by sqrt(embed_dim)."""
        cfg = _small_text_config()
        emb = Embedder(cfg)
        tokens = torch.tensor([[0]])
        out = emb.encode(tokens)
        raw = emb.token_embedding(tokens)
        expected = raw * (64 ** 0.5)
        assert torch.allclose(out, expected)

    def test_decode_logits_shape(self):
        cfg = _small_text_config()
        emb = Embedder(cfg)
        x = torch.randn(2, 8, 64)
        logits = emb.decode_logits(x)
        assert logits.shape == (2, 8, 128)  # vocab_size

    def test_pli_embedding(self):
        cfg = _small_text_config(per_layer_input_dim=16)
        emb = Embedder(cfg)
        tokens = torch.randint(0, 128, (1, 4))
        x = emb.encode(tokens)
        pli = emb.encode_per_layer_input(x, tokens)
        assert pli.shape == (1, 4, 4, 16)  # [B, L, num_layers, pli_dim]


class TestPerLayerMapping:
    def test_output_shape(self):
        plm = PerLayerMapping(64, 16)
        x = torch.randn(2, 8, 64)
        pli = torch.randn(2, 8, 16)
        out = plm(x, pli)
        assert out.shape == (2, 8, 64)

    def test_uses_tanh_gelu_module(self):
        plm = PerLayerMapping(64, 16)
        assert isinstance(plm.act, TanhGELU)


class TestTransformerBlock:
    def test_dense_block(self):
        cfg = _small_text_config()
        block = TransformerBlock(cfg, layer_idx=0, attn_type=AttentionType.LOCAL_SLIDING)
        B, L = 1, 8
        x = torch.randn(B, L, 64)
        pos = torch.arange(L).unsqueeze(0)
        mask = torch.ones(B, L, L, dtype=torch.bool)
        cache, out = block(x, pos, mask)
        assert out.shape == (B, L, 64)

    def test_skip_scale(self):
        """skip_scale is a learnable parameter initialized to 1."""
        cfg = _small_text_config()
        block = TransformerBlock(cfg, layer_idx=0, attn_type=AttentionType.GLOBAL)
        assert block.skip_scale.shape == (1,)
        assert block.skip_scale.item() == pytest.approx(1.0)

    def test_with_pli(self):
        cfg = _small_text_config(per_layer_input_dim=16)
        block = TransformerBlock(cfg, layer_idx=0, attn_type=AttentionType.GLOBAL)
        assert block.pli_mapping is not None
        B, L = 1, 4
        x = torch.randn(B, L, 64)
        pos = torch.arange(L).unsqueeze(0)
        mask = torch.ones(B, L, L, dtype=torch.bool)
        pli = torch.randn(B, L, 16)
        cache, out = block(x, pos, mask, per_layer_input=pli)
        assert out.shape == (B, L, 64)


class TestTextDecoder:
    def test_end_to_end(self):
        cfg = _small_text_config()
        decoder = TextDecoder(cfg)
        B, L = 1, 8
        x = torch.randn(B, L, 64)
        pos = torch.arange(L).unsqueeze(0)
        mask = torch.ones(B, L, L, dtype=torch.bool)
        logits, _ = decoder(x, pos, mask)
        assert logits.shape == (B, L, 128)

    def test_with_cache(self):
        cfg = _small_text_config()
        decoder = TextDecoder(cfg)
        B, L, cache_len = 1, 4, 16

        # Build cache
        from gemma4_pt_claude.attention import Attention
        cache = {}
        for i in range(cfg.num_layers):
            cache[f"layer_{i}"] = Attention.init_cache(cache_len, 2, 16, B)

        x = torch.randn(B, L, 64)
        pos = torch.arange(L).unsqueeze(0)
        mask = torch.ones(B, L, cache_len, dtype=torch.bool)
        logits, new_cache = decoder(x, pos, mask, cache=cache)
        assert logits.shape == (B, L, 128)
        assert new_cache is not None
        assert len(new_cache) == cfg.num_layers
