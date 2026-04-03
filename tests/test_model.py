"""Unit tests for top-level Gemma4Model and generation."""

import torch
import pytest

from gemma4_pt_claude.config import AttentionType, Gemma4Config, TextConfig
from gemma4_pt_claude.model import Gemma4Model, make_causal_mask, make_causal_mask_with_cache
from gemma4_pt_claude.generate import generate, init_cache


def _tiny_config() -> Gemma4Config:
    text = TextConfig(
        vocab_size=64,
        embed_dim=32,
        hidden_dim=64,
        num_heads=2,
        head_dim=16,
        num_kv_heads=2,
        num_layers=2,
        sliding_window_size=16,
        attention_pattern=(AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL),
        use_qk_norm=True,
        use_value_norm=False,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
    )
    return Gemma4Config(text=text)


class TestCausalMask:
    def test_causal_mask_shape(self):
        mask = make_causal_mask(8, torch.device("cpu"))
        assert mask.shape == (1, 8, 8)

    def test_causal_mask_is_lower_triangular(self):
        mask = make_causal_mask(4, torch.device("cpu"))
        assert mask[0, 0, 0].item() is True
        assert mask[0, 0, 1].item() is False
        assert mask[0, 3, 0].item() is True

    def test_cache_mask_prefill(self):
        mask = make_causal_mask_with_cache(4, 8, torch.device("cpu"))
        assert mask.shape == (1, 4, 8)

    def test_cache_mask_single_token(self):
        mask = make_causal_mask_with_cache(1, 8, torch.device("cpu"))
        assert mask.shape == (1, 1, 8)
        # Single token decode: all cache positions visible
        assert mask.all()


class TestGemma4Model:
    def test_forward_shape(self):
        cfg = _tiny_config()
        model = Gemma4Model(cfg)
        tokens = torch.randint(0, 64, (1, 8))
        logits, _ = model(tokens)
        assert logits.shape == (1, 8, 64)

    def test_forward_with_cache(self):
        cfg = _tiny_config()
        model = Gemma4Model(cfg)
        B, L = 1, 4
        cache = init_cache(cfg, B, 16)

        tokens = torch.randint(0, 64, (B, L))
        logits, new_cache = model(tokens, cache=cache)
        assert logits.shape == (B, L, 64)
        assert new_cache is not None

    def test_text_only(self):
        cfg = _tiny_config()
        model = Gemma4Model(cfg)
        assert model.vision_encoder is None
        assert model.audio_encoder is None


class TestGenerate:
    def test_generate_length(self):
        cfg = _tiny_config()
        model = Gemma4Model(cfg)
        model.eval()

        tokens = torch.randint(0, 64, (1, 4))
        output = generate(model, tokens, max_new_tokens=8, temperature=0.0)
        assert output.shape == (1, 12)  # 4 prompt + 8 generated

    def test_generate_greedy_deterministic(self):
        cfg = _tiny_config()
        model = Gemma4Model(cfg)
        model.eval()

        tokens = torch.randint(0, 64, (1, 4))
        out1 = generate(model, tokens, max_new_tokens=4, temperature=0.0)
        out2 = generate(model, tokens, max_new_tokens=4, temperature=0.0)
        assert torch.equal(out1, out2)
