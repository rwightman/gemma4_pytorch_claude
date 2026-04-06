"""Unit tests for top-level Gemma4Model and generation."""

import torch
import pytest

from gemma4_pt_claude.config import AttentionType, Gemma4Config, TextConfig
from gemma4_pt_claude.model import (
    Gemma4Model,
    build_audio_token_mask,
    flatten_multimodal_tokens,
    make_causal_mask,
    make_causal_mask_with_cache,
    make_causal_bidirectional_mask,
)
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

    def test_cache_mask_prefill_from_empty(self):
        # Prefill 4 tokens into an 8-slot cache starting empty (offset=0)
        mask = make_causal_mask_with_cache(4, 8, 0, torch.device("cpu"))
        assert mask.shape == (1, 4, 8)
        # Should be causal in first 4 columns, all-True in remaining
        # (valid_mask in attention handles the unfilled slots)
        assert mask[0, 0, 0].item() is True
        assert mask[0, 0, 1].item() is False  # row 0 can't see future
        assert mask[0, 3, 3].item() is True
        assert mask[0, 3, 4].item() is False   # no diagonal shift

    def test_cache_mask_prefill_with_offset(self):
        # 4 tokens into 8-slot cache with 2 previous entries (offset=2)
        mask = make_causal_mask_with_cache(4, 8, 2, torch.device("cpu"))
        assert mask.shape == (1, 4, 8)
        # Row 0: columns 0,1 (prev) + column 2 (self) = True; column 3+ = False
        assert mask[0, 0, 1].item() is True   # previous entry
        assert mask[0, 0, 2].item() is True   # self (first new token)
        assert mask[0, 0, 3].item() is False  # future new token

    def test_cache_mask_single_token(self):
        mask = make_causal_mask_with_cache(1, 8, 5, torch.device("cpu"))
        assert mask.shape == (1, 1, 8)
        # Single token at offset 5: can see columns 0..5
        assert mask[0, 0, 5].item() is True
        assert mask[0, 0, 6].item() is False


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


class TestCacheValidMask:
    def test_valid_mask_after_prefill(self):
        cfg = _tiny_config()
        model = Gemma4Model(cfg)
        B, L = 1, 4
        cache = init_cache(cfg, B, 16)

        tokens = torch.randint(0, 64, (B, L))
        _, new_cache = model(tokens, cache=cache)

        first_layer = new_cache["layer_0"]
        # First L positions should be valid
        assert first_layer["valid_mask"][:, :L].all()
        # Rest should be invalid
        assert not first_layer["valid_mask"][:, L:].any()

    def test_valid_mask_after_decode_step(self):
        cfg = _tiny_config()
        model = Gemma4Model(cfg)
        B, L = 1, 4
        cache = init_cache(cfg, B, 16)

        # Prefill
        tokens = torch.randint(0, 64, (B, L))
        _, cache = model(tokens, cache=cache)

        # Decode 1 token
        next_tok = torch.randint(0, 64, (B, 1))
        _, cache = model(next_tok, cache=cache)

        first_layer = cache["layer_0"]
        # First L+1 positions should be valid
        assert first_layer["valid_mask"][:, :L + 1].all()
        assert not first_layer["valid_mask"][:, L + 1:].any()


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


class TestAudioHelpers:
    def test_build_audio_token_mask_caps_tokens(self):
        counts = torch.tensor([3, 1], dtype=torch.long)
        mask = build_audio_token_mask(counts, total_audio_tokens=5)
        expected = torch.tensor(
            [
                [True, True, True, False, False],
                [True, False, False, False, False],
            ],
            dtype=torch.bool,
        )
        assert torch.equal(mask, expected)

    def test_flatten_multimodal_tokens_single_prompt(self):
        mm_embeddings = torch.tensor(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
            ]
        )
        mm_mask = torch.tensor(
            [
                [True, False, True],
                [False, True, False],
            ],
            dtype=torch.bool,
        )

        flat_embeddings, flat_mask = flatten_multimodal_tokens(mm_embeddings, mm_mask)

        expected_embeddings = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]])
        expected_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
        assert torch.equal(flat_embeddings, expected_embeddings)
        assert torch.equal(flat_mask, expected_mask)
