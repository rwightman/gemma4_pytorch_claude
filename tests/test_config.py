"""Unit tests for configuration and factory functions."""

import pytest

from gemma4.config import (
    AttentionType,
    Gemma4Config,
    KVCacheSharingConfig,
    TextConfig,
    build_kv_sharing_patterns,
    make_attention_pattern,
)
from gemma4.factory import gemma4_e2b, gemma4_e4b, gemma4_31b, gemma4_26b_a4b


class TestAttentionPattern:
    def test_4_1_pattern(self):
        pattern = (
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        )
        result = make_attention_pattern(pattern, 10)
        assert len(result) == 10
        assert result[4] == AttentionType.GLOBAL
        assert result[9] == AttentionType.GLOBAL

    def test_truncation(self):
        pattern = (AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL)
        result = make_attention_pattern(pattern, 3)
        assert len(result) == 3
        assert result == (
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
            AttentionType.LOCAL_SLIDING,
        )


class TestKVSharing:
    def test_no_sharing(self):
        patterns = build_kv_sharing_patterns(10, [AttentionType.GLOBAL] * 10, None)
        assert patterns == list(range(10))

    def test_zero_frac(self):
        cfg = KVCacheSharingConfig(frac_shared_layers=0.0)
        patterns = build_kv_sharing_patterns(10, [AttentionType.GLOBAL] * 10, cfg)
        assert patterns == list(range(10))

    def test_sharing_enabled(self):
        attn_types = list(make_attention_pattern(
            (AttentionType.LOCAL_SLIDING,) * 4 + (AttentionType.GLOBAL,),
            num_layers=10,
        ))
        cfg = KVCacheSharingConfig(
            frac_shared_layers=0.5,
            share_global=True,
            share_local=True,
        )
        patterns = build_kv_sharing_patterns(10, attn_types, cfg)
        assert len(patterns) == 10
        # First 5 layers should be unshared (own KV)
        assert patterns[:5] == list(range(5))
        # Shared layers should point to earlier layers
        for i in range(5, 10):
            assert patterns[i] < 5


class TestFactoryConfigs:
    """Verify factory configs match JAX reference values."""

    def test_e2b_config(self):
        m = gemma4_e2b(text_only=True)
        cfg = m.cfg.text
        assert cfg.num_layers == 35
        assert cfg.embed_dim == 1536
        assert cfg.num_heads == 8
        assert cfg.num_kv_heads == 1
        assert cfg.hidden_dim == 6144  # 1536 * 4
        assert cfg.head_dim == 256
        assert cfg.sliding_window_size == 512
        assert cfg.per_layer_input_dim == 256
        assert cfg.final_logit_softcap == 30.0
        assert cfg.global_head_dim == 512
        assert cfg.override_kv_shared_ffw_hidden == 12288  # 1536 * 4 * 2

    def test_e4b_config(self):
        m = gemma4_e4b(text_only=True)
        cfg = m.cfg.text
        assert cfg.num_layers == 42
        assert cfg.embed_dim == 2560
        assert cfg.num_heads == 8
        assert cfg.num_kv_heads == 2
        assert cfg.hidden_dim == 10240  # 2560 * 4
        assert cfg.sliding_window_size == 512
        # 5:1 pattern
        assert cfg.attention_pattern[5] == AttentionType.GLOBAL
        assert len(cfg.attention_pattern) == 6
        # KV sharing: 18/42
        assert cfg.kv_sharing is not None
        assert abs(cfg.kv_sharing.frac_shared_layers - 18.0 / 42) < 1e-6

    def test_31b_config(self):
        m = gemma4_31b(text_only=True)
        cfg = m.cfg.text
        assert cfg.num_layers == 60
        assert cfg.embed_dim == 5376
        assert cfg.num_heads == 32
        assert cfg.num_kv_heads == 16
        assert cfg.hidden_dim == 21504
        assert cfg.num_global_kv_heads == 4
        assert cfg.k_eq_v_global is True
        assert cfg.bidirectional_vision is True
        assert cfg.kv_sharing is None  # no sharing for 31B

    def test_26b_a4b_config(self):
        m = gemma4_26b_a4b(text_only=True)
        cfg = m.cfg.text
        assert cfg.num_layers == 30
        assert cfg.embed_dim == 2816
        assert cfg.num_heads == 16
        assert cfg.num_kv_heads == 8
        assert cfg.num_global_kv_heads == 2
        assert cfg.moe is not None
        assert cfg.moe.num_experts == 128
        assert cfg.moe.top_k == 8
        assert cfg.moe.expert_dim == 704
        assert cfg.moe.dense_hidden_dim == 2112
        assert cfg.k_eq_v_global is True

    def test_text_only_no_vision(self):
        m = gemma4_e2b(text_only=True)
        assert m.vision_encoder is None
        assert m.audio_encoder is None

    def test_with_vision(self):
        m = gemma4_e2b(text_only=False)
        assert m.vision_encoder is not None
        assert m.audio_encoder is not None
