"""Unit tests for attention module."""

import torch
import pytest

from gemma4.attention import Attention, LayerCache, create_sliding_mask
from gemma4.config import AttentionType


class TestSlidingMask:
    def test_shape(self):
        pos = torch.arange(8).unsqueeze(0)  # [1, 8]
        mask = create_sliding_mask(pos, None, sliding_window_size=4)
        assert mask.shape == (1, 8, 8)

    def test_local_window(self):
        pos = torch.arange(10).unsqueeze(0)
        mask = create_sliding_mask(pos, None, sliding_window_size=3)
        # Position 5 should attend to positions 3,4,5,6,7 (within window of 3)
        row5 = mask[0, 5]
        assert row5[4].item() and row5[5].item() and row5[6].item()
        assert not row5[0].item()  # too far away


class TestAttention:
    @pytest.fixture
    def small_attn(self):
        return Attention(
                embed_dim=64,
                num_heads=4,
                num_kv_heads=4,
                head_dim=16,
                attn_type=AttentionType.GLOBAL,
                use_qk_norm=True,
                use_value_norm=False,
        )

    def test_output_shape(self, small_attn):
        B, L, D = 2, 8, 64
        x = torch.randn(B, L, D)
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        mask = torch.ones(B, L, L, dtype=torch.bool)
        _, out = small_attn(x, pos, mask)
        assert out.shape == (B, L, D)

    def test_gqa(self):
        """GQA: fewer KV heads than query heads."""
        attn = Attention(
                embed_dim=64,
                num_heads=8,
                num_kv_heads=2,
                head_dim=8,
                attn_type=AttentionType.GLOBAL,
                use_qk_norm=True,
        )
        B, L = 1, 4
        x = torch.randn(B, L, 64)
        pos = torch.arange(L).unsqueeze(0)
        mask = torch.ones(B, L, L, dtype=torch.bool)
        _, out = attn(x, pos, mask)
        assert out.shape == (B, L, 64)

    def test_k_eq_v(self):
        """k_eq_v=True should not have a v_proj parameter."""
        attn = Attention(
                embed_dim=32,
                num_heads=2,
                num_kv_heads=2,
                head_dim=16,
                attn_type=AttentionType.GLOBAL,
                use_qk_norm=False,
                k_eq_v=True,
        )
        assert not hasattr(attn, "v_proj")
        B, L = 1, 4
        x = torch.randn(B, L, 32)
        pos = torch.arange(L).unsqueeze(0)
        mask = torch.ones(B, L, L, dtype=torch.bool)
        _, out = attn(x, pos, mask)
        assert out.shape == (B, L, 32)

    def test_sliding_window(self):
        attn = Attention(
                embed_dim=32,
                num_heads=2,
                num_kv_heads=2,
                head_dim=16,
                attn_type=AttentionType.LOCAL_SLIDING,
                sliding_window_size=4,
                use_qk_norm=False,
        )
        B, L = 1, 8
        x = torch.randn(B, L, 32)
        pos = torch.arange(L).unsqueeze(0)
        mask = torch.ones(B, L, L, dtype=torch.bool)
        _, out = attn(x, pos, mask)
        assert out.shape == (B, L, 32)

    def test_kv_cache(self):
        attn = Attention(
                embed_dim=32,
                num_heads=2,
                num_kv_heads=2,
                head_dim=16,
                attn_type=AttentionType.GLOBAL,
                use_qk_norm=False,
        )
        B, cache_len = 1, 16
        cache = Attention.init_cache(cache_len, 2, 16, B)

        # Prefill with 4 tokens
        x = torch.randn(B, 4, 32)
        pos = torch.arange(4).unsqueeze(0)
        mask = torch.ones(B, 4, cache_len, dtype=torch.bool)
        new_cache, out = attn(x, pos, mask, cache=cache)
        assert out.shape == (B, 4, 32)
        assert new_cache["end_index"].item() == 4

        # Decode 1 token
        x2 = torch.randn(B, 1, 32)
        pos2 = torch.tensor([[4]])
        mask2 = torch.ones(B, 1, cache_len, dtype=torch.bool)
        new_cache2, out2 = attn(x2, pos2, mask2, cache=new_cache)
        assert out2.shape == (B, 1, 32)
        assert new_cache2["end_index"].item() == 5
