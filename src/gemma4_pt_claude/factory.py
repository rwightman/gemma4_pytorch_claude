"""Factory functions for Gemma4 model variants.

Each function returns a fully-configured ``Gemma4Model``.
Pass ``text_only=True`` to omit vision/audio encoders.
"""

from __future__ import annotations

from .config import (
    AttentionType,
    AudioConfig,
    Gemma4Config,
    KVCacheSharingConfig,
    MoEConfig,
    TextConfig,
    VisionConfig,
)
from .model import Gemma4Model

# Common attention pattern for Gemma4 (4:1 local:global)
_GEMMA4_PATTERN_4_1 = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)

# 5:1 pattern (used by E4B, 31B, 26B-A4B)
_GEMMA4_PATTERN_5_1 = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)

_FFW_HIDDEN_RATIO = 4
_DEFAULT_GLOBAL_KEY_SIZE = 512


def _e2b_e4b_vision(
        text_embed_dim: int,
) -> VisionConfig:
    """Vision config for E2B/E4B (same encoder, different text_embed_dim)."""
    return VisionConfig(
        d_model=768,
        num_layers=16,
        num_heads=12,
        head_dim=64,
        ffw_hidden=3072,
        patch_size=16,
        output_length=280,
        pooling_kernel_size=3,
        position_embedding_size=10240,
        use_clipped_linear=True,
        text_embed_dim=text_embed_dim,
    )


def _large_vision(
        text_embed_dim: int,
) -> VisionConfig:
    """Vision config for 31B/26B-A4B (SigLiP-So400m-class)."""
    return VisionConfig(
        d_model=1152,
        num_layers=27,
        num_heads=16,
        head_dim=72,
        ffw_hidden=4304,
        patch_size=16,
        output_length=280,
        pooling_kernel_size=3,
        position_embedding_size=10240,
        use_clipped_linear=False,
        standardize=True,
        text_embed_dim=text_embed_dim,
    )


def _default_audio() -> AudioConfig:
    return AudioConfig(
        hidden_size=1024,
        num_layers=12,
        num_heads=8,
        lm_model_dims=1536,
    )


# ---------------------------------------------------------------------------
# Gemma4 E2B (Nano-class, ~2B effective params)
# ---------------------------------------------------------------------------

def gemma4_e2b(
        text_only: bool = False,
        *,
        device: str | None = None,
        dtype=None,
) -> Gemma4Model:
    """Gemma4 E2B: 35 layers, embed=1536, 8H/1KV, 4:1 pattern, sw=512, pli=256."""
    num_layers = 35
    embed_dim = 1536
    text = TextConfig(
        vocab_size=262_144,
        embed_dim=embed_dim,
        hidden_dim=embed_dim * _FFW_HIDDEN_RATIO,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        num_layers=num_layers,
        sliding_window_size=512,
        final_logit_softcap=30.0,
        attention_pattern=_GEMMA4_PATTERN_4_1,
        use_qk_norm=True,
        use_value_norm=True,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        per_layer_input_dim=256,
        local_rope_base=10_000,
        global_rope_base=1_000_000,
        rope_proportion=1.0,
        global_rope_proportion=0.25,
        global_head_dim=_DEFAULT_GLOBAL_KEY_SIZE,
        kv_sharing=KVCacheSharingConfig(
            frac_shared_layers=20.0 / num_layers,
            share_global=True,
            share_local=True,
        ),
        override_kv_shared_ffw_hidden=embed_dim * _FFW_HIDDEN_RATIO * 2,
    )
    cfg = Gemma4Config(
        text=text,
        vision=None if text_only else _e2b_e4b_vision(embed_dim),
        audio=None if text_only else _default_audio(),
    )
    return Gemma4Model(cfg, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Gemma4 E4B (Nano-class, ~4B effective params)
# ---------------------------------------------------------------------------

def gemma4_e4b(
        text_only: bool = False,
        *,
        device: str | None = None,
        dtype=None,
) -> Gemma4Model:
    """Gemma4 E4B: 42 layers, embed=2560, 8H/2KV, 5:1 pattern, sw=512, pli=256."""
    num_layers = 42
    embed_dim = 2560
    text = TextConfig(
        vocab_size=262_144,
        embed_dim=embed_dim,
        hidden_dim=embed_dim * _FFW_HIDDEN_RATIO,
        num_heads=8,
        head_dim=256,
        num_kv_heads=2,
        num_layers=num_layers,
        sliding_window_size=512,
        final_logit_softcap=30.0,
        attention_pattern=_GEMMA4_PATTERN_5_1,
        use_qk_norm=True,
        use_value_norm=True,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        per_layer_input_dim=256,
        local_rope_base=10_000,
        global_rope_base=1_000_000,
        rope_proportion=1.0,
        global_rope_proportion=0.25,
        global_head_dim=_DEFAULT_GLOBAL_KEY_SIZE,
        kv_sharing=KVCacheSharingConfig(
            frac_shared_layers=18.0 / num_layers,
            share_global=True,
            share_local=True,
        ),
    )
    cfg = Gemma4Config(
        text=text,
        vision=None if text_only else _e2b_e4b_vision(embed_dim),
        audio=None if text_only else _default_audio(),
    )
    return Gemma4Model(cfg, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Gemma4 31B (Gemma3 27B successor)
# ---------------------------------------------------------------------------

def gemma4_31b(
        text_only: bool = False,
        *,
        device: str | None = None,
        dtype=None,
) -> Gemma4Model:
    """Gemma4 31B: 60 layers, embed=5376, 32H/16KV(4 global KV), 5:1, sw=1024."""
    num_layers = 60
    embed_dim = 5376
    text = TextConfig(
        vocab_size=262_144,
        embed_dim=embed_dim,
        hidden_dim=embed_dim * _FFW_HIDDEN_RATIO,
        num_heads=32,
        head_dim=256,
        num_kv_heads=16,
        num_layers=num_layers,
        sliding_window_size=1024,
        final_logit_softcap=30.0,
        attention_pattern=_GEMMA4_PATTERN_5_1,
        use_qk_norm=True,
        use_value_norm=True,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        local_rope_base=10_000,
        global_rope_base=1_000_000,
        rope_proportion=1.0,
        global_rope_proportion=0.25,
        k_eq_v_global=True,
        bidirectional_vision=True,
        num_global_kv_heads=4,
        global_head_dim=_DEFAULT_GLOBAL_KEY_SIZE,
    )
    cfg = Gemma4Config(
        text=text,
        vision=None if text_only else _large_vision(embed_dim),
    )
    return Gemma4Model(cfg, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Gemma4 26B-A4B (MoE variant)
# ---------------------------------------------------------------------------

def gemma4_26b_a4b(
        text_only: bool = False,
        *,
        device: str | None = None,
        dtype=None,
) -> Gemma4Model:
    """Gemma4 26B-A4B: 30 layers, embed=2816, 16H/8KV(2 global KV), 5:1, MoE."""
    num_layers = 30
    embed_dim = 2816
    text = TextConfig(
        vocab_size=262_144,
        embed_dim=embed_dim,
        hidden_dim=2112,  # Dense shared MLP (mlp2) hidden dim
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        num_layers=num_layers,
        sliding_window_size=1024,
        final_logit_softcap=30.0,
        attention_pattern=_GEMMA4_PATTERN_5_1,
        use_qk_norm=True,
        use_value_norm=True,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        local_rope_base=10_000,
        global_rope_base=1_000_000,
        rope_proportion=1.0,
        global_rope_proportion=0.25,
        k_eq_v_global=True,
        bidirectional_vision=True,
        num_global_kv_heads=2,
        global_head_dim=_DEFAULT_GLOBAL_KEY_SIZE,
        moe=MoEConfig(
            num_experts=128,
            top_k=8,
            expert_dim=704,
            dense_hidden_dim=2112,
        ),
    )
    cfg = Gemma4Config(
        text=text,
        vision=None if text_only else _large_vision(embed_dim),
    )
    return Gemma4Model(cfg, device=device, dtype=dtype)
