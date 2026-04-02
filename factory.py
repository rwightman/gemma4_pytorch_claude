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


def _default_vision(text_embed_dim: int, use_clipped_linear: bool = True) -> VisionConfig:
    return VisionConfig(
        d_model=1152,
        num_layers=27,
        num_heads=16,
        mlp_dim=4304,
        patch_size=14,
        image_size=896,
        output_length=256,
        use_clipped_linear=use_clipped_linear,
        text_embed_dim=text_embed_dim,
    )


def _default_audio(text_embed_dim: int) -> AudioConfig:
    return AudioConfig(
        hidden_size=1536,
        num_layers=12,
        num_heads=8,
        text_embed_dim=text_embed_dim,
        lm_model_dims=text_embed_dim,
    )


# ---------------------------------------------------------------------------
# Gemma4 E2B (Nano-class, ~2B effective params)
# ---------------------------------------------------------------------------

def gemma4_e2b(text_only: bool = False) -> Gemma4Model:
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
        vision=None if text_only else _default_vision(embed_dim, use_clipped_linear=True),
        audio=None if text_only else _default_audio(embed_dim),
    )
    return Gemma4Model(cfg)


# ---------------------------------------------------------------------------
# Gemma4 E4B (Nano-class, ~4B effective params)
# ---------------------------------------------------------------------------

def gemma4_e4b(text_only: bool = False) -> Gemma4Model:
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
        vision=None if text_only else _default_vision(embed_dim, use_clipped_linear=True),
        audio=None if text_only else _default_audio(embed_dim),
    )
    return Gemma4Model(cfg)


# ---------------------------------------------------------------------------
# Gemma4 31B (Gemma3 27B successor)
# ---------------------------------------------------------------------------

def gemma4_31b(text_only: bool = False) -> Gemma4Model:
    """Gemma4 31B: 60 layers, embed=5376, 32H/16KV(4 global KV), 5:1, sw=1024, k_eq_v_global."""
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
        use_value_norm=False,
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
    vision = None
    if not text_only:
        vision = VisionConfig(
            d_model=1152,
            num_layers=27,
            num_heads=16,
            mlp_dim=4304,
            use_clipped_linear=False,
            text_embed_dim=embed_dim,
        )
    cfg = Gemma4Config(
        text=text,
        vision=vision,
    )
    return Gemma4Model(cfg)


# ---------------------------------------------------------------------------
# Gemma4 26B-A4B (MoE variant)
# ---------------------------------------------------------------------------

def gemma4_26b_a4b(text_only: bool = False) -> Gemma4Model:
    """Gemma4 26B-A4B: 30 layers, embed=2816, 16H/8KV(2 global KV), 5:1, sw=1024, MoE(128,top8,704), k_eq_v_global."""
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
        use_value_norm=False,
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
    vision = None
    if not text_only:
        vision = VisionConfig(
            d_model=1152,
            num_layers=27,
            num_heads=16,
            mlp_dim=4304,
            output_length=280,
            use_clipped_linear=False,
            text_embed_dim=embed_dim,
        )
    cfg = Gemma4Config(
        text=text,
        vision=vision,
    )
    return Gemma4Model(cfg)
