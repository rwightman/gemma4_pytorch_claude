"""Gemma4 configuration dataclasses.

All model configs live here: text, vision, audio, MoE, and top-level.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AttentionType(enum.Enum):
    LOCAL_SLIDING = 1
    GLOBAL = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_attention_pattern(
        pattern: tuple[AttentionType, ...],
        num_layers: int,
) -> tuple[AttentionType, ...]:
    """Tile *pattern* to cover *num_layers*, truncating the last repeat."""
    n = len(pattern)
    full = pattern * (num_layers // n)
    remainder = pattern[: num_layers % n]
    return full + remainder


def build_kv_sharing_patterns(
        num_layers: int,
        attention_types: Sequence[AttentionType],
        kv_sharing: KVCacheSharingConfig | None,
) -> list[int]:
    """Return per-layer index saying *which* layer's KV to reuse.

    Layer *i* uses KV from ``patterns[i]``.  When ``patterns[i] == i`` the
    layer computes its own KV; otherwise it borrows from an earlier layer.
    """
    if kv_sharing is None or kv_sharing.frac_shared_layers == 0.0:
        return list(range(num_layers))

    num_unshared = int(num_layers - kv_sharing.frac_shared_layers * num_layers)
    patterns: list[int] = []
    for i in range(num_layers):
        if i < num_unshared:
            patterns.append(i)
        else:
            if (
                attention_types[i] == AttentionType.GLOBAL
                and kv_sharing.share_global
            ):
                # Share with last unshared global layer
                patterns.append(num_unshared - 1)
            elif (
                attention_types[i] == AttentionType.LOCAL_SLIDING
                and kv_sharing.share_local
            ):
                # Share with last unshared local layer
                patterns.append(num_unshared - 2)
            else:
                patterns.append(i)
    return patterns


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KVCacheSharingConfig:
    frac_shared_layers: float = 0.0
    share_global: bool = False
    share_local: bool = False


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int = 128
    top_k: int = 8
    expert_dim: int = 704
    dense_hidden_dim: int = 0  # 0 means no parallel dense branch


@dataclass(frozen=True)
class VisionConfig:
    d_model: int = 1152
    num_layers: int = 27
    num_heads: int = 16
    mlp_dim: int = 4304
    patch_size: int = 14
    image_size: int = 896
    output_length: int = 256
    use_clipped_linear: bool = False
    text_embed_dim: int = 2048  # projection target


@dataclass(frozen=True)
class AudioConfig:
    hidden_size: int = 1536
    num_layers: int = 12
    num_heads: int = 8
    chunk_size: int = 12
    context_left: int = 13
    context_right: int = 0
    attn_logit_cap: float = 50.0
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    input_feat_size: int = 128
    gradient_clipping: float = 1e10
    sscp_channels: tuple[int, int] = (128, 32)
    sscp_kernel_sizes: tuple[tuple[int, int], tuple[int, int]] = ((3, 3), (3, 3))
    sscp_stride_sizes: tuple[tuple[int, int], tuple[int, int]] = ((2, 2), (2, 2))
    sscp_group_norm_eps: float = 1e-3
    text_embed_dim: int = 2048  # projection target
    lm_model_dims: int = 2048  # intermediate proj dim (can differ from text_embed_dim)


# ---------------------------------------------------------------------------
# Text config
# ---------------------------------------------------------------------------

GEMMA4_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


@dataclass(frozen=True)
class TextConfig:
    vocab_size: int = 262_144
    embed_dim: int = 2048
    hidden_dim: int = 8192
    num_heads: int = 8
    head_dim: int = 256
    num_kv_heads: int = 2
    num_layers: int = 35

    sliding_window_size: int = 512
    final_logit_softcap: float | None = None
    attn_logits_soft_cap: float | None = None

    local_rope_base: int = 10_000
    global_rope_base: int = 1_000_000
    global_rope_scale_factor: float = 1.0
    rope_proportion: float = 1.0
    global_rope_proportion: float = 0.25

    use_qk_norm: bool = True
    use_value_norm: bool = True
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True

    per_layer_input_dim: int = 0
    attention_pattern: tuple[AttentionType, ...] = GEMMA4_ATTENTION_PATTERN
    kv_sharing: KVCacheSharingConfig | None = None
    moe: MoEConfig | None = None

    # Whether K==V (shared projection) for local sliding layers
    k_eq_v: bool = False
    # Whether K==V for global layers only (separate from k_eq_v)
    k_eq_v_global: bool = False
    # Whether to use bidirectional attention for vision tokens
    bidirectional_vision: bool = False

    # Global layers can have different KV head count and head dim
    num_global_kv_heads: int | None = None  # None → same as num_kv_heads
    global_head_dim: int | None = None      # None → same as head_dim

    # Override FFW hidden dim for KV-shared layers
    override_kv_shared_ffw_hidden: int | None = None

    # Attention implementation: "sdpa" (F.scaled_dot_product_attention) or "eager" (manual einsum)
    attn_impl: str = "sdpa"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gemma4Config:
    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig | None = None
    audio: AudioConfig | None = None
