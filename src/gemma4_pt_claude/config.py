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

    def last_unshared_of(attn_type: AttentionType) -> int | None:
        """Index of the last non-sharing layer with *attn_type*, if any."""
        for j in range(num_unshared - 1, -1, -1):
            if attention_types[j] == attn_type:
                return j
        return None

    share_source = {
        AttentionType.GLOBAL: last_unshared_of(AttentionType.GLOBAL) if kv_sharing.share_global else None,
        AttentionType.LOCAL_SLIDING: (
            last_unshared_of(AttentionType.LOCAL_SLIDING) if kv_sharing.share_local else None
        ),
    }

    patterns: list[int] = []
    for i in range(num_layers):
        source = None if i < num_unshared else share_source[attention_types[i]]
        # A layer with no eligible source computes its own KV.
        patterns.append(i if source is None else source)
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
    d_model: int = 768
    num_layers: int = 16
    num_heads: int = 12
    head_dim: int = 64
    ffw_hidden: int = 3072
    patch_size: int = 16
    output_length: int = 280          # soft tokens per image
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240  # max per-dim positions
    use_clipped_linear: bool = False
    standardize: bool = False
    rms_norm_eps: float = 1e-6
    init_std: float = 1e-2
    position_init_std: float = 2e-2
    residual_init_std: float | None = None
    use_depth_scaled_residual_init: bool = False
    rope_base_frequency: float = 100.0
    text_embed_dim: int = 1536        # projection target (set per variant)

    @property
    def max_patches(self) -> int:
        return self.output_length * self.pooling_kernel_size ** 2


@dataclass(frozen=True)
class EncoderFreeVisionConfig:
    """Vision config for encoder-free variants (12B, HF ``gemma4_unified``).

    These models have no vision tower at all.  Images are resized and patchified
    at ``patch_size`` exactly as for the tower models, then ``pooling_kernel_size``
    square groups of patches are merged into single ``model_patch_size`` patches
    which are projected straight into text space.  The pooling that the tower
    models apply *after* their encoder therefore happens here, before any
    projection.
    """

    patch_size: int = 16                # teacher patch size used for resizing
    pooling_kernel_size: int = 3        # k x k teacher patches merged per model patch
    output_length: int = 280            # soft tokens per image (max)
    mm_embed_dim: int = 3840            # width of the patch projection
    output_proj_dims: int = 3840        # input width of the final text projection
    position_embedding_size: int = 1120  # max per-axis positions
    rms_norm_eps: float = 1e-6
    init_std: float = 1e-2
    position_init_std: float = 2e-2
    text_embed_dim: int = 3840          # projection target

    @property
    def model_patch_size(self) -> int:
        """Side length of a merged patch (48 = 3 * 16)."""
        return self.patch_size * self.pooling_kernel_size

    @property
    def patch_dim(self) -> int:
        """Flattened size of one merged patch (48*48*3 = 6912)."""
        return self.model_patch_size ** 2 * 3

    @property
    def max_patches(self) -> int:
        """Teacher patches before merging."""
        return self.output_length * self.pooling_kernel_size ** 2


@dataclass(frozen=True)
class EncoderFreeAudioConfig:
    """Audio config for encoder-free variants (12B, HF ``gemma4_unified``).

    No mel spectrogram and no conformer: the raw 16 kHz waveform is chunked into
    fixed frames of ``audio_samples_per_token`` samples, and each frame becomes
    one soft token via a norm + linear projection.
    """

    audio_samples_per_token: int = 640   # 640 samples @ 16 kHz = 40 ms
    output_proj_dims: int = 640
    sample_rate: int = 16_000
    rms_norm_eps: float = 1e-6
    init_std: float = 1e-2
    text_embed_dim: int = 3840


@dataclass(frozen=True)
class AudioConfig:
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 8
    chunk_size: int = 12
    context_left: int = 13
    context_right: int = 0
    attn_logit_cap: float = 50.0
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    init_std: float = 1e-2
    residual_init_std: float | None = None
    use_depth_scaled_residual_init: bool = False
    input_feat_size: int = 128
    gradient_clipping: float = 1e10
    sscp_channels: tuple[int, int] = (128, 32)
    sscp_kernel_sizes: tuple[tuple[int, int], tuple[int, int]] = ((3, 3), (3, 3))
    sscp_stride_sizes: tuple[tuple[int, int], tuple[int, int]] = ((2, 2), (2, 2))
    sscp_group_norm_eps: float = 1e-3
    lm_model_dims: int = 1536


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
    init_std: float = 1e-2
    residual_init_std: float | None = None
    use_depth_scaled_residual_init: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gemma4Config:
    """Top-level config.

    ``vision``/``audio`` accept either the tower configs (E2B/E4B/31B/26B-A4B) or
    the encoder-free configs (12B); the model builds the matching modules.
    """

    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig | EncoderFreeVisionConfig | None = None
    audio: AudioConfig | EncoderFreeAudioConfig | None = None

    @property
    def is_encoder_free(self) -> bool:
        """True when the multimodal towers are replaced by direct projections."""
        return isinstance(self.vision, EncoderFreeVisionConfig) or isinstance(
            self.audio, EncoderFreeAudioConfig
        )
