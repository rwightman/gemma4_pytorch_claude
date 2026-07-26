"""Gemma4 — canonical PyTorch implementation.

Public API exports for model construction, configuration, and generation.
"""

# Configs
from .config import (
    AttentionType,
    AudioConfig,
    EncoderFreeAudioConfig,
    EncoderFreeVisionConfig,
    Gemma4Config,
    KVCacheSharingConfig,
    MoEConfig,
    TextConfig,
    VisionConfig,
    build_kv_sharing_patterns,
    make_attention_pattern,
)

# Core modules
from .layers import TanhGELU, RMSNorm, VisionRMSNorm, GatedMLP, ClippedLinear, apply_rope, apply_multidimensional_rope
from .attention import Attention, LayerCache, cache_offset, create_sliding_mask
from .moe import MoELayer, MoERouter, MoEExperts
from .transformer import TextDecoder, TransformerBlock, Embedder
from .vision_encoder import (
    VisionEncoder,
    VisionPatchEmbedder,
    VisionAttention,
    VisionMLP,
    VisionBlock,
    VisionPooler,
)
from .audio_encoder import AudioEncoder
from .encoder_free import (
    EncoderFreeAudioEmbedder,
    EncoderFreeVisionEmbedder,
    MultimodalProjection,
)
from .image_processing import merge_patches, preprocess_image, preprocess_images
from .audio_processing import preprocess_audio, extract_mel_spectrogram, frame_waveform
from .composer import (
    Composer,
    ComposedInput,
    ImageTransform,
    PreparedImage,
    AudioTransform,
    PreparedAudio,
)
from .module_utils import InitContext

# Top-level model
from .model import Gemma4Model, VisionEmbedder, AudioEmbedder

# Generation
from .generate import generate, init_cache, chat

# Factory functions
from .factory import gemma4_e2b, gemma4_e4b, gemma4_12b, gemma4_31b, gemma4_26b_a4b

# Tokenizer
from .tokenizer import Gemma4Tokenizer

# Weight loading
from .load import from_pretrained, load_weights, load_weights_streaming

# Version
from .version import __version__

__all__ = [
    # Configs
    "AttentionType",
    "AudioConfig",
    "EncoderFreeAudioConfig",
    "EncoderFreeVisionConfig",
    "Gemma4Config",
    "InitContext",
    "KVCacheSharingConfig",
    "MoEConfig",
    "TextConfig",
    "VisionConfig",
    "build_kv_sharing_patterns",
    "make_attention_pattern",
    # Layers
    "RMSNorm",
    "VisionRMSNorm",
    "TanhGELU",
    "GatedMLP",
    "ClippedLinear",
    "apply_rope",
    "apply_multidimensional_rope",
    # Attention
    "Attention",
    "LayerCache",
    "cache_offset",
    "create_sliding_mask",
    # MoE
    "MoELayer",
    "MoERouter",
    "MoEExperts",
    # Transformer
    "TextDecoder",
    "TransformerBlock",
    "Embedder",
    # Encoders
    "VisionEncoder",
    "VisionPatchEmbedder",
    "VisionAttention",
    "VisionMLP",
    "VisionBlock",
    "VisionPooler",
    "AudioEncoder",
    "EncoderFreeVisionEmbedder",
    "EncoderFreeAudioEmbedder",
    "MultimodalProjection",
    # Image processing
    "merge_patches",
    "preprocess_image",
    "preprocess_images",
    # Audio processing
    "preprocess_audio",
    "extract_mel_spectrogram",
    "frame_waveform",
    # Composer
    "Composer",
    "ComposedInput",
    "ImageTransform",
    "PreparedImage",
    "AudioTransform",
    "PreparedAudio",
    # Model
    "Gemma4Model",
    "VisionEmbedder",
    "AudioEmbedder",
    # Generation
    "generate",
    "init_cache",
    "chat",
    # Factory
    "gemma4_e2b",
    "gemma4_e4b",
    "gemma4_12b",
    "gemma4_31b",
    "gemma4_26b_a4b",
    # Tokenizer
    "Gemma4Tokenizer",
    # Weight loading
    "from_pretrained",
    "load_weights",
    "load_weights_streaming",
    # Version
    "__version__",
]
