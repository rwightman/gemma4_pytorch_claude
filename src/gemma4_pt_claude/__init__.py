"""Gemma4 — canonical PyTorch implementation.

Public API exports for model construction, configuration, and generation.
"""

# Configs
from .config import (
    AttentionType,
    AudioConfig,
    Gemma4Config,
    KVCacheSharingConfig,
    MoEConfig,
    TextConfig,
    VisionConfig,
    build_kv_sharing_patterns,
    make_attention_pattern,
)

# Core modules
from .layers import TanhGELU, RMSNorm, GatedMLP, ClippedLinear, apply_rope, apply_multidimensional_rope
from .attention import Attention, LayerCache
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
from .image_processing import preprocess_image, preprocess_images
from .audio_processing import preprocess_audio, extract_mel_spectrogram
from .composer import (
    Composer,
    ComposedInput,
    ImageTransform,
    PreparedImage,
    AudioTransform,
    PreparedAudio,
)

# Top-level model
from .model import Gemma4Model, VisionEmbedder, AudioEmbedder

# Generation
from .generate import generate, init_cache, chat

# Factory functions
from .factory import gemma4_e2b, gemma4_e4b, gemma4_31b, gemma4_26b_a4b

# Tokenizer
from .tokenizer import Gemma4Tokenizer

# Weight loading
from .load import load_weights

# Version
from .version import __version__

__all__ = [
    # Configs
    "AttentionType",
    "AudioConfig",
    "Gemma4Config",
    "KVCacheSharingConfig",
    "MoEConfig",
    "TextConfig",
    "VisionConfig",
    "build_kv_sharing_patterns",
    "make_attention_pattern",
    # Layers
    "RMSNorm",
    "TanhGELU",
    "GatedMLP",
    "ClippedLinear",
    "apply_rope",
    "apply_multidimensional_rope",
    # Attention
    "Attention",
    "LayerCache",
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
    # Image processing
    "preprocess_image",
    "preprocess_images",
    # Audio processing
    "preprocess_audio",
    "extract_mel_spectrogram",
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
    "gemma4_31b",
    "gemma4_26b_a4b",
    # Tokenizer
    "Gemma4Tokenizer",
    # Weight loading
    "load_weights",
    # Version
    "__version__",
]
