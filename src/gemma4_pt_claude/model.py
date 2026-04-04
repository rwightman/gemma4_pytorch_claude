"""Top-level Gemma4Model: text decoder, vision encoder, audio encoder as peers."""

from __future__ import annotations

import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder
from .config import Gemma4Config
from .layers import RMSNorm
from .transformer import TextDecoder
from .vision_encoder import VisionEncoder


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular bool mask ``[1, L, L]``."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).unsqueeze(0)


def make_causal_mask_with_cache(
        seq_len: int,
        cache_len: int,
        device: torch.device,
) -> torch.Tensor:
    """``[1, seq_len, cache_len]`` causal mask suitable for cached decoding."""
    mask = torch.ones(seq_len, cache_len, dtype=torch.bool, device=device)
    # For single-token decode, all cached positions are visible
    if seq_len == 1:
        return mask.unsqueeze(0)
    # For prefill, use causal
    return torch.tril(
        torch.ones(seq_len, cache_len, dtype=torch.bool, device=device),
        diagonal=cache_len - seq_len,
    ).unsqueeze(0)


# ---------------------------------------------------------------------------
# Multimodal embedding merge
# ---------------------------------------------------------------------------

def merge_multimodal_embeddings(
        text_embeddings: torch.Tensor,
        mm_embeddings: torch.Tensor,
        placeholder_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter multimodal embeddings into text embedding tensor at placeholder positions.

    Args:
        text_embeddings: ``[B, L, D]``
        mm_embeddings: flat tensor of all valid soft tokens
        placeholder_mask: ``[B, L]`` bool — True at positions to replace

    Returns:
        ``[B, L, D]`` with MM embeddings inserted.
    """
    mask_3d = placeholder_mask.unsqueeze(-1).expand_as(text_embeddings)
    return text_embeddings.masked_scatter(
        mask_3d.to(text_embeddings.device),
        mm_embeddings.to(text_embeddings.device),
    )


# ---------------------------------------------------------------------------
# MultimodalEmbedder
# ---------------------------------------------------------------------------

class MultimodalEmbedder(nn.Module):
    """Project multimodal encoder outputs into text embedding space."""

    def __init__(self, mm_dim: int, text_dim: int, eps: float = 1e-6):
        super().__init__()
        self.pre_norm = RMSNorm(mm_dim, with_scale=False, eps=eps)
        self.proj = nn.Linear(mm_dim, text_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pre_norm(x))


# ---------------------------------------------------------------------------
# Gemma4Model
# ---------------------------------------------------------------------------

class Gemma4Model(nn.Module):
    """Top-level Gemma4 model.

    Architecture: text_decoder, vision_encoder, audio_encoder as peers.
    Vision/audio encoders are optional and fully standalone.
    """

    def __init__(self, cfg: Gemma4Config):
        super().__init__()
        self.cfg = cfg
        self.text_decoder = TextDecoder(cfg.text)

        self.vision_encoder = None
        self.embed_vision = None
        if cfg.vision is not None:
            self.vision_encoder = VisionEncoder(cfg.vision)
            self.embed_vision = MultimodalEmbedder(
                cfg.vision.d_model, cfg.vision.text_embed_dim,
                eps=cfg.vision.rms_norm_eps,
            )

        self.audio_encoder = None
        if cfg.audio is not None:
            self.audio_encoder = AudioEncoder(cfg.audio)

    def forward(
            self,
            tokens: torch.Tensor,
            positions: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            cache: dict | None = None,
            # Vision inputs
            pixel_values: torch.Tensor | None = None,
            image_position_ids: torch.Tensor | None = None,
            image_mask: torch.Tensor | None = None,
            # Audio inputs
            audio_mel: torch.Tensor | None = None,
            audio_mel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            tokens: ``[B, L]`` token ids
            positions: ``[B, L]`` absolute positions (auto-generated if None)
            attention_mask: ``[B, L, S]`` bool (auto-generated causal if None)
            cache: KV cache dict (optional)
            pixel_values: ``[B, max_patches, patch_dim]`` — patchified images
            image_position_ids: ``[B, max_patches, 2]`` — (x,y) coords (-1 = padding)
            image_mask: ``[B, L]`` bool — True at image placeholder positions in text
            audio_mel: ``[B, T, F]`` — mel spectrograms
            audio_mel_mask: ``[B, T]`` bool — True for padded frames

        Returns:
            (logits, new_cache)
        """
        B, L = tokens.shape
        device = tokens.device

        # --- Default positions ---
        if positions is None:
            if cache is not None:
                # Get offset from cache
                first_cache = next(iter(cache.values()))
                offset = first_cache["end_index"][0].item()
                positions = torch.arange(offset, offset + L, device=device).unsqueeze(0).expand(B, -1)
            else:
                positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # --- Default causal mask ---
        if attention_mask is None:
            if cache is not None:
                cache_len = next(iter(cache.values()))["k"].shape[1]
                attention_mask = make_causal_mask_with_cache(L, cache_len, device).expand(B, -1, -1)
            else:
                attention_mask = make_causal_mask(L, device).expand(B, -1, -1)

        # --- Embed text tokens ---
        x = self.text_decoder.embedder.encode(tokens)

        # --- Vision: encode + project + merge ---
        if pixel_values is not None and self.vision_encoder is not None:
            vision_out, _ = self.vision_encoder(pixel_values, image_position_ids)
            vision_embeddings = self.embed_vision(vision_out)
            vision_embeddings = vision_embeddings.to(x.device, x.dtype)
            if image_mask is not None:
                x = merge_multimodal_embeddings(x, vision_embeddings, image_mask)

        # --- Audio: encode + merge ---
        if audio_mel is not None and self.audio_encoder is not None:
            assert audio_mel_mask is not None, "audio_mel_mask required when audio_mel is provided"
            audio_embeddings = self.audio_encoder(audio_mel, audio_mel_mask)
            # Audio embeddings would be merged similarly to vision
            # (placeholder positions would be marked in the token sequence)

        # --- Per-layer inputs ---
        # Replace image placeholder tokens with pad (0) for PLI lookup — image
        # positions get vision embeddings via the merge above, so PLI should use
        # the neutral pad embedding at those positions (matches JAX behaviour).
        per_layer_inputs = None
        if self.text_decoder.embedder.per_layer_input_dim > 0:
            pli_tokens = tokens
            if image_mask is not None:
                pli_tokens = tokens.clone()
                pli_tokens[image_mask] = 0
            per_layer_inputs = self.text_decoder.embedder.encode_per_layer_input(x, pli_tokens)

        # --- Run text decoder ---
        logits, new_cache = self.text_decoder(
            x, positions, attention_mask,
            per_layer_inputs=per_layer_inputs,
            cache=cache,
        )

        return logits, new_cache
