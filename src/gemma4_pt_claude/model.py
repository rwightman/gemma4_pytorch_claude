"""Top-level Gemma4Model: text decoder, vision encoder, audio encoder as peers."""

from __future__ import annotations

import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder
from .config import Gemma4Config
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
        mm_embeddings: ``[B, P, D]`` where P is total number of MM tokens
        placeholder_mask: ``[B, L]`` bool — True at positions to replace

    Returns:
        ``[B, L, D]`` with MM embeddings inserted.
    """
    result = text_embeddings.clone()
    for b in range(text_embeddings.shape[0]):
        indices = placeholder_mask[b].nonzero(as_tuple=True)[0]
        n = min(len(indices), mm_embeddings.shape[1])
        if n > 0:
            result[b, indices[:n]] = mm_embeddings[b, :n]
    return result


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
        if cfg.vision is not None:
            self.vision_encoder = VisionEncoder(cfg.vision)

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
            image_patches: torch.Tensor | None = None,
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
            image_patches: ``[B, N, P, C]`` or ``[B, P, C]`` — patchified images
            image_mask: ``[B, L]`` bool — True at image placeholder positions
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

        # --- Vision: encode + merge ---
        if image_patches is not None and self.vision_encoder is not None:
            vision_embeddings = self.vision_encoder(image_patches)
            if image_mask is not None:
                x = merge_multimodal_embeddings(x, vision_embeddings.view(B, -1, x.shape[-1]), image_mask)

        # --- Audio: encode + merge ---
        if audio_mel is not None and self.audio_encoder is not None:
            assert audio_mel_mask is not None, "audio_mel_mask required when audio_mel is provided"
            audio_embeddings = self.audio_encoder(audio_mel, audio_mel_mask)
            # Audio embeddings would be merged similarly to vision
            # (placeholder positions would be marked in the token sequence)

        # --- Per-layer inputs ---
        per_layer_inputs = None
        if self.text_decoder.embedder.per_layer_input_dim > 0:
            per_layer_inputs = self.text_decoder.embedder.encode_per_layer_input(x, tokens)

        # --- Run text decoder ---
        logits, new_cache = self.text_decoder(
            x, positions, attention_mask,
            per_layer_inputs=per_layer_inputs,
            cache=cache,
        )

        return logits, new_cache
