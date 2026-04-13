"""Top-level Gemma4Model: text decoder, vision encoder, audio encoder as peers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_encoder import AudioEncoder
from .config import Gemma4Config
from .layers import RMSNorm
from .module_utils import InitContext, InitModule, factory_kwargs
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
        cache_offset: int,
        device: torch.device,
) -> torch.Tensor:
    """``[1, seq_len, cache_len]`` causal mask suitable for cached decoding.

    Args:
        seq_len: number of new query tokens.
        cache_len: total cache capacity.
        cache_offset: number of entries already in the cache (``end_index``).
        device: torch device.

    The mask allows each query to attend to all previous cache entries
    (columns ``0..cache_offset-1``) and causally within the new tokens
    (columns ``cache_offset..cache_offset+seq_len-1``).  Unfilled slots
    beyond ``cache_offset+seq_len`` are left True here — ``valid_mask``
    in the attention layer handles them.
    """
    return torch.tril(
        torch.ones(seq_len, cache_len, dtype=torch.bool, device=device),
        diagonal=cache_offset,
    ).unsqueeze(0)


def make_causal_bidirectional_mask(
        causal_mask: torch.Tensor,
        bidirectional_mask: torch.Tensor,
) -> torch.Tensor:
    """Augment causal mask so tokens within contiguous True regions attend bidirectionally.

    Args:
        causal_mask: ``[B, L, S]`` bool.
        bidirectional_mask: ``[B, L]`` bool — True at image positions.

    Returns:
        ``[B, L, S]`` bool with bidirectional attention within contiguous
        True spans of ``bidirectional_mask``.
    """
    # Number contiguous True regions with unique block IDs
    padded = F.pad(bidirectional_mask.long(), (1, 0), value=0)
    boundary = padded[:, 1:] > padded[:, :-1]  # rising edges
    block_ids = bidirectional_mask.long() * boundary.long().cumsum(dim=-1)  # [B, L]

    # Allow bidirectional within same block (same nonzero block_id)
    S = causal_mask.shape[2]
    # Truncate or pad block_ids to match S dimension (for cache case, S may differ from L)
    if S != block_ids.shape[1]:
        # During cached decode L=1, S=cache_len; block_ids is [B, L=1].
        # Bidirectional mask is only meaningful during prefill (L == S).
        return causal_mask

    bidir = (block_ids[:, :, None] == block_ids[:, None, :]) & (block_ids[:, :, None] > 0)
    return causal_mask | bidir


# ---------------------------------------------------------------------------
# Multimodal embedding merge
# ---------------------------------------------------------------------------

def merge_multimodal_embeddings(
        text_embeddings: torch.Tensor,
        mm_embeddings: torch.Tensor,
        mm_mask: torch.Tensor,
        placeholder_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter multimodal embeddings into text embedding tensor at placeholder positions.

    Args:
        text_embeddings: ``[B, L, D]``
        mm_embeddings: ``[B, T, D]`` — pooled encoder outputs
        mm_mask: ``[B, T]`` bool — True for valid soft tokens in mm_embeddings
        placeholder_mask: ``[B, L]`` bool — True at positions to replace

    Returns:
        ``[B, L, D]`` with MM embeddings inserted.
    """
    out = text_embeddings.clone()
    for b in range(text_embeddings.shape[0]):
        tgt = placeholder_mask[b].nonzero(as_tuple=True)[0]
        src = mm_mask[b].nonzero(as_tuple=True)[0]
        assert tgt.shape[0] == src.shape[0], (
            f"Batch {b}: {tgt.shape[0]} placeholders vs {src.shape[0]} soft tokens"
        )
        out[b, tgt] = mm_embeddings[b, src].to(out.dtype)
    return out


def build_audio_token_mask(
        num_soft_tokens: torch.Tensor,
        total_audio_tokens: int,
) -> torch.Tensor:
    """Build a per-clip valid-token mask capped by explicit soft-token counts."""
    counts = num_soft_tokens.to(dtype=torch.long).clamp(min=0, max=total_audio_tokens)
    positions = torch.arange(total_audio_tokens, device=counts.device)
    return positions.unsqueeze(0) < counts.unsqueeze(1)


def flatten_multimodal_tokens(
        mm_embeddings: torch.Tensor,
        mm_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten valid multimodal tokens across clip/image entries for a single prompt."""
    pieces = []
    for idx in range(mm_embeddings.shape[0]):
        valid = mm_mask[idx]
        if valid.any():
            pieces.append(mm_embeddings[idx, valid])

    if not pieces:
        empty_embeddings = mm_embeddings.new_zeros(1, 0, mm_embeddings.shape[-1])
        empty_mask = torch.zeros(1, 0, dtype=torch.bool, device=mm_mask.device)
        return empty_embeddings, empty_mask

    flat = torch.cat(pieces, dim=0).unsqueeze(0)
    flat_mask = torch.ones(1, flat.shape[1], dtype=torch.bool, device=flat.device)
    return flat, flat_mask


# ---------------------------------------------------------------------------
# Multimodal Embedders
# ---------------------------------------------------------------------------

class VisionEmbedder(InitModule):
    """Vision: RMSNorm → Linear projection (norm before projection)."""

    def __init__(
            self,
            mm_dim: int,
            text_dim: int,
            init_std: float,
            eps: float = 1e-6,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.init_std = init_std
        dd = factory_kwargs(device, dtype)
        self.norm = RMSNorm(mm_dim, with_scale=False, eps=eps, **dd)
        self.proj = nn.Linear(mm_dim, text_dim, bias=False, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))

    def _init_weights(self, ctx) -> None:
        nn.init.normal_(self.proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        if self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)


class AudioEmbedder(InitModule):
    """Audio: RMSNorm → Linear projection (norm before projection)."""

    def __init__(
            self,
            mm_dim: int,
            text_dim: int,
            init_std: float,
            eps: float = 1e-6,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.init_std = init_std
        dd = factory_kwargs(device, dtype)
        self.norm = RMSNorm(mm_dim, with_scale=False, eps=eps, **dd)
        self.proj = nn.Linear(mm_dim, text_dim, bias=False, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))

    def _init_weights(self, ctx) -> None:
        nn.init.normal_(self.proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        if self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)


# ---------------------------------------------------------------------------
# Gemma4Model
# ---------------------------------------------------------------------------

class Gemma4Model(InitModule):
    """Top-level Gemma4 model.

    Architecture: text_decoder, vision_encoder, audio_encoder as peers.
    Vision/audio encoders are optional and fully standalone.
    """

    def __init__(
            self,
            cfg: Gemma4Config,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.cfg = cfg
        self.text_decoder = TextDecoder(cfg.text, **dd)

        self.vision_encoder = None
        self.embed_vision = None
        if cfg.vision is not None:
            self.vision_encoder = VisionEncoder(cfg.vision, **dd)
            self.embed_vision = VisionEmbedder(
                cfg.vision.d_model, cfg.vision.text_embed_dim,
                init_std=cfg.text.init_std,
                eps=cfg.vision.rms_norm_eps,
                **dd,
            )

        self.audio_encoder = None
        self.embed_audio = None
        if cfg.audio is not None:
            self.audio_encoder = AudioEncoder(cfg.audio, **dd)
            self.embed_audio = AudioEmbedder(
                cfg.audio.lm_model_dims, cfg.text.embed_dim,
                init_std=cfg.text.init_std,
                **dd,
            )

        if not any(param.is_meta for param in self.parameters()):
            self.init_weights()

    def _init_non_persistent_buffers(self) -> None:
        return

    def materialize(
            self,
            *,
            device: torch.device | str,
            dtype: torch.dtype | None = None,
            init_weights: bool = True,
            ctx: InitContext | None = None,
    ) -> "Gemma4Model":
        target_device = torch.device(device)
        if any(param.device.type == "meta" for param in self.parameters()):
            if dtype is not None:
                self.to(dtype=dtype)
            self.to_empty(device=target_device)
        else:
            self.to(device=target_device, dtype=dtype)
        if init_weights:
            self.init_weights(ctx)
        else:
            self.init_non_persistent_buffers()
        return self

    def _init_weights(self, ctx) -> None:
        return

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
            audio_mask: torch.Tensor | None = None,
            audio_num_soft_tokens: torch.Tensor | None = None,
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
            audio_mel_mask: ``[B, T]`` bool — True for valid mel frames
            audio_mask: ``[B, L]`` bool — True at audio placeholder positions in text
            audio_num_soft_tokens: ``[B_audio]`` int — explicit soft-token counts per clip

        Returns:
            (logits, new_cache)
        """
        B, L = tokens.shape
        device = tokens.device

        # --- Default positions ---
        if positions is None:
            if cache is not None:
                # All batch elements must share the same cache fill level
                first_cache = next(iter(cache.values()))
                offset = first_cache["end_index"][0].item()
                assert (first_cache["end_index"] == offset).all(), (
                    "Heterogeneous batch end_index not supported"
                )
                positions = torch.arange(offset, offset + L, device=device).unsqueeze(0).expand(B, -1)
            else:
                positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # --- Default causal mask ---
        if attention_mask is None:
            if cache is not None:
                first_cache = next(iter(cache.values()))
                cache_len = first_cache["k"].shape[1]
                cache_offset = first_cache["end_index"][0].item()
                # Build causal mask; valid_mask is applied inside attention.py
                attention_mask = make_causal_mask_with_cache(
                    L, cache_len, cache_offset, device,
                ).expand(B, -1, -1)
            else:
                attention_mask = make_causal_mask(L, device).expand(B, -1, -1)

        # --- Embed text tokens ---
        # Sanitize negative placeholder tokens for embedding lookup.
        # Internal soft-token placeholders (-2 image, -4 audio) can't index nn.Embedding.
        # Replace with PAD (0); actual embeddings come from encoders via merge below.
        embed_tokens = tokens
        if image_mask is not None or audio_mask is not None:
            embed_tokens = tokens.clone()
            if image_mask is not None:
                embed_tokens[image_mask] = 0
            if audio_mask is not None:
                embed_tokens[audio_mask] = 0
        x = self.text_decoder.embedder.encode(embed_tokens)

        # --- Vision: encode + project + merge ---
        if pixel_values is not None and self.vision_encoder is not None:
            vision_out, pooler_mask = self.vision_encoder(pixel_values, image_position_ids)
            vision_embeddings = self.embed_vision(vision_out)
            vision_embeddings = vision_embeddings.to(x.device, x.dtype)
            if image_mask is not None:
                x = merge_multimodal_embeddings(
                    x, vision_embeddings, pooler_mask, image_mask,
                )

        # --- Audio: encode + project + merge ---
        if audio_mel is not None and self.audio_encoder is not None:
            assert audio_mel_mask is not None, "audio_mel_mask required when audio_mel is provided"
            # Cast mel to encoder dtype (may differ from text dtype — JAX keeps
            # audio encoder in float32 even when text is bfloat16).
            enc_dtype = next(self.audio_encoder.parameters()).dtype
            audio_mel = audio_mel.to(enc_dtype)
            # Encoder expects True=padded; our public API uses True=valid
            audio_out, audio_pad_mask = self.audio_encoder(audio_mel, ~audio_mel_mask)
            audio_embeddings = self.embed_audio(audio_out)
            audio_embeddings = audio_embeddings.to(x.device, x.dtype)
            # audio_pad_mask: True for *padded* positions → invert for valid mask
            audio_valid_mask = ~audio_pad_mask
            if audio_num_soft_tokens is not None:
                audio_valid_mask = audio_valid_mask & build_audio_token_mask(
                    audio_num_soft_tokens.to(audio_valid_mask.device),
                    audio_embeddings.shape[1],
                )
            if audio_mask is not None:
                if audio_embeddings.shape[0] != B:
                    if B != 1:
                        raise ValueError(
                            "Multimodal audio batch size must match token batch size unless B == 1."
                        )
                    audio_embeddings, audio_valid_mask = flatten_multimodal_tokens(
                        audio_embeddings,
                        audio_valid_mask,
                    )
                x = merge_multimodal_embeddings(
                    x, audio_embeddings, audio_valid_mask, audio_mask,
                )

        # --- Bidirectional vision (31B, 26B-A4B) ---
        if self.cfg.text.bidirectional_vision and image_mask is not None:
            attention_mask = make_causal_bidirectional_mask(attention_mask, image_mask)

        # --- Per-layer inputs ---
        # embed_tokens already has 0 at placeholder positions, so PLI lookup
        # uses neutral pad embedding there (matches JAX).
        per_layer_inputs = None
        if self.text_decoder.embedder.per_layer_input_dim > 0:
            per_layer_inputs = self.text_decoder.embedder.encode_per_layer_input(x, embed_tokens)

        # --- Run text decoder ---
        logits, new_cache = self.text_decoder(
            x, positions, attention_mask,
            per_layer_inputs=per_layer_inputs,
            cache=cache,
        )

        return logits, new_cache
