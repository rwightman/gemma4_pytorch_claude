"""Encoder-free multimodal embedders (12B, HF ``gemma4_unified``).

These variants drop the vision tower and the conformer audio encoder entirely.
Images arrive as pre-merged raw pixel patches and audio as raw waveform frames;
both are projected straight into text space.

Vision: ``patches -> LN -> Linear -> LN -> +2-D posemb -> LN -> RMSNorm -> Linear``
Audio:  ``frames  -> RMSNorm -> Linear``
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import EncoderFreeAudioConfig, EncoderFreeVisionConfig
from .layers import RMSNorm
from .module_utils import InitModule, factory_kwargs


# ---------------------------------------------------------------------------
# Shared projection into text space
# ---------------------------------------------------------------------------

class MultimodalProjection(InitModule):
    """RMSNorm (no scale) followed by a linear projection into text space.

    Shared by the vision and audio paths, matching the reference where both
    modalities end in the same projection block.
    """

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
        return self.proj(self.norm(x.to(self.proj.weight.dtype)))

    def _init_weights(self, ctx) -> None:
        nn.init.normal_(self.proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

class EncoderFreeVisionEmbedder(InitModule):
    """Project merged raw pixel patches into text space, with no vision tower."""

    def __init__(
            self,
            cfg: EncoderFreeVisionConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.cfg = cfg
        self.init_std = cfg.init_std
        self.position_init_std = cfg.position_init_std

        self.patch_ln1 = nn.LayerNorm(cfg.patch_dim, **dd)
        self.patch_dense = nn.Linear(cfg.patch_dim, cfg.mm_embed_dim, **dd)
        self.patch_ln2 = nn.LayerNorm(cfg.mm_embed_dim, **dd)

        # Factorised 2-D position embedding: [max_positions, 2 (x/y), width]
        self.pos_embedding = nn.Parameter(
            torch.zeros(cfg.position_embedding_size, 2, cfg.mm_embed_dim, **dd),
        )
        self.pos_norm = nn.LayerNorm(cfg.mm_embed_dim, **dd)

        self.multimodal_projection = MultimodalProjection(
            cfg.output_proj_dims,
            cfg.text_embed_dim,
            init_std=cfg.init_std,
            eps=cfg.rms_norm_eps,
            **dd,
        )

    def forward(
            self,
            pixel_values: torch.Tensor,
            position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: ``[B, L, patch_dim]`` — merged raw pixel patches in [0, 1].
            position_ids: ``[B, L, 2]`` — (x, y) coords, -1 for padding.

        Returns:
            ``(embeddings [B, L, text_dim], valid_mask [B, L])``
        """
        hidden = self.patch_ln1(pixel_values.to(self.patch_dense.weight.dtype))
        hidden = self.patch_dense(hidden)
        hidden = self.patch_ln2(hidden)

        # Gather the x and y embeddings and sum them; padding contributes zero.
        clamped = position_ids.clamp(min=0).long()
        valid = (position_ids != -1).to(self.pos_embedding.dtype).unsqueeze(-1)
        axes = torch.arange(2, device=position_ids.device)
        pos_emb = (self.pos_embedding[clamped, axes] * valid).sum(dim=-2)

        hidden = self.pos_norm(hidden + pos_emb)
        hidden = self.multimodal_projection(hidden)

        valid_mask = (position_ids != -1).all(dim=-1)
        return hidden, valid_mask

    def _init_weights(self, ctx) -> None:
        for norm in (self.patch_ln1, self.patch_ln2, self.pos_norm):
            if norm.weight is not None:
                nn.init.ones_(norm.weight)
            if norm.bias is not None:
                nn.init.zeros_(norm.bias)
        nn.init.normal_(self.patch_dense.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.patch_dense.bias is not None:
            nn.init.zeros_(self.patch_dense.bias)
        nn.init.normal_(
            self.pos_embedding, mean=0.0, std=self.position_init_std, generator=ctx.generator,
        )


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

class EncoderFreeAudioEmbedder(InitModule):
    """Project raw waveform frames into text space, with no audio encoder."""

    def __init__(
            self,
            cfg: EncoderFreeAudioConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.multimodal_projection = MultimodalProjection(
            cfg.output_proj_dims,
            cfg.text_embed_dim,
            init_std=cfg.init_std,
            eps=cfg.rms_norm_eps,
            **factory_kwargs(device, dtype),
        )

    def forward(
            self,
            audio_frames: torch.Tensor,
            frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_frames: ``[B, T, audio_samples_per_token]`` — raw waveform frames.
            frame_mask: ``[B, T]`` bool — True for valid frames (all valid if None).

        Returns:
            ``(embeddings [B, T, text_dim], valid_mask [B, T])``
        """
        embeddings = self.multimodal_projection(audio_frames)
        if frame_mask is None:
            frame_mask = torch.ones(
                audio_frames.shape[:2], dtype=torch.bool, device=audio_frames.device,
            )
        return embeddings, frame_mask
