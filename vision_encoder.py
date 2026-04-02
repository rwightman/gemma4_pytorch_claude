"""Standalone vision encoder (SigLiP-style) with 2-D RoPE and spatial pooling.

Owns its own projection to text embedding space, so it can be instantiated
and used independently of the text decoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VisionConfig
from .layers import GatedMLP, RMSNorm, apply_multidimensional_rope


# ---------------------------------------------------------------------------
# Patch embedding + factorised 2-D positional embedding
# ---------------------------------------------------------------------------

class VisionPatchEmbed(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        patch_dim = cfg.patch_size ** 2 * 3
        self.proj = nn.Linear(patch_dim, cfg.d_model, bias=True)

        num_patches_side = cfg.image_size // cfg.patch_size
        num_patches = num_patches_side ** 2
        # Learnable position embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, cfg.d_model) * (1.0 / math.sqrt(cfg.d_model))
        )
        self.num_patches_side = num_patches_side

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: ``[B, num_patches, patch_dim]`` — already-patchified pixels.

        Returns:
            ``[B, num_patches, d_model]``
        """
        x = self.proj(patches)
        x = x + self.pos_embedding[:, : x.shape[1]]
        return x


# ---------------------------------------------------------------------------
# Vision attention block with 2-D RoPE + QK/V-norm
# ---------------------------------------------------------------------------

class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        self.d_model = cfg.d_model

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.kv_proj = nn.Linear(cfg.d_model, cfg.d_model * 2, bias=True)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

        # QK-norm with zero-init scale (vision style)
        self.q_norm = RMSNorm(self.head_dim, zero_init=True)
        self.k_norm = RMSNorm(self.head_dim, zero_init=True)
        # V-norm without scale
        self.v_norm = RMSNorm(self.head_dim, with_scale=False)

        self.scale = 1.0  # QK-norm replaces 1/sqrt(d)
        self.num_patches_side = cfg.image_size // cfg.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = self.num_heads
        H = self.head_dim
        S = self.num_patches_side

        q = self.q_proj(x).view(B, L, N, H)
        kv = self.kv_proj(x).view(B, L, 2, N, H)
        k, v = kv[:, :, 0], kv[:, :, 1]

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # 2-D RoPE: build row/col position vectors
        device = x.device
        rows = torch.arange(S, device=device).unsqueeze(1).expand(S, S).reshape(-1)
        cols = torch.arange(S, device=device).unsqueeze(0).expand(S, S).reshape(-1)
        # Expand to batch: [1, L]
        rows = rows.unsqueeze(0).expand(B, -1)[:, :L]
        cols = cols.unsqueeze(0).expand(B, -1)[:, :L]

        q = apply_multidimensional_rope(q, rows, cols, base_frequency=100.0)
        k = apply_multidimensional_rope(k, rows, cols, base_frequency=100.0)

        q = q * self.scale

        # Standard attention
        q = q.transpose(1, 2)  # [B, N, L, H]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)  # [B, N, L, H]
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out)


class VisionBlock(nn.Module):
    """Single vision transformer block with zero-init norms."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.pre_attn_norm = nn.LayerNorm(cfg.d_model)
        self.attn = VisionAttention(cfg)
        self.pre_ffw_norm = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_dim, bias=True),
            nn.GELU(),
            nn.Linear(cfg.mlp_dim, cfg.d_model, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.pre_attn_norm(x))
        x = x + self.mlp(self.pre_ffw_norm(x))
        return x


# ---------------------------------------------------------------------------
# Spatial pooling (avg-pool to output_length)
# ---------------------------------------------------------------------------

class SpatialPool(nn.Module):
    """Avg-pool spatial tokens from L to output_length, then scale by sqrt(d)."""

    def __init__(self, output_length: int, d_model: int):
        super().__init__()
        self.output_length = output_length
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        if L == self.output_length:
            return x
        side = int(math.sqrt(L))
        assert side * side == L, f"Input length {L} is not a perfect square"
        out_side = int(math.sqrt(self.output_length))
        assert out_side * out_side == self.output_length

        x = x.view(B, side, side, D)
        window = side // out_side
        # Use avg_pool2d
        x = x.permute(0, 3, 1, 2)  # [B, D, H, W]
        x = F.avg_pool2d(x, kernel_size=window, stride=window)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, D)  # [B, output_length, D]
        return x


# ---------------------------------------------------------------------------
# VisionEncoder (standalone)
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """Full vision encoder: patches -> blocks -> pool -> project to text space.

    Owns its projection, so it can be used independently.
    """

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = VisionPatchEmbed(cfg)
        self.blocks = nn.ModuleList([VisionBlock(cfg) for _ in range(cfg.num_layers)])
        self.encoder_norm = nn.LayerNorm(cfg.d_model)
        self.spatial_pool = SpatialPool(cfg.output_length, cfg.d_model)
        # Projection to text embedding dim
        self.pre_proj_norm = RMSNorm(cfg.d_model, with_scale=False)
        self.proj = nn.Linear(cfg.d_model, cfg.text_embed_dim, bias=False)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: ``[B, num_patches, patch_dim]`` or ``[B, N, num_patches, patch_dim]``
                If 4-D, the ``N`` dimension is treated as num_images and is
                flattened with the batch dim.

        Returns:
            ``[B, output_length, text_embed_dim]`` (or ``[B, N, output_length, text_embed_dim]``)
        """
        unflatten = False
        if patches.ndim == 4:
            B, N, P, C = patches.shape
            patches = patches.view(B * N, P, C)
            unflatten = True

        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x)
        x = self.encoder_norm(x)
        x = self.spatial_pool(x)
        x = self.pre_proj_norm(x)
        x = self.proj(x)

        if unflatten:
            x = x.view(B, N, *x.shape[1:])
        return x
