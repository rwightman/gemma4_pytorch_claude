"""Standalone audio encoder (USM/Conformer-based).

MelFilterbank -> SubSampling -> Conformer layers -> projection to text space.
Owns its own projection, so it can be instantiated independently.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AudioConfig
from .layers import RMSNorm


# ---------------------------------------------------------------------------
# Relative position embedding (TransformerXL style)
# ---------------------------------------------------------------------------

class RelativePositionEmbedding(nn.Module):
    """Sinusoidal relative position embedding + learned projection."""

    def __init__(self, channels: int, num_heads: int, head_dim: int, max_backward: int, max_forward: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_backward = max_backward
        self.max_forward = max_forward

        self.pos_proj = nn.Linear(channels, num_heads * head_dim, bias=False)

        # Sinusoidal timescales
        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = channels // 2
        log_inc = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
        inv_ts = min_timescale * torch.exp(torch.arange(num_timescales) * -log_inc)
        self.register_buffer("inv_timescales", inv_ts.float().unsqueeze(0).unsqueeze(0), persistent=False)

    def _timing_signal(self, positions: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        pos = positions.float().unsqueeze(-1)
        scaled = pos * self.inv_timescales.to(pos.device)
        return torch.cat([scaled.sin(), scaled.cos()], dim=-1).to(dtype)

    def _relative_shift(self, bd: torch.Tensor, key_ctx: int) -> torch.Tensor:
        B, N, U, W, F_span = bd.shape
        pad = key_ctx + 1 - F_span
        bd = F.pad(bd, (0, pad))  # [B, N, U, W, key_ctx+1]
        bd = bd.reshape(B, N, U, W * (key_ctx + 1))
        bd = bd[:, :, :, : W * key_ctx]
        return bd.reshape(B, N, U, W, key_ctx)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: ``[B, U, W, N, H]``
            keys: ``[B, U, C, N, H]``
        Returns:
            logits: ``[B, N, U, W, C]``
        """
        B, U, W, N, H = queries.shape
        C = keys.shape[2]

        # Relative positions: [max_backward, ..., -max_forward]
        pos = torch.arange(self.max_backward, -self.max_forward - 1, -1, device=queries.device).unsqueeze(0)
        F_span = pos.shape[1]

        sin_emb = self._timing_signal(pos, queries.dtype)  # [1, F_span, channels]
        sin_emb = self.pos_proj(sin_emb).view(1, F_span, N, H).squeeze(0)  # [F_span, N, H]

        # Content: Q·K^T -> [B, N, U, W, C]
        q_p = queries.permute(0, 3, 1, 2, 4)  # [B, N, U, W, H]
        k_p = keys.permute(0, 3, 1, 4, 2)     # [B, N, U, H, C]
        term_ac = torch.matmul(q_p, k_p)

        # Position: Q·pos^T -> [B, N, U, W, F_span]
        s_p = sin_emb.permute(1, 2, 0)  # [N, H, F_span]
        q_flat = q_p.reshape(B, N, U * W, H)
        term_bd = torch.matmul(q_flat, s_p).reshape(B, N, U, W, F_span)
        term_bd = self._relative_shift(term_bd, C)

        return term_ac + term_bd


# ---------------------------------------------------------------------------
# Chunked local attention
# ---------------------------------------------------------------------------

class ChunkedLocalAttention(nn.Module):
    """Conformer-style chunked local attention with per-dim query scaling."""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.chunk_size = cfg.chunk_size
        self.max_past = max(0, cfg.context_left - 1)
        self.max_future = cfg.context_right
        self.context_size = self.chunk_size + self.max_past + self.max_future
        self.attn_logit_cap = cfg.attn_logit_cap

        self.rel_pos_emb = RelativePositionEmbedding(
            cfg.hidden_size, cfg.num_heads, self.head_dim,
            self.max_past, self.max_future,
        )
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False)

        # Query scale: rsoftplus(0)/sqrt(H) * softplus(per_dim_scale)
        r_softplus_0 = 1.0 / F.softplus(torch.tensor(0.0)).item()
        self.register_buffer("q_scale_base", torch.tensor(self.head_dim ** -0.5 * r_softplus_0), persistent=False)

        # Local causal mask: [W, C]
        self.register_buffer("local_causal_mask", self._build_causal_mask(), persistent=False)

    def _build_causal_mask(self) -> torch.Tensor:
        W, C = self.chunk_size, self.context_size
        lower = torch.tril(torch.ones(C, W, dtype=torch.bool), diagonal=0).T
        upper = torch.tril(torch.ones(W, C, dtype=torch.bool), diagonal=self.max_past + self.max_future)
        return lower & upper

    def _to_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, ...]`` -> ``[B, U, W, ...]`` with padding."""
        B, T = x.shape[:2]
        tail = x.shape[2:]
        U = (T + self.chunk_size - 1) // self.chunk_size
        pad_len = U * self.chunk_size - T
        if pad_len > 0:
            # Pad time dim (dim=1). F.pad pads last dims first, so we need
            # zeros for all tail dims then time dim padding.
            x = F.pad(x, [0] * (2 * len(tail)) + [0, pad_len])
        return x.view(B, U, self.chunk_size, *tail)

    def _pad_time(self, x: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
        """Pad dimension 1 (time) of tensor ``[B, T, ...]``."""
        B, T = x.shape[:2]
        tail = x.shape[2:]
        left = x.new_zeros(B, pad_left, *tail)
        right = x.new_zeros(B, pad_right, *tail)
        return torch.cat([left, x, right], dim=1)

    def _extract_context(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, ...]`` -> ``[B, U, C, ...]`` sliding-window context.

        First pads T to a multiple of chunk_size, then adds left/right
        context padding, then extracts overlapping windows via unfold.
        """
        B, T = x.shape[:2]
        tail = x.shape[2:]

        # First pad T to a multiple of chunk_size (same as _to_blocks)
        U = (T + self.chunk_size - 1) // self.chunk_size
        pad_to_multiple = U * self.chunk_size - T
        if pad_to_multiple > 0:
            x = self._pad_time(x, 0, pad_to_multiple)

        # Then add context padding
        pad_left = self.max_past
        pad_right = self.max_future + self.chunk_size - 1
        x = self._pad_time(x, pad_left, pad_right)

        # Unfold along time dim (dim=1)
        x_unf = x.unfold(1, self.context_size, self.chunk_size)
        # For input [B, T_padded, N, H]: x_unf is [B, U, N, H, C]
        # Move C to position 2: [B, U, C, N, H]
        if len(tail) > 0:
            x_unf = torch.movedim(x_unf, -1, 2)
        return x_unf.contiguous()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, D]``
            mask: ``[B, T]`` bool — True for *padded* positions
        Returns:
            ``[B, T, N, H]``
        """
        B, T, D = x.shape
        shape = (*x.shape[:-1], self.num_heads, self.head_dim)

        q = self.q_proj(x).view(shape)
        k = self.k_proj(x).view(shape)
        v = self.v_proj(x).view(shape)

        # Per-dim query scaling
        pds = F.softplus(self.per_dim_scale)
        q = q * self.q_scale_base * pds

        q_blocks = self._to_blocks(q)      # [B, U, W, N, H]
        k_ctx = self._extract_context(k)    # [B, U, C, N, H]
        v_ctx = self._extract_context(v)

        # Validity mask (True = valid)
        valid = ~mask  # [B, T]
        valid_ctx = self._extract_context(valid)  # [B, U, C]

        # Logits with relative position
        logits = self.rel_pos_emb(q_blocks, k_ctx)  # [B, N, U, W, C]

        # Softcap
        logits = torch.tanh(logits / self.attn_logit_cap) * self.attn_logit_cap

        # Combined mask: validity & causal
        # valid_ctx: [B, U, C] -> [B, 1, U, 1, C]
        cond_valid = valid_ctx.unsqueeze(1).unsqueeze(-2)
        cond_causal = self.local_causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        combined = cond_valid & cond_causal.to(cond_valid.device)

        logits = torch.where(combined, logits, torch.finfo(logits.dtype).min)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(v.dtype)

        # Weighted sum: [B, N, U, W, C] x [B, U, C, N, H] -> [B, U, W, N, H]
        b, n, u, w, c = probs.shape
        h = v_ctx.shape[-1]
        prob_flat = probs.permute(0, 2, 1, 3, 4).reshape(-1, w, c)
        v_flat = v_ctx.permute(0, 1, 3, 2, 4).reshape(-1, c, h)
        ctx = torch.bmm(prob_flat, v_flat).reshape(b, u, n, w, h).permute(0, 1, 3, 2, 4)
        # [B, U, W, N, H] -> [B, U*W, N, H] -> trim to [B, T, N, H]
        ctx = ctx.reshape(B, -1, self.num_heads, self.head_dim)[:, :T]
        return ctx


# ---------------------------------------------------------------------------
# SubSampling block (2 conv layers, 4x reduction)
# ---------------------------------------------------------------------------

class SubSamplingBlock(nn.Module):
    """Two Conv2d layers with cumulative group norm + ReLU, then linear project."""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        c1, c2 = cfg.sscp_channels
        k1, k2 = cfg.sscp_kernel_sizes
        s1, s2 = cfg.sscp_stride_sizes

        self.conv1 = nn.Conv2d(1, c1, kernel_size=k1, stride=s1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=k2, stride=s2, padding=0, bias=False)

        # Calculate output freq dims for norm + linear sizing
        f_in = cfg.input_feat_size
        # Padding: (pad_f_left, pad_f_right, pad_t_top, pad_t_bottom)
        self.pad1 = (1, 1, 0, k1[0] - 1)
        f1 = (f_in + 2 - k1[1]) // s1[1] + 1
        self.pad2 = (1, 1, 0, k2[0] - 1)
        f2 = (f1 + 2 - k2[1]) // s2[1] + 1

        self.norm1 = nn.GroupNorm(1, c1)
        self.norm2 = nn.GroupNorm(1, c2)

        self.proj = nn.Linear(c2 * f2, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, F]`` — mel spectrogram frames
        Returns:
            ``[B, T', D]`` — subsampled hidden states
        """
        x = x.unsqueeze(1)  # [B, 1, T, F]

        x = F.pad(x, self.pad1)
        x = F.relu(self.norm1(self.conv1(x)))

        x = F.pad(x, self.pad2)
        x = F.relu(self.norm2(self.conv2(x)))

        B, C, T, Freq = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, T, Freq * C)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Conformer feed-forward block
# ---------------------------------------------------------------------------

class FFNBlock(nn.Module):
    """Conformer FFN: clip -> norm -> linear(4x) -> swish -> linear -> clip -> norm -> residual * 0.5"""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.gradient_clip = cfg.gradient_clipping
        self.pre_norm = RMSNorm(cfg.hidden_size)
        self.up = nn.Linear(cfg.hidden_size, cfg.hidden_size * 4, bias=False)
        self.down = nn.Linear(cfg.hidden_size * 4, cfg.hidden_size, bias=False)
        self.post_norm = RMSNorm(cfg.hidden_size)
        self.residual_weight = cfg.residual_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.clamp(x, -self.gradient_clip, self.gradient_clip)
        x = self.pre_norm(x)
        x = F.silu(self.up(x))
        x = self.down(x)
        x = torch.clamp(x, -self.gradient_clip, self.gradient_clip)
        x = self.post_norm(x)
        return residual + x * self.residual_weight


# ---------------------------------------------------------------------------
# Lightweight conv block
# ---------------------------------------------------------------------------

class LightweightConvBlock(nn.Module):
    """RMSNorm -> linear(2x) -> GLU -> depthwise causal conv1d -> norm -> swish -> linear -> residual."""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.gradient_clip = cfg.gradient_clipping
        self.pre_norm = RMSNorm(cfg.hidden_size)
        self.linear_start = nn.Linear(cfg.hidden_size, cfg.hidden_size * 2, bias=False)
        self.dwconv = nn.Conv1d(
            cfg.hidden_size, cfg.hidden_size,
            kernel_size=cfg.conv_kernel_size,
            stride=1, padding=0,
            groups=cfg.hidden_size, bias=False,
        )
        self.causal_pad = cfg.conv_kernel_size - 1
        self.conv_norm = RMSNorm(cfg.hidden_size)
        self.linear_end = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        x = F.glu(self.linear_start(x), dim=-1)
        # Conv1d: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        x = F.pad(x, (self.causal_pad, 0))
        x = self.dwconv(x).transpose(1, 2)  # back to [B, T, D]
        x = torch.clamp(x, -self.gradient_clip, self.gradient_clip)
        x = self.conv_norm(x)
        x = F.silu(x)
        x = self.linear_end(x)
        return x + residual


# ---------------------------------------------------------------------------
# Conformer attention wrapper
# ---------------------------------------------------------------------------

class ConformerAttention(nn.Module):
    """Wraps chunked local attention with pre/post norms and O-projection."""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.gradient_clip = cfg.gradient_clipping
        self.pre_norm = RMSNorm(cfg.hidden_size)
        self.attn = ChunkedLocalAttention(cfg)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.post_norm = RMSNorm(cfg.hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.clamp(x, -self.gradient_clip, self.gradient_clip)
        x = self.pre_norm(x)
        ctx = self.attn(x, mask)  # [B, T, N, H]
        B, T, N, H = ctx.shape
        ctx = ctx.reshape(B, T, N * H)
        ctx = self.o_proj(ctx)
        ctx = torch.clamp(ctx, -self.gradient_clip, self.gradient_clip)
        return residual + self.post_norm(ctx)


# ---------------------------------------------------------------------------
# ConformerLayer
# ---------------------------------------------------------------------------

class ConformerLayer(nn.Module):
    """FFN(0.5) -> Attn -> mask -> LConv -> FFN(0.5) -> clip -> RMSNorm."""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.ffw_start = FFNBlock(cfg)
        self.attn = ConformerAttention(cfg)
        self.lconv = LightweightConvBlock(cfg)
        self.ffw_end = FFNBlock(cfg)
        self.gradient_clip = cfg.gradient_clipping
        self.norm = RMSNorm(cfg.hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.ffw_start(x)
        x = self.attn(x, mask)
        # Zero-out padded positions before conv
        valid = (~mask).unsqueeze(-1).to(x.dtype)
        x = x * valid
        x = self.lconv(x)
        x = self.ffw_end(x)
        x = torch.clamp(x, -self.gradient_clip, self.gradient_clip)
        return self.norm(x)


# ---------------------------------------------------------------------------
# AudioEncoder (standalone)
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """Full audio encoder: mel -> subsample -> conformer -> project to text space.

    Owns its projection to text embedding space, so it can be used independently.
    """

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample = SubSamplingBlock(cfg)
        self.conformer = nn.ModuleList([ConformerLayer(cfg) for _ in range(cfg.num_layers)])
        self.output_proj = nn.Linear(cfg.hidden_size, cfg.lm_model_dims, bias=False)
        self.audio_proj = nn.Linear(cfg.lm_model_dims, cfg.text_embed_dim, bias=False)
        self.audio_norm = RMSNorm(cfg.text_embed_dim, with_scale=False)

    def forward(
        self, mel: torch.Tensor, mel_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            mel: ``[B, T, F]`` — mel spectrogram frames
            mel_mask: ``[B, T]`` bool — True for padded frames

        Returns:
            ``[B, T', text_embed_dim]``
        """
        x = self.subsample(mel)  # [B, T', D]

        # Subsample the mask to match
        T_sub = x.shape[1]
        if mel_mask.shape[1] != T_sub:
            # Simple subsampling: take every reduction_factor-th mask value
            factor = mel_mask.shape[1] // T_sub if T_sub > 0 else 1
            mel_mask = mel_mask[:, ::factor][:, :T_sub]

        for layer in self.conformer:
            x = layer(x, mel_mask)

        x = self.output_proj(x)
        x = self.audio_proj(x)
        x = self.audio_norm(x)
        return x
