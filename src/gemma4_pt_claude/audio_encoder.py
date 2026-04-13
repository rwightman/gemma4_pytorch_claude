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
from .layers import ClippedLinear, RMSNorm
from .module_utils import InitModule, factory_kwargs, resolve_residual_init_std


# ---------------------------------------------------------------------------
# Relative position embedding (TransformerXL style)
# ---------------------------------------------------------------------------

class RelativePositionEmbedding(InitModule):
    """Sinusoidal relative position embedding + learned projection."""

    def __init__(
            self,
            channels: int,
            num_heads: int,
            head_dim: int,
            max_backward: int,
            max_forward: int,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.init_std = 1e-2
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_backward = max_backward
        self.max_forward = max_forward

        self.pos_proj = nn.Linear(channels, num_heads * head_dim, bias=False, **dd)
        self.register_buffer(
            "inv_timescales",
            torch.empty(1, 1, channels // 2, dtype=torch.float32, device=device),
            persistent=False,
        )
        self._init_non_persistent_buffers()

    def _build_inv_timescales(self) -> torch.Tensor:
        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = self.channels // 2
        log_inc = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
        inv_ts = min_timescale * torch.exp(
            torch.arange(num_timescales, device=self.pos_proj.weight.device) * -log_inc
        )
        return inv_ts.float().unsqueeze(0).unsqueeze(0)

    def _init_non_persistent_buffers(self) -> None:
        inv_timescales = self._build_inv_timescales()
        if self.inv_timescales.is_meta or inv_timescales.is_meta:
            self.inv_timescales = inv_timescales
        else:
            with torch.no_grad():
                self.inv_timescales.copy_(inv_timescales)

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

    def _init_weights(self, ctx) -> None:
        nn.init.xavier_uniform_(self.pos_proj.weight)
        if self.pos_proj.bias is not None:
            nn.init.zeros_(self.pos_proj.bias)
        self._init_non_persistent_buffers()

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

class ChunkedLocalAttention(InitModule):
    """Conformer-style chunked local attention with per-dim query scaling."""

    def __init__(
            self,
            cfg: AudioConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.init_std = cfg.init_std
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.chunk_size = cfg.chunk_size
        self.max_past = max(0, cfg.context_left - 1)
        self.max_future = cfg.context_right
        self.context_size = self.chunk_size + self.max_past + self.max_future
        self.attn_logit_cap = cfg.attn_logit_cap
        self.num_layers = cfg.num_layers

        self.rel_pos_emb = RelativePositionEmbedding(
            cfg.hidden_size, cfg.num_heads, self.head_dim,
            self.max_past, self.max_future,
            **dd,
        )
        self.per_dim_scale = nn.Parameter(torch.ones(self.head_dim, **dd))

        self.q_proj = ClippedLinear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False, init_std=self.init_std, **dd)
        self.k_proj = ClippedLinear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False, init_std=self.init_std, **dd)
        self.v_proj = ClippedLinear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False, init_std=self.init_std, **dd)

        self.q_scale_base = self._build_q_scale_base()
        self.key_scale = self._build_key_scale()
        self.register_buffer(
            "local_causal_mask",
            torch.empty(self.chunk_size, self.context_size, dtype=torch.bool, device=device),
            persistent=False,
        )
        self._init_non_persistent_buffers()

    def _build_q_scale_base(self) -> float:
        r_softplus_0 = 1.0 / math.log(2.0)
        return self.head_dim ** -0.5 * r_softplus_0

    def _build_key_scale(self) -> float:
        r_softplus_0 = 1.0 / math.log(2.0)
        return r_softplus_0 * math.log1p(math.e)

    def _build_causal_mask(self) -> torch.Tensor:
        W, C = self.chunk_size, self.context_size
        lower = torch.tril(
            torch.ones(C, W, dtype=torch.bool, device=self.per_dim_scale.device),
            diagonal=0,
        ).T
        upper = torch.tril(
            torch.ones(W, C, dtype=torch.bool, device=self.per_dim_scale.device),
            diagonal=self.max_past + self.max_future,
        )
        return lower & upper

    def _init_non_persistent_buffers(self) -> None:
        local_causal_mask = self._build_causal_mask()
        if self.local_causal_mask.is_meta or local_causal_mask.is_meta:
            self.local_causal_mask = local_causal_mask
        else:
            with torch.no_grad():
                self.local_causal_mask.copy_(local_causal_mask)

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

        # Key scaling (matches JAX LocalDotProductAttention)
        k = k * self.key_scale

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

    def _init_weights(self, ctx) -> None:
        self._init_non_persistent_buffers()
        nn.init.ones_(self.per_dim_scale)


# ---------------------------------------------------------------------------
# SubSampling block (2 conv layers, 4x reduction)
# ---------------------------------------------------------------------------

class SubSamplingBlock(InitModule):
    """Two Conv2d layers with LayerNorm + ReLU, then linear project.

    JAX uses symmetric ``padding=((1,1),(1,1))`` on both convolutions,
    equivalent to PyTorch ``padding=1``.
    """

    def __init__(
            self,
            cfg: AudioConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_layers = cfg.num_layers
        self.init_std = cfg.init_std
        c1, c2 = cfg.sscp_channels
        k1, k2 = cfg.sscp_kernel_sizes
        s1, s2 = cfg.sscp_stride_sizes
        self.stride1 = s1
        self.stride2 = s2

        # JAX: padding=((1,1),(1,1)) — symmetric on both axes
        dd = factory_kwargs(device, dtype)
        self.conv1 = nn.Conv2d(1, c1, kernel_size=k1, stride=s1, padding=1, bias=False, **dd)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=k2, stride=s2, padding=1, bias=False, **dd)

        # Calculate output freq dims for norm + linear sizing
        # With padding=1, kernel=3, stride=2: out = (in + 2*1 - 3)//2 + 1 = (in-1)//2 + 1
        f_in = cfg.input_feat_size
        f1 = (f_in + 2 - k1[1]) // s1[1] + 1
        f2 = (f1 + 2 - k2[1]) // s2[1] + 1

        # JAX uses nn.LayerNorm(use_bias=False, use_scale=True) on NHWC data.
        # Our Conv2d outputs NCHW, so we permute to NHWC for norm, then back.
        self.norm1 = nn.LayerNorm(c1, bias=False, **dd)
        self.norm2 = nn.LayerNorm(c2, bias=False, **dd)

        self.proj = nn.Linear(c2 * f2, cfg.hidden_size, bias=False, **dd)

    def _init_weights(self, ctx) -> None:
        nn.init.normal_(self.conv1.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        if self.norm1.weight is not None:
            nn.init.ones_(self.norm1.weight)
        if self.norm1.bias is not None:
            nn.init.zeros_(self.norm1.bias)
        if self.norm2.weight is not None:
            nn.init.ones_(self.norm2.weight)
        if self.norm2.bias is not None:
            nn.init.zeros_(self.norm2.bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def _subsample_mask(
            self, mask: torch.Tensor, stride: tuple[int, int], T_out: int,
    ) -> torch.Tensor:
        """Subsample a ``[B, T]`` bool mask along the time axis."""
        return mask[:, :: stride[0]][:, :T_out]

    def forward(
            self, x: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``[B, T, F]`` — mel spectrogram frames
            mask: ``[B, T]`` bool — True for *padded* frames
        Returns:
            ``(hidden_states [B, T', D], mask [B, T'])``
        """
        x = x.unsqueeze(1)  # [B, 1, T, F]

        x = self.conv1(x)  # [B, C1, T1, F1]
        # LayerNorm on channels: permute NCHW→NHWC, norm, permute back
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.relu(x)
        mask = self._subsample_mask(mask, self.stride1, x.shape[2])

        x = self.conv2(x)  # [B, C2, T2, F2]
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.relu(x)
        mask = self._subsample_mask(mask, self.stride2, x.shape[2])

        B, C, T, Freq = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, T, Freq * C)
        return self.proj(x), mask


# ---------------------------------------------------------------------------
# Conformer feed-forward block
# ---------------------------------------------------------------------------

class FFNBlock(InitModule):
    """Conformer FFN: clip -> norm -> linear(4x) -> swish -> linear -> clip -> norm -> residual * 0.5"""

    def __init__(
            self,
            cfg: AudioConfig,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.init_std = cfg.init_std
        self.residual_init_std = cfg.init_std if residual_init_std is None else residual_init_std
        self.gradient_clip = cfg.gradient_clipping
        dd = factory_kwargs(device, dtype)
        self.pre_norm = RMSNorm(cfg.hidden_size, **dd)
        self.up = ClippedLinear(cfg.hidden_size, cfg.hidden_size * 4, bias=False, init_std=self.init_std, **dd)
        self.down = ClippedLinear(
            cfg.hidden_size * 4,
            cfg.hidden_size,
            bias=False,
            init_std=self.init_std,
            residual_init_std=self.residual_init_std,
            **dd,
        )
        self.post_norm = RMSNorm(cfg.hidden_size, **dd)
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

    def _init_weights(self, ctx) -> None:
        if self.pre_norm.weight is not None:
            nn.init.ones_(self.pre_norm.weight)
        if self.post_norm.weight is not None:
            nn.init.ones_(self.post_norm.weight)


# ---------------------------------------------------------------------------
# Lightweight conv block
# ---------------------------------------------------------------------------

class LightweightConvBlock(InitModule):
    """RMSNorm -> linear(2x) -> GLU -> depthwise causal conv1d -> norm -> swish -> linear -> residual."""

    def __init__(
            self,
            cfg: AudioConfig,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.init_std = cfg.init_std
        self.residual_init_std = cfg.init_std if residual_init_std is None else residual_init_std
        self.gradient_clip = cfg.gradient_clipping
        dd = factory_kwargs(device, dtype)
        self.pre_norm = RMSNorm(cfg.hidden_size, **dd)
        self.linear_start = ClippedLinear(cfg.hidden_size, cfg.hidden_size * 2, bias=False, init_std=self.init_std, **dd)
        self.dwconv = nn.Conv1d(
            cfg.hidden_size, cfg.hidden_size,
            kernel_size=cfg.conv_kernel_size,
            stride=1, padding=0,
            groups=cfg.hidden_size, bias=False,
            **dd,
        )
        self.causal_pad = cfg.conv_kernel_size - 1
        self.conv_norm = RMSNorm(cfg.hidden_size, **dd)
        self.linear_end = ClippedLinear(
            cfg.hidden_size,
            cfg.hidden_size,
            bias=False,
            init_std=self.init_std,
            residual_init_std=self.residual_init_std,
            **dd,
        )

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

    def _init_weights(self, ctx) -> None:
        if self.pre_norm.weight is not None:
            nn.init.ones_(self.pre_norm.weight)
        nn.init.normal_(self.dwconv.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)
        if self.conv_norm.weight is not None:
            nn.init.ones_(self.conv_norm.weight)


# ---------------------------------------------------------------------------
# Conformer attention wrapper
# ---------------------------------------------------------------------------

class ConformerAttention(InitModule):
    """Wraps chunked local attention with pre/post norms and O-projection."""

    def __init__(
            self,
            cfg: AudioConfig,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.init_std = cfg.init_std
        self.residual_init_std = cfg.init_std if residual_init_std is None else residual_init_std
        self.gradient_clip = cfg.gradient_clipping
        dd = factory_kwargs(device, dtype)
        self.pre_norm = RMSNorm(cfg.hidden_size, **dd)
        self.attn = ChunkedLocalAttention(cfg, **dd)
        self.o_proj = ClippedLinear(
            cfg.hidden_size,
            cfg.hidden_size,
            bias=False,
            init_std=self.init_std,
            residual_init_std=self.residual_init_std,
            **dd,
        )
        self.post_norm = RMSNorm(cfg.hidden_size, **dd)

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

    def _init_weights(self, ctx) -> None:
        if self.pre_norm.weight is not None:
            nn.init.ones_(self.pre_norm.weight)
        if self.post_norm.weight is not None:
            nn.init.ones_(self.post_norm.weight)


# ---------------------------------------------------------------------------
# ConformerLayer
# ---------------------------------------------------------------------------

class ConformerLayer(InitModule):
    """FFN(0.5) -> Attn -> mask -> LConv -> FFN(0.5) -> clip -> RMSNorm."""

    def __init__(
            self,
            cfg: AudioConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dd = factory_kwargs(device, dtype)
        residual_init_std = resolve_residual_init_std(
            cfg.init_std,
            cfg.residual_init_std,
            cfg.use_depth_scaled_residual_init,
            cfg.num_layers,
        )
        self.ffw_start = FFNBlock(
            cfg,
            residual_init_std=residual_init_std,
            **dd,
        )
        self.attn = ConformerAttention(
            cfg,
            residual_init_std=residual_init_std,
            **dd,
        )
        self.lconv = LightweightConvBlock(
            cfg,
            residual_init_std=residual_init_std,
            **dd,
        )
        self.ffw_end = FFNBlock(
            cfg,
            residual_init_std=residual_init_std,
            **dd,
        )
        self.gradient_clip = cfg.gradient_clipping
        self.norm = RMSNorm(cfg.hidden_size, **dd)

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

    def _init_weights(self, ctx) -> None:
        if self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)


# ---------------------------------------------------------------------------
# AudioEncoder (standalone)
# ---------------------------------------------------------------------------

class AudioEncoder(InitModule):
    """Full audio encoder: mel -> subsample -> conformer -> output projection.

    Projection from ``lm_model_dims`` to text embedding space lives in
    the model-level ``embed_audio`` (MultimodalEmbedder), not here.
    """

    def __init__(
            self,
            cfg: AudioConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.cfg = cfg
        self.init_std = cfg.init_std
        self.subsample = SubSamplingBlock(cfg, **dd)
        self.conformer = nn.ModuleList([ConformerLayer(cfg, **dd) for _ in range(cfg.num_layers)])
        self.output_proj = nn.Linear(cfg.hidden_size, cfg.lm_model_dims, bias=True, **dd)

    def forward(
            self, mel: torch.Tensor, mel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: ``[B, T, F]`` — mel spectrogram frames
            mel_mask: ``[B, T]`` bool — True for padded frames

        Returns:
            ``(embeddings [B, T', lm_model_dims], mask [B, T'])``
        """
        x, mel_mask = self.subsample(mel, mel_mask)  # [B, T', D]

        for layer in self.conformer:
            x = layer(x, mel_mask)

        x = self.output_proj(x)

        # Zero-out padded positions
        valid = (~mel_mask).unsqueeze(-1).to(x.dtype)
        x = x * valid

        return x, mel_mask

    def _init_weights(self, ctx) -> None:
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
