"""Shared primitives: RMSNorm, RoPE, TanhGELU, GatedMLP, ClippedLinear."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_utils import InitModule, factory_kwargs


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.RMSNorm):
    """Builtin-backed RMSNorm with optional zero-init weight."""

    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            with_scale: bool = True,
            zero_init: bool = False,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        self.zero_init = zero_init
        super().__init__(
            dim,
            eps=eps,
            elementwise_affine=with_scale,
            **factory_kwargs(device, dtype),
        )

    def reset_parameters(self) -> None:
        if self.weight is not None:
            if self.zero_init:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)


class VisionRMSNorm(RMSNorm):
    """Vision RMSNorm with zero-initialized affine weight."""

    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__(
            dim,
            eps=eps,
            with_scale=True,
            zero_init=True,
            device=device,
            dtype=dtype,
        )


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def apply_rope(
        x: torch.Tensor,
        positions: torch.Tensor,
        base_frequency: int = 10_000,
        scale_factor: float = 1.0,
        rope_proportion: float = 1.0,
) -> torch.Tensor:
    """Apply 1-D Rotary Position Embedding.

    Args:
        x: ``[B, L, N, H]`` — queries or keys.
        positions: ``[B, L]`` — absolute positions.
        base_frequency: RoPE base.
        scale_factor: positional interpolation factor (>= 1).
        rope_proportion: fraction of head_dim to rotate. The rest gets
            an infinite timescale (i.e. no rotation).
    """
    head_dim = x.shape[-1]
    rope_angles = int(rope_proportion * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles

    freq_exponents = (2.0 / head_dim) * torch.arange(
        rope_angles, dtype=torch.float32, device=x.device
    )
    timescale = base_frequency ** freq_exponents
    if nope_angles > 0:
        timescale = F.pad(timescale, (0, nope_angles), value=float("inf"))

    # sinusoid_inp: [B, L, H//2]
    sinusoid_inp = positions[..., None].float() / timescale[None, None, :]
    sinusoid_inp = sinusoid_inp.unsqueeze(-2)  # [B, L, 1, H//2]
    sinusoid_inp = sinusoid_inp / scale_factor

    sin = sinusoid_inp.sin()
    cos = sinusoid_inp.cos()

    first_half, second_half = x.float().chunk(2, dim=-1)
    out = torch.cat(
        [first_half * cos - second_half * sin,
         second_half * cos + first_half * sin],
        dim=-1,
    )
    return out.to(x.dtype)


def apply_multidimensional_rope(
        x: torch.Tensor,
        positions_h: torch.Tensor,
        positions_w: torch.Tensor,
        base_frequency: float = 100.0,
) -> torch.Tensor:
    """Apply 2-D factorised RoPE for vision (height, width).

    Args:
        x: ``[B, L, N, H]``
        positions_h, positions_w: ``[B, L]`` — row/col positions.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    # Split head_dim into two halves: one for height, one for width.
    x_h, x_w = x[..., :half], x[..., half:]
    x_h = apply_rope(x_h, positions_h, base_frequency=int(base_frequency))
    x_w = apply_rope(x_w, positions_w, base_frequency=int(base_frequency))
    return torch.cat([x_h, x_w], dim=-1)


# ---------------------------------------------------------------------------
# TanhGELU
# ---------------------------------------------------------------------------


class TanhGELU(InitModule):
    """Gemma-style GELU using the tanh approximation."""

    def __init__(
            self,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _ = device, dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


# ---------------------------------------------------------------------------
# GatedMLP
# ---------------------------------------------------------------------------


class GatedMLP(InitModule):
    """Gated feed-forward: ``down(gelu(gate) * up)``.

    ``gate_up_proj`` is a single linear producing ``2 * hidden_dim`` outputs
    that are split into gate and up.
    """

    def __init__(
            self,
            features: int,
            hidden_dim: int,
            init_std: float = 1e-2,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.init_std = init_std
        self.residual_init_std = init_std if residual_init_std is None else residual_init_std
        self.gate_up_proj = nn.Linear(features, 2 * hidden_dim, bias=False, **factory_kwargs(device, dtype))
        self.act = TanhGELU()
        self.down_proj = nn.Linear(hidden_dim, features, bias=False, **factory_kwargs(device, dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act(gate) * up)

    def _init_weights(self, ctx) -> None:
        nn.init.normal_(self.gate_up_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.gate_up_proj.bias is not None:
            nn.init.zeros_(self.gate_up_proj.bias)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=self.residual_init_std, generator=ctx.generator)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)


# ---------------------------------------------------------------------------
# ClippedLinear (audio encoder)
# ---------------------------------------------------------------------------

class ClippedLinear(InitModule):
    """Linear layer with four scalar clip bounds (loaded as buffers, init ±inf)."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            init_std: float = 1e-2,
            residual_init_std: float | None = None,
            bias: bool = False,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.init_std = init_std
        self.residual_init_std = residual_init_std if residual_init_std is not None else init_std
        dd = factory_kwargs(device, dtype)
        self.linear = nn.Linear(in_features, out_features, bias=bias, **dd)
        self.register_buffer("input_min", torch.tensor(float("-inf"), **dd))
        self.register_buffer("input_max", torch.tensor(float("inf"), **dd))
        self.register_buffer("output_min", torch.tensor(float("-inf"), **dd))
        self.register_buffer("output_max", torch.tensor(float("inf"), **dd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x)
        x = torch.clamp(x, self.output_min, self.output_max)
        return x

    def _init_weights(self, ctx) -> None:
        nn.init.normal_(self.linear.weight, mean=0.0, std=self.residual_init_std, generator=ctx.generator)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
