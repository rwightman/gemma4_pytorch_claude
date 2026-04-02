"""Shared primitives: RMSNorm, RoPE, GatedMLP, ClippedLinear."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm with optional learnable scale.

    Args:
        dim: feature dimension.
        eps: epsilon for numerical stability.
        with_scale: if True, learns a per-feature scale parameter.
        zero_init: if True, initialise scale to zeros (vision encoder uses this).
        scale_plus_one: if True, effective scale is ``1 + learned_scale``
            (default for text); otherwise raw ``learned_scale`` (Gemma3n style).
    """

    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            with_scale: bool = True,
            zero_init: bool = False,
            scale_plus_one: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        self.scale_plus_one = scale_plus_one
        if with_scale:
            if zero_init:
                init_val = torch.zeros(dim)
            elif scale_plus_one:
                # effective scale = 1 + param; init param to 0 → effective 1
                init_val = torch.zeros(dim)
            else:
                # effective scale = param directly; init to 1 → effective 1
                init_val = torch.ones(dim)
            self.scale = nn.Parameter(init_val)
        else:
            self.register_parameter("scale", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        if self.with_scale:
            s = self.scale.float()
            if self.scale_plus_one:
                x = x * (1.0 + s)
            else:
                x = x * s
        return x.to(dtype)


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
# GatedMLP
# ---------------------------------------------------------------------------

class GatedMLP(nn.Module):
    """Gated feed-forward: ``down(gelu(gate) * up)``.

    ``gate_up_proj`` is a single linear producing ``2 * hidden_dim`` outputs
    that are split into gate and up.
    """

    def __init__(self, features: int, hidden_dim: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(features, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.gelu(gate) * up)


# ---------------------------------------------------------------------------
# ClippedLinear (audio encoder)
# ---------------------------------------------------------------------------

class ClippedLinear(nn.Module):
    """Linear layer with four learnable scalar clip bounds (init ±inf)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Learnable clip bounds (initialised to ±inf so they are no-ops initially)
        self.clip_min_pre = nn.Parameter(torch.tensor(float("-inf")))
        self.clip_max_pre = nn.Parameter(torch.tensor(float("inf")))
        self.clip_min_post = nn.Parameter(torch.tensor(float("-inf")))
        self.clip_max_post = nn.Parameter(torch.tensor(float("inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.clip_min_pre, self.clip_max_pre)
        x = self.linear(x)
        x = torch.clamp(x, self.clip_min_post, self.clip_max_post)
        return x
