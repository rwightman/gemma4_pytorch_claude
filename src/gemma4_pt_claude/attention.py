"""Text decoder attention with GQA, QK/V-norm, sliding window, and KV cache."""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AttentionType
from .layers import RMSNorm, apply_rope
from .module_utils import InitModule, factory_kwargs

K_MASK = -2.3819763e38


# ---------------------------------------------------------------------------
# KV cache type
# ---------------------------------------------------------------------------

class LayerCache(TypedDict):
    k: torch.Tensor       # [B, cache_len, kv_heads, head_dim]
    v: torch.Tensor       # [B, cache_len, kv_heads, head_dim]
    positions: torch.Tensor  # [B, cache_len]
    end_index: torch.Tensor  # [B]
    valid_mask: torch.Tensor  # [B, cache_len] bool — True for filled slots


# ---------------------------------------------------------------------------
# Sliding-window mask
# ---------------------------------------------------------------------------

def create_sliding_mask(
        positions: torch.Tensor,
        cache_positions: torch.Tensor | None,
        sliding_window_size: int,
) -> torch.Tensor:
    """``[B, L, S]`` bool mask — True where position is within the window."""
    if cache_positions is None:
        cache_positions = positions
    cp = cache_positions[:, None, :]   # [B, 1, S]
    pp = positions[:, :, None]          # [B, L, 1]
    mask = (cp > pp - sliding_window_size) & (cp < pp + sliding_window_size)
    return mask


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(InitModule):
    """Multi-head attention (GQA) with QK-norm, V-norm, RoPE, sliding window.

    When ``k_eq_v`` is True the key and value projections share weights
    (the value projection is dropped and the key projection is used for both).
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_kv_heads: int,
            head_dim: int,
            attn_type: AttentionType,
            init_std: float,
            residual_init_std: float | None = None,
            rope_base: int = 10_000,
            rope_scale_factor: float = 1.0,
            rope_proportion: float = 1.0,
            sliding_window_size: int | None = None,
            attn_logits_soft_cap: float | None = None,
            use_qk_norm: bool = True,
            use_value_norm: bool = False,
            k_eq_v: bool = False,
            attn_impl: str = "sdpa",
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.init_std = init_std
        self.residual_init_std = init_std if residual_init_std is None else residual_init_std
        self.attn_type = attn_type
        self.rope_base = rope_base
        self.rope_scale_factor = rope_scale_factor
        self.rope_proportion = rope_proportion
        self.sliding_window_size = sliding_window_size
        self.attn_logits_soft_cap = attn_logits_soft_cap
        self.k_eq_v = k_eq_v
        self.attn_impl = attn_impl
        self.groups = num_heads // num_kv_heads  # GQA group count

        # Projections
        dd = factory_kwargs(device, dtype)
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False, **dd)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False, **dd)
        if not k_eq_v:
            self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False, **dd)
        self.o_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False, **dd)

        # Norms: QK-norm replaces 1/sqrt(d) scaling, so attn scale = 1.0
        self.query_pre_attn_scalar = 1.0
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, with_scale=True, **dd)
            self.k_norm = RMSNorm(head_dim, with_scale=True, **dd)
        else:
            self.q_norm = self.k_norm = None

        if use_value_norm:
            self.v_norm = RMSNorm(head_dim, with_scale=False, **dd)
        else:
            self.v_norm = None

    def _init_weights(self, ctx) -> None:
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if hasattr(self, "v_proj"):
            nn.init.normal_(self.v_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
            if self.v_proj.bias is not None:
                nn.init.zeros_(self.v_proj.bias)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=self.residual_init_std, generator=ctx.generator)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)
        if self.q_norm is not None and self.q_norm.weight is not None:
            nn.init.ones_(self.q_norm.weight)
        if self.k_norm is not None and self.k_norm.weight is not None:
            nn.init.ones_(self.k_norm.weight)
        if self.v_norm is not None and self.v_norm.weight is not None:
            nn.init.ones_(self.v_norm.weight)

    # ---- cache helpers ---------------------------------------------------

    @staticmethod
    def init_cache(
            cache_length: int,
            num_kv_heads: int,
            head_dim: int,
            batch_size: int,
            dtype: torch.dtype = torch.bfloat16,
            device: torch.device | str = "cpu",
    ) -> LayerCache:
        shape = (batch_size, cache_length, num_kv_heads, head_dim)
        return LayerCache(
            k=torch.zeros(shape, dtype=dtype, device=device),
            v=torch.zeros(shape, dtype=dtype, device=device),
            positions=torch.zeros(batch_size, cache_length, dtype=torch.int32, device=device),
            end_index=torch.zeros(batch_size, dtype=torch.int32, device=device),
            valid_mask=torch.zeros(batch_size, cache_length, dtype=torch.bool, device=device),
        )

    # ---- forward ---------------------------------------------------------

    def forward(
            self,
            x: torch.Tensor,
            positions: torch.Tensor,
            attn_mask: torch.Tensor,
            cache: LayerCache | None = None,
            shared_kv_cache: LayerCache | None = None,
    ) -> tuple[LayerCache | None, torch.Tensor]:
        """
        Args:
            x: ``[B, L, D]``
            positions: ``[B, L]``
            attn_mask: ``[B, L, S]`` bool (True = attend)
            cache: optional KV cache for this layer
            shared_kv_cache: if not None, reuse KV from another layer
        """
        B, L, _ = x.shape

        # --- Q projection ---
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)

        # --- K/V projection (or reuse from shared cache) ---
        if shared_kv_cache is not None:
            k = shared_kv_cache["k"].to(q.dtype)
            v = shared_kv_cache["v"].to(q.dtype)
        else:
            k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
            v = (k if self.k_eq_v
                 else self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim))

        # --- Norms ---
        if self.q_norm is not None:
            q = self.q_norm(q)
        if shared_kv_cache is None and self.k_norm is not None:
            k = self.k_norm(k)
        if shared_kv_cache is None and self.v_norm is not None:
            v = self.v_norm(v)

        # --- RoPE ---
        q = apply_rope(q, positions, self.rope_base, self.rope_scale_factor, self.rope_proportion)
        if shared_kv_cache is None:
            k = apply_rope(k, positions, self.rope_base, self.rope_scale_factor, self.rope_proportion)

        # --- Scale (QK-norm → scale=1.0) ---
        q = q * self.query_pre_attn_scalar

        # --- KV cache update ---
        cache_positions = None
        valid_mask = None
        if shared_kv_cache is not None:
            # Reuse positions from the layer we're sharing KV with
            cache_positions = shared_kv_cache.get("positions")
            valid_mask = shared_kv_cache.get("valid_mask")
        elif cache is not None:
            # All batch elements must have the same end_index (single-batch
            # decode or uniform-length batched decode). Assert to catch misuse.
            end = cache["end_index"][0].item()
            assert (cache["end_index"] == end).all(), (
                "Heterogeneous batch end_index not supported; all batch "
                "elements must have the same cache fill level"
            )
            cache_len = cache["v"].shape[1]
            idx = end % cache_len
            cache["k"][:, idx : idx + L] = k.to(cache["k"].dtype)
            cache["v"][:, idx : idx + L] = v.to(cache["v"].dtype)
            cache["positions"][:, idx : idx + L] = positions
            cache["valid_mask"][:, idx : idx + L] = True
            k = cache["k"].to(q.dtype)
            v = cache["v"].to(q.dtype)
            cache_positions = cache["positions"]
            valid_mask = cache["valid_mask"]

        # --- Mask out unfilled cache slots ---
        if valid_mask is not None:
            attn_mask = attn_mask & valid_mask[:, None, :]  # [B, 1, S] broadcast over L

        # --- Sliding window mask ---
        if self.attn_type == AttentionType.LOCAL_SLIDING:
            assert self.sliding_window_size is not None
            slide = create_sliding_mask(
                positions,
                cache_positions=cache_positions,
                sliding_window_size=self.sliding_window_size,
            )
            attn_mask = attn_mask & slide

        # --- Compute attention (SDPA or eager) ---
        use_sdpa = (
            self.attn_impl == "sdpa"
            and self.attn_logits_soft_cap is None
        )
        if use_sdpa:
            out = self._sdpa_attention(q, k, v, attn_mask)
        else:
            out = self._eager_attention(q, k, v, attn_mask)

        out = out.reshape(B, L, -1)
        out = self.o_proj(out)

        # --- Build new cache ---
        new_cache: LayerCache | None = None
        if cache is not None:
            # Return the original cache tensors (already updated in-place) to
            # preserve the cache dtype (e.g. bfloat16) rather than the float32
            # copies used for attention computation.
            new_cache = {
                "k": cache["k"] if shared_kv_cache is None else cache["k"],
                "v": cache["v"] if shared_kv_cache is None else cache["v"],
                "end_index": cache["end_index"] + L,
                "positions": cache_positions if cache_positions is not None else cache["positions"],
                "valid_mask": cache["valid_mask"],
            }
        elif shared_kv_cache is None:
            # Still return layer-sharing KV (vertical sharing)
            new_cache = {"k": k, "v": v}

        return new_cache, out

    # ---- attention backends ----------------------------------------------

    def _sdpa_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA path using ``F.scaled_dot_product_attention``.

        Args:
            q: ``[B, L, num_heads, head_dim]``
            k: ``[B, S, num_kv_heads, head_dim]``
            v: ``[B, S, num_kv_heads, head_dim]``
            attn_mask: ``[B, L, S]`` bool (True = attend)

        Returns:
            ``[B, L, num_heads, head_dim]``
        """
        # Transpose to [B, H, L/S, D] for SDPA
        q = q.transpose(1, 2)  # [B, num_heads, L, head_dim]
        k = k.transpose(1, 2)  # [B, num_kv_heads, S, head_dim]
        v = v.transpose(1, 2)  # [B, num_kv_heads, S, head_dim]

        # Convert bool mask [B, L, S] → [B, 1, L, S] for SDPA broadcast over heads
        sdpa_mask = attn_mask.unsqueeze(1)

        # enable_gqa=True lets SDPA handle mismatched Q/KV head counts natively
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            scale=self.query_pre_attn_scalar,
            enable_gqa=True,
        )
        # Back to [B, L, num_heads, head_dim]
        return out.transpose(1, 2)

    def _eager_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Eager (manual einsum) attention path.

        Args:
            q: ``[B, L, num_heads, head_dim]``
            k: ``[B, S, num_kv_heads, head_dim]``
            v: ``[B, S, num_kv_heads, head_dim]``
            attn_mask: ``[B, L, S]`` bool (True = attend)

        Returns:
            ``[B, L, num_heads, head_dim]``
        """
        B = q.shape[0]
        L = q.shape[1]

        # Attention logits (GQA via reshape)
        if self.groups > 1:
            q_r = q.view(B, L, self.num_kv_heads, self.groups, self.head_dim)
            logits = torch.einsum("blkgh,bskh->blkgs", q_r, k)
            B2, L2, K2, G2, S2 = logits.shape
            logits = logits.reshape(B2, L2, K2 * G2, S2)
        else:
            logits = torch.einsum("blnh,bsnh->blns", q, k)

        # Softcap
        if self.attn_logits_soft_cap is not None:
            logits = torch.tanh(logits / self.attn_logits_soft_cap) * self.attn_logits_soft_cap

        # Masked softmax (float32 for numerical stability in bf16)
        padded = torch.where(attn_mask.unsqueeze(-2), logits, K_MASK)
        probs = F.softmax(padded.float(), dim=-1).to(k.dtype)

        # Weighted sum
        if self.groups > 1:
            probs = probs.view(B, L, self.num_kv_heads, self.groups, -1)
            out = torch.einsum("blkgs,bskh->blkgh", probs, v)
            out = out.reshape(B, L, self.num_heads, self.head_dim)
        else:
            out = torch.einsum("blns,bsnh->blnh", probs, v)

        return out
