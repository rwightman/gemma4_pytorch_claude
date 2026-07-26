"""Text decoder attention with GQA, QK/V-norm, sliding window, and KV cache."""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AttentionType
from .layers import RMSNorm, apply_rope
from .module_utils import DEFAULT_INIT_STD, InitModule, factory_kwargs

K_MASK = -2.3819763e38


# ---------------------------------------------------------------------------
# KV cache type
# ---------------------------------------------------------------------------

class LayerCache(TypedDict):
    k: torch.Tensor          # [B, cache_len, kv_heads, head_dim]
    v: torch.Tensor          # [B, cache_len, kv_heads, head_dim]
    positions: torch.Tensor  # [B, cache_len]
    end_index: torch.Tensor  # [B]
    valid_mask: torch.Tensor  # [B, cache_len] bool — True for filled slots
    offset: int              # host-side fill level (avoids a device sync per layer)
    rolling: bool            # True → writes may wrap and evict the oldest slots


def cache_offset(cache: LayerCache) -> int:
    """Host-side fill level of *cache*.

    Prefers the plain-int ``offset`` entry so the decode loop does not pay a
    device synchronization per layer; falls back to ``end_index`` for caches
    built by older code.
    """
    offset = cache.get("offset")
    if offset is None:
        return int(cache["end_index"].reshape(-1)[0].item())
    return int(offset)


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
# Bidirectional (image-span) mask
# ---------------------------------------------------------------------------

def bidirectional_block_ids(bidirectional_mask: torch.Tensor) -> torch.Tensor:
    """Number each contiguous True run in ``[B, L]`` with a unique id (0 = none)."""
    padded = F.pad(bidirectional_mask.long(), (1, 0), value=0)
    boundary = padded[:, 1:] > padded[:, :-1]        # rising edges
    return bidirectional_mask.long() * boundary.long().cumsum(dim=-1)


def create_bidirectional_mask(
        positions: torch.Tensor,
        key_positions: torch.Tensor,
        bidirectional_mask: torch.Tensor,
) -> torch.Tensor:
    """``[B, L, S]`` bool — True where query and key share a bidirectional span.

    Works for both the cacheless path (``key_positions is positions``) and the
    cached path, where the key axis is a KV-cache slot layout: keys are matched
    to queries by absolute position, so the mask is independent of where a token
    physically landed in the cache.
    """
    block_ids = bidirectional_block_ids(bidirectional_mask)          # [B, L]
    same_token = key_positions[:, None, :] == positions[:, :, None]  # [B, L, S]
    # Project query-side block ids onto the key axis (0 for keys not in this chunk).
    key_blocks = torch.einsum(
        "bls,bl->bs", same_token.to(torch.float32), block_ids.to(torch.float32)
    ).round().long()
    q_blocks = block_ids[:, :, None]
    return (q_blocks == key_blocks[:, None, :]) & (q_blocks > 0)


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
            init_std: float = DEFAULT_INIT_STD,
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
            rolling: bool = False,
    ) -> LayerCache:
        """Allocate a KV cache for one layer.

        Args:
            cache_length: number of slots.
            num_kv_heads: KV head count for this layer.
            head_dim: head dimension for this layer.
            batch_size: batch size.
            dtype: cache dtype.
            device: cache device.
            rolling: when True, writes past the end wrap around and evict the
                oldest slots.  Only valid for sliding-window layers whose
                ``cache_length`` covers the window; otherwise an overflow raises.
        """
        shape = (batch_size, cache_length, num_kv_heads, head_dim)
        return LayerCache(
            k=torch.zeros(shape, dtype=dtype, device=device),
            v=torch.zeros(shape, dtype=dtype, device=device),
            positions=torch.zeros(batch_size, cache_length, dtype=torch.int32, device=device),
            end_index=torch.zeros(batch_size, dtype=torch.int32, device=device),
            valid_mask=torch.zeros(batch_size, cache_length, dtype=torch.bool, device=device),
            offset=0,
            rolling=rolling,
        )

    def _write_cache(
            self,
            cache: LayerCache,
            k: torch.Tensor,
            v: torch.Tensor,
            positions: torch.Tensor,
            start: int,
    ) -> None:
        """Write ``k``/``v``/``positions`` into *cache* at slot ``start``.

        Raises when the write would overflow a non-rolling cache, rather than
        silently evicting entries the attention still needs.
        """
        L = k.shape[1]
        cache_len = cache["v"].shape[1]

        if L > cache_len:
            raise ValueError(
                f"KV cache overflow: writing {L} tokens into a {cache_len}-slot cache. "
                f"Allocate a cache of at least prompt_len + max_new_tokens slots."
            )
        if start + L > cache_len and not cache.get("rolling", False):
            raise ValueError(
                f"KV cache overflow: slot {start} + {L} tokens exceeds {cache_len} slots. "
                f"Allocate a larger cache (cache_length >= prompt_len + max_new_tokens)."
            )

        idx = start % cache_len
        end = idx + L
        if end <= cache_len:
            slices = ((slice(idx, end), slice(0, L)),)
        else:
            # Rolling cache: split the write across the buffer boundary.
            head = cache_len - idx
            slices = (
                (slice(idx, cache_len), slice(0, head)),
                (slice(0, L - head), slice(head, L)),
            )

        for dst, src in slices:
            cache["k"][:, dst] = k[:, src].to(cache["k"].dtype)
            cache["v"][:, dst] = v[:, src].to(cache["v"].dtype)
            cache["positions"][:, dst] = positions[:, src].to(cache["positions"].dtype)
            cache["valid_mask"][:, dst] = True

    # ---- forward ---------------------------------------------------------

    def forward(
            self,
            x: torch.Tensor,
            positions: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            cache: LayerCache | None = None,
            shared_kv_cache: LayerCache | None = None,
            bidirectional_mask: torch.Tensor | None = None,
    ) -> tuple[LayerCache | None, torch.Tensor]:
        """
        Args:
            x: ``[B, L, D]``
            positions: ``[B, L]``
            attn_mask: ``[B, L, S]`` bool (True = attend).  When None a causal
                mask is derived from ``positions`` and the cached key positions,
                which is layout-independent and therefore also correct for
                rolling caches.
            cache: optional KV cache for this layer
            shared_kv_cache: if not None, reuse KV from another layer
            bidirectional_mask: ``[B, L]`` bool — tokens inside each contiguous
                True span attend to each other regardless of causal order.
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
        start = 0
        if shared_kv_cache is not None:
            # Reuse positions from the layer we're sharing KV with
            cache_positions = shared_kv_cache.get("positions")
            valid_mask = shared_kv_cache.get("valid_mask")
        elif cache is not None:
            start = cache_offset(cache)
            self._write_cache(cache, k, v, positions, start)
            k = cache["k"].to(q.dtype)
            v = cache["v"].to(q.dtype)
            cache_positions = cache["positions"]
            valid_mask = cache["valid_mask"]

        # --- Attention mask ---
        # Derive causally from positions when not supplied: this stays correct
        # for rolling caches, where slot order does not track position order.
        key_positions = cache_positions if cache_positions is not None else positions
        if attn_mask is None:
            attn_mask = key_positions[:, None, :] <= positions[:, :, None]

        if bidirectional_mask is not None:
            attn_mask = attn_mask | create_bidirectional_mask(
                positions, key_positions, bidirectional_mask,
            )

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
                **cache,
                "end_index": cache["end_index"] + L,
                "offset": start + L,
                "positions": cache_positions if cache_positions is not None else cache["positions"],
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
