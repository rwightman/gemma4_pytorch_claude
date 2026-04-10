"""TransformerBlock + TextDecoder (Embedder, per-layer-input, KV sharing)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, LayerCache
from .config import AttentionType, TextConfig, build_kv_sharing_patterns, make_attention_pattern
from .layers import GatedMLP, RMSNorm, TanhGELU
from .moe import MoELayer


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder(nn.Module):
    """Token embedding + optional per-layer-input embedding + tied logit decode."""

    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.vocab_size = cfg.vocab_size
        self.embed_scale = math.sqrt(cfg.embed_dim)

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)

        # Per-layer input: small per-token-per-layer embedding
        self.per_layer_input_dim = cfg.per_layer_input_dim
        if cfg.per_layer_input_dim > 0:
            num_layers = cfg.num_layers
            self.pli_embedding = nn.Embedding(cfg.vocab_size, num_layers * cfg.per_layer_input_dim)
            self.pli_proj = nn.Linear(cfg.embed_dim, num_layers * cfg.per_layer_input_dim, bias=False)
            self.pli_proj_norm = RMSNorm(cfg.per_layer_input_dim, scale_plus_one=False)
            self.num_layers = num_layers
            self.pli_proj_scale = self._build_pli_proj_scale()
            self.pli_combine_scale = self._build_pli_combine_scale()

    def _build_pli_proj_scale(self) -> float:
        return float(self.embed_dim) ** -0.5

    def _build_pli_combine_scale(self) -> float:
        return 1.0 / math.sqrt(2.0)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, L] -> [B, L, D]`` with sqrt(D) scaling."""
        return self.token_embedding(tokens) * self.embed_scale

    def decode_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Tied weight logits: ``x @ W^T``."""
        return F.linear(x, self.token_embedding.weight)

    def encode_per_layer_input(
            self, x: torch.Tensor, tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-layer inputs: ``[B, L, num_layers, pli_dim]``.

        Combines a projection of the main embedding ``x`` with a direct
        per-layer-input embedding lookup, scaled by ``rsqrt(2)``.
        """
        pli_dim = self.per_layer_input_dim
        nl = self.num_layers

        # Replace out-of-vocab tokens (e.g. MM placeholders) with 0 for lookup
        safe_tokens = torch.where(
            (tokens >= 0) & (tokens < self.vocab_size), tokens, torch.zeros_like(tokens)
        )

        # Projection path: x -> project -> reshape -> norm (over pli_dim)
        proj = self.pli_proj(x) * self.pli_proj_scale
        proj = proj.view(*x.shape[:-1], nl, pli_dim)
        proj = self.pli_proj_norm(proj)

        # Embedding path
        emb = self.pli_embedding(safe_tokens)
        emb = emb * math.sqrt(pli_dim)
        emb = emb.view(*x.shape[:-1], nl, pli_dim)

        return (proj + emb) * self.pli_combine_scale


# ---------------------------------------------------------------------------
# PerLayerMapping (within each block)
# ---------------------------------------------------------------------------

class PerLayerMapping(nn.Module):
    """Gate + project per-layer input back to embed_dim."""

    def __init__(self, embed_dim: int, pli_dim: int):
        super().__init__()
        self.gate = nn.Linear(embed_dim, pli_dim, bias=False)
        self.act = TanhGELU()
        self.proj = nn.Linear(pli_dim, embed_dim, bias=False)
        self.norm = RMSNorm(embed_dim, scale_plus_one=False)

    def forward(self, x: torch.Tensor, pli: torch.Tensor) -> torch.Tensor:
        g = self.act(self.gate(x))
        out = self.proj(g * pli)
        return self.norm(out)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single transformer block with optional MoE and per-layer-input."""

    def __init__(
            self,
            cfg: TextConfig,
            layer_idx: int,
            attn_type: AttentionType,
            *,
            head_dim: int | None = None,
            num_kv_heads: int | None = None,
            hidden_dim: int | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_type = attn_type

        # Resolve per-layer dimensions (global layers can differ)
        effective_head_dim = head_dim or cfg.head_dim
        effective_kv_heads = num_kv_heads or cfg.num_kv_heads
        effective_hidden = hidden_dim or cfg.hidden_dim

        # Choose RoPE base by layer type
        rope_base = (
            cfg.local_rope_base
            if attn_type == AttentionType.LOCAL_SLIDING
            else cfg.global_rope_base
        )
        rope_scale = (
            1.0
            if attn_type == AttentionType.LOCAL_SLIDING
            else cfg.global_rope_scale_factor
        )
        rope_prop = (
            cfg.rope_proportion
            if attn_type == AttentionType.LOCAL_SLIDING
            else cfg.global_rope_proportion
            if cfg.global_rope_proportion > 0
            else cfg.rope_proportion
        )
        k_eq_v = (
            cfg.k_eq_v
            if attn_type == AttentionType.LOCAL_SLIDING
            else cfg.k_eq_v_global
        )

        # Pre/post attention norms
        self.pre_attn_norm = RMSNorm(cfg.embed_dim, scale_plus_one=False)
        self.attn = Attention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_kv_heads=effective_kv_heads,
            head_dim=effective_head_dim,
            attn_type=attn_type,
            rope_base=rope_base,
            rope_scale_factor=rope_scale,
            rope_proportion=rope_prop,
            sliding_window_size=cfg.sliding_window_size,
            attn_logits_soft_cap=cfg.attn_logits_soft_cap,
            use_qk_norm=cfg.use_qk_norm,
            use_value_norm=cfg.use_value_norm,
            k_eq_v=k_eq_v,
            attn_impl=cfg.attn_impl,
        )
        self.post_attn_norm = (
            RMSNorm(cfg.embed_dim, scale_plus_one=False) if cfg.use_post_attn_norm else None
        )

        # Feed-forward
        self.is_moe = cfg.moe is not None
        self.pre_ffw_norm = RMSNorm(cfg.embed_dim, scale_plus_one=False)
        if self.is_moe:
            # MoE branch: router + experts only (no dense inside MoELayer)
            self.moe = MoELayer(
                features=cfg.embed_dim,
                num_experts=cfg.moe.num_experts,
                top_k=cfg.moe.top_k,
                expert_dim=cfg.moe.expert_dim,
            )
            self.post_ffw1_norm = (
                RMSNorm(cfg.embed_dim, scale_plus_one=False) if cfg.use_post_ffw_norm else None
            )
            # Dense branch (parallel to MoE)
            dense_hid = cfg.moe.dense_hidden_dim if cfg.moe.dense_hidden_dim > 0 else effective_hidden
            self.pre_ffw2_norm = RMSNorm(cfg.embed_dim, scale_plus_one=False)
            self.mlp2 = GatedMLP(cfg.embed_dim, dense_hid)
            self.post_ffw2_norm = (
                RMSNorm(cfg.embed_dim, scale_plus_one=False) if cfg.use_post_ffw_norm else None
            )
            # Combined post-norm
            self.post_ffw_norm = (
                RMSNorm(cfg.embed_dim, scale_plus_one=False) if cfg.use_post_ffw_norm else None
            )
        else:
            self.ffw = GatedMLP(cfg.embed_dim, effective_hidden)
            self.post_ffw_norm = (
                RMSNorm(cfg.embed_dim, scale_plus_one=False) if cfg.use_post_ffw_norm else None
            )

        # Per-layer input
        if cfg.per_layer_input_dim > 0:
            self.pli_mapping = PerLayerMapping(cfg.embed_dim, cfg.per_layer_input_dim)
        else:
            self.pli_mapping = None

        # Skip scale: scalar learnable multiplier applied at end of block
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(
            self,
            x: torch.Tensor,
            positions: torch.Tensor,
            attn_mask: torch.Tensor,
            cache: LayerCache | None = None,
            shared_kv_cache: LayerCache | None = None,
            per_layer_input: torch.Tensor | None = None,
    ) -> tuple[LayerCache | None, torch.Tensor]:
        # --- Attention ---
        residual = x
        h = self.pre_attn_norm(x)
        cache, h = self.attn(h, positions, attn_mask, cache=cache, shared_kv_cache=shared_kv_cache)
        if self.post_attn_norm is not None:
            h = self.post_attn_norm(h)
        x = residual + h

        # --- Feed-forward ---
        residual = x
        if self.is_moe:
            # MoE branch
            moe_out = self.pre_ffw_norm(x)
            moe_out = self.moe(moe_out)
            if self.post_ffw1_norm is not None:
                moe_out = self.post_ffw1_norm(moe_out)
            # Dense branch (parallel)
            dense_out = self.pre_ffw2_norm(x)
            dense_out = self.mlp2(dense_out)
            if self.post_ffw2_norm is not None:
                dense_out = self.post_ffw2_norm(dense_out)
            # Combine
            h = moe_out + dense_out
            if self.post_ffw_norm is not None:
                h = self.post_ffw_norm(h)
        else:
            h = self.pre_ffw_norm(x)
            h = self.ffw(h)
            if self.post_ffw_norm is not None:
                h = self.post_ffw_norm(h)
        x = residual + h

        # --- Per-layer input ---
        if self.pli_mapping is not None and per_layer_input is not None:
            x = x + self.pli_mapping(x, per_layer_input)

        # --- Skip scale ---
        x = x * self.skip_scale

        return cache, x


# ---------------------------------------------------------------------------
# TextDecoder
# ---------------------------------------------------------------------------

class TextDecoder(nn.Module):
    """Full text decoder stack: embedder + N blocks + final norm."""

    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.embedder = Embedder(cfg)

        # Build attention type sequence
        attn_types = make_attention_pattern(cfg.attention_pattern, cfg.num_layers)
        self.attn_types = attn_types

        # Build KV sharing patterns
        self.kv_sharing_patterns = build_kv_sharing_patterns(
            cfg.num_layers, attn_types, cfg.kv_sharing
        )

        # Transformer blocks — resolve per-layer dimensions
        blocks: list[TransformerBlock] = []
        for i in range(cfg.num_layers):
            is_global = attn_types[i] == AttentionType.GLOBAL
            is_shared = self.kv_sharing_patterns[i] != i

            layer_head_dim = (cfg.global_head_dim or cfg.head_dim) if is_global else cfg.head_dim
            layer_kv_heads = (cfg.num_global_kv_heads or cfg.num_kv_heads) if is_global else cfg.num_kv_heads
            layer_hidden = cfg.hidden_dim
            if is_shared and cfg.override_kv_shared_ffw_hidden is not None:
                layer_hidden = cfg.override_kv_shared_ffw_hidden

            blocks.append(TransformerBlock(
                cfg, i, attn_types[i],
                head_dim=layer_head_dim,
                num_kv_heads=layer_kv_heads,
                hidden_dim=layer_hidden,
            ))
        self.blocks = nn.ModuleList(blocks)

        self.final_norm = RMSNorm(cfg.embed_dim, scale_plus_one=False)

    def forward(
            self,
            x: torch.Tensor,
            positions: torch.Tensor,
            attn_mask: torch.Tensor,
            per_layer_inputs: torch.Tensor | None = None,
            cache: dict[str, LayerCache] | None = None,
    ) -> tuple[torch.Tensor, dict[str, LayerCache] | None]:
        """
        Args:
            x: ``[B, L, D]`` — already-embedded tokens
            positions: ``[B, L]``
            attn_mask: ``[B, L, S]`` bool
            per_layer_inputs: ``[B, L, num_layers, pli_dim]`` or None
            cache: dict mapping ``"layer_i"`` to LayerCache

        Returns:
            logits: ``[B, L, V]``
            new_cache: updated cache dict (or None)
        """
        old_cache = cache or {}
        new_cache: dict[str, LayerCache] = {}

        for i, block in enumerate(self.blocks):
            layer_name = f"layer_{i}"
            source_idx = self.kv_sharing_patterns[i]
            is_shared = source_idx != i

            # KV sharing: check if this layer borrows KV from an earlier layer
            shared_kv = None
            if is_shared:
                shared_name = f"layer_{source_idx}"
                shared_kv = new_cache.get(shared_name)

            pli = per_layer_inputs[..., i, :] if per_layer_inputs is not None else None

            layer_cache, x = block(
                x,
                positions,
                attn_mask,
                cache=old_cache.get(layer_name),  # None for shared layers
                shared_kv_cache=shared_kv,
                per_layer_input=pli,
            )

            if is_shared:
                # Store the canonical source cache — self-consistent, no phantom buffers
                new_cache[layer_name] = new_cache[f"layer_{source_idx}"]
            else:
                new_cache[layer_name] = layer_cache

        x = self.final_norm(x)
        logits = self.embedder.decode_logits(x)

        # Final logit softcap
        if self.cfg.final_logit_softcap is not None:
            cap = self.cfg.final_logit_softcap
            logits = torch.tanh(logits / cap) * cap

        return logits, (new_cache if cache is not None else None)
