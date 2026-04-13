"""Gemma4 vision encoder: patchify → transformer blocks → spatial pooling.

Ported from the JAX reference: ``gemma/gm/nn/gemma4/vision/``.
Uses factorised 2-D positional embeddings, QK/V-norm, gated MLP,
multidimensional RoPE, and position-aware spatial pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VisionConfig
from .layers import ClippedLinear, RMSNorm, TanhGELU, VisionRMSNorm, apply_multidimensional_rope
from .module_utils import InitModule, factory_kwargs, resolve_residual_init_std


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vision_proj(
        cfg: VisionConfig,
        in_features: int,
        out_features: int,
        *,
        residual_init_std: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
) -> ClippedLinear | nn.Linear:
    """Build a projection layer — ClippedLinear when the config asks for it."""
    if cfg.use_clipped_linear:
        return ClippedLinear(
            in_features,
            out_features,
            bias=False,
            init_std=cfg.init_std,
            residual_init_std=residual_init_std,
            device=device,
            dtype=dtype,
        )
    return nn.Linear(in_features, out_features, bias=False, **factory_kwargs(device, dtype))


# ---------------------------------------------------------------------------
# Patch embedder with factorised 2-D positional embedding
# ---------------------------------------------------------------------------

class VisionPatchEmbedder(InitModule):
    def __init__(
            self,
            cfg: VisionConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.cfg = cfg
        self.init_std = cfg.init_std
        self.position_init_std = cfg.position_init_std
        patch_dim = 3 * cfg.patch_size ** 2
        # JAX uses plain Einsum (not ClippedEinsum) for input_projection
        # across all model sizes, so always use nn.Linear here.
        self.input_proj = nn.Linear(patch_dim, cfg.d_model, bias=False, **dd)
        self.position_embedding_table = nn.Parameter(
            torch.zeros(2, cfg.position_embedding_size, cfg.d_model, **dd),
        )

    def _position_embeddings(
            self,
            position_ids: torch.Tensor,
            padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute factorised positional embeddings from 2-D coords.

        Args:
            position_ids: ``[B, L, 2]`` (x, y).
            padding_mask: ``[B, L]`` True where padding.
        """
        clamped = position_ids.clamp(min=0)  # [B, L, 2]
        one_hot = F.one_hot(
            clamped, num_classes=self.cfg.position_embedding_size,
        )  # [B, L, 2, pos_size]
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)  # [B, 2, L, pos_size]
        pos_emb = one_hot @ self.position_embedding_table  # [B, 2, L, d_model]
        pos_emb = pos_emb.sum(dim=1)  # [B, L, d_model]
        pos_emb = torch.where(padding_mask.unsqueeze(-1), 0.0, pos_emb)
        return pos_emb

    def forward(
            self,
            pixel_values: torch.Tensor,
            position_ids: torch.Tensor,
            padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: ``[B, L, patch_dim]`` raw patch pixels in [0,1].
            position_ids: ``[B, L, 2]``.
            padding_mask: ``[B, L]`` True for padding patches.
        """
        pixel_values = 2.0 * (pixel_values - 0.5)
        proj_dtype = next(self.input_proj.parameters()).dtype
        hidden = self.input_proj(pixel_values.to(proj_dtype))
        pos_emb = self._position_embeddings(position_ids, padding_mask)
        return hidden + pos_emb

    def _init_weights(self, ctx) -> None:
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(
            self.position_embedding_table,
            mean=0.0,
            std=self.position_init_std,
            generator=ctx.generator,
        )


# ---------------------------------------------------------------------------
# Vision MLP (gated, separate gate/up/down)
# ---------------------------------------------------------------------------

class VisionMLP(InitModule):
    def __init__(
            self,
            cfg: VisionConfig,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.init_std = cfg.init_std
        self.residual_init_std = cfg.init_std if residual_init_std is None else residual_init_std
        dd = factory_kwargs(device, dtype)
        self.gate_proj = _make_vision_proj(cfg, cfg.d_model, cfg.ffw_hidden, residual_init_std=cfg.init_std, **dd)
        self.up_proj = _make_vision_proj(cfg, cfg.d_model, cfg.ffw_hidden, residual_init_std=cfg.init_std, **dd)
        self.act = TanhGELU(**dd)
        self.down_proj = _make_vision_proj(cfg, cfg.ffw_hidden, cfg.d_model, residual_init_std=residual_init_std, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

    def _init_weights(self, ctx) -> None:
        if isinstance(self.gate_proj, nn.Linear):
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
            if self.gate_proj.bias is not None:
                nn.init.zeros_(self.gate_proj.bias)
        if isinstance(self.up_proj, nn.Linear):
            nn.init.normal_(self.up_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
            if self.up_proj.bias is not None:
                nn.init.zeros_(self.up_proj.bias)
        if isinstance(self.down_proj, nn.Linear):
            nn.init.normal_(self.down_proj.weight, mean=0.0, std=self.residual_init_std, generator=ctx.generator)
            if self.down_proj.bias is not None:
                nn.init.zeros_(self.down_proj.bias)


# ---------------------------------------------------------------------------
# Vision attention (full MHA with 2-D RoPE + QK/V-norm)
# ---------------------------------------------------------------------------

class VisionAttention(InitModule):
    def __init__(
            self,
            cfg: VisionConfig,
            residual_init_std: float | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.init_std = cfg.init_std
        self.residual_init_std = cfg.init_std if residual_init_std is None else residual_init_std
        self.rope_base_frequency = cfg.rope_base_frequency
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        dd = factory_kwargs(device, dtype)

        self.q_proj = _make_vision_proj(cfg, cfg.d_model, cfg.num_heads * cfg.head_dim, residual_init_std=self.init_std, **dd)
        self.k_proj = _make_vision_proj(cfg, cfg.d_model, cfg.num_heads * cfg.head_dim, residual_init_std=self.init_std, **dd)
        self.v_proj = _make_vision_proj(cfg, cfg.d_model, cfg.num_heads * cfg.head_dim, residual_init_std=self.init_std, **dd)
        self.o_proj = _make_vision_proj(cfg, cfg.num_heads * cfg.head_dim, cfg.d_model, residual_init_std=self.residual_init_std, **dd)

        self.q_norm = VisionRMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps, **dd)
        self.k_norm = VisionRMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps, **dd)
        self.v_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps, with_scale=False, **dd)

    def _init_weights(self, ctx) -> None:
        if isinstance(self.q_proj, nn.Linear):
            nn.init.normal_(self.q_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
            if self.q_proj.bias is not None:
                nn.init.zeros_(self.q_proj.bias)
        if isinstance(self.k_proj, nn.Linear):
            nn.init.normal_(self.k_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
            if self.k_proj.bias is not None:
                nn.init.zeros_(self.k_proj.bias)
        if isinstance(self.v_proj, nn.Linear):
            nn.init.normal_(self.v_proj.weight, mean=0.0, std=self.init_std, generator=ctx.generator)
            if self.v_proj.bias is not None:
                nn.init.zeros_(self.v_proj.bias)
        if isinstance(self.o_proj, nn.Linear):
            nn.init.normal_(self.o_proj.weight, mean=0.0, std=self.residual_init_std, generator=ctx.generator)
            if self.o_proj.bias is not None:
                nn.init.zeros_(self.o_proj.bias)
        if self.q_norm.weight is not None:
            nn.init.zeros_(self.q_norm.weight)
        if self.k_norm.weight is not None:
            nn.init.zeros_(self.k_norm.weight)
        if self.v_norm.weight is not None:
            nn.init.ones_(self.v_norm.weight)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: ``[B, L, D]``.
            attention_mask: ``[B, 1, L, L]`` additive mask (0 attend, -inf ignore).
            position_ids: ``[B, L, 2]`` — (x, y) patch coords.
        """
        B, L, _ = x.shape
        N, H = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, N, H)
        k = self.k_proj(x).view(B, L, N, H)
        v = self.v_proj(x).view(B, L, N, H)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # 2-D RoPE: first half ↔ positions[..., 0] (x/width),
        #           second half ↔ positions[..., 1] (y/height).
        # Matches JAX: apply_multidimensional_rope splits head_dim and applies
        # apply_rope to each spatial dimension independently.
        q = apply_multidimensional_rope(
            q, position_ids[:, :, 0], position_ids[:, :, 1],
            base_frequency=self.rope_base_frequency,
        )
        k = apply_multidimensional_rope(
            k, position_ids[:, :, 0], position_ids[:, :, 1],
            base_frequency=self.rope_base_frequency,
        )

        # Transpose to [B, N, L, H] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)

        out = torch.matmul(attn_weights, v)  # [B, N, L, H]
        out = out.transpose(1, 2).reshape(B, L, N * H)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Vision transformer block
# ---------------------------------------------------------------------------

class VisionBlock(InitModule):
    def __init__(
            self,
            cfg: VisionConfig,
            layer_idx: int,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        residual_init_std = resolve_residual_init_std(
            cfg.init_std,
            cfg.residual_init_std,
            cfg.use_depth_scaled_residual_init,
            cfg.num_layers,
        )
        dd = factory_kwargs(device, dtype)
        self.pre_attn_norm = VisionRMSNorm(cfg.d_model, eps=cfg.rms_norm_eps, **dd)
        self.post_attn_norm = VisionRMSNorm(cfg.d_model, eps=cfg.rms_norm_eps, **dd)
        self.pre_ffw_norm = VisionRMSNorm(cfg.d_model, eps=cfg.rms_norm_eps, **dd)
        self.post_ffw_norm = VisionRMSNorm(cfg.d_model, eps=cfg.rms_norm_eps, **dd)
        self.attn = VisionAttention(cfg, residual_init_std=residual_init_std, **dd)
        self.mlp = VisionMLP(cfg, residual_init_std=residual_init_std, **dd)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x = self.pre_attn_norm(x)
        x = self.attn(x, attention_mask, position_ids)
        x = self.post_attn_norm(x)
        x = residual + x

        residual = x
        x = self.pre_ffw_norm(x)
        x = self.mlp(x)
        x = self.post_ffw_norm(x)
        x = residual + x
        return x

    def _init_weights(self, ctx) -> None:
        if self.pre_attn_norm.weight is not None:
            nn.init.zeros_(self.pre_attn_norm.weight)
        if self.post_attn_norm.weight is not None:
            nn.init.zeros_(self.post_attn_norm.weight)
        if self.pre_ffw_norm.weight is not None:
            nn.init.zeros_(self.pre_ffw_norm.weight)
        if self.post_ffw_norm.weight is not None:
            nn.init.zeros_(self.post_ffw_norm.weight)


# ---------------------------------------------------------------------------
# Spatial pooler (position-aware average pooling)
# ---------------------------------------------------------------------------

class VisionPooler(InitModule):
    def __init__(
            self,
            cfg: VisionConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        _ = device, dtype
        self.root_hidden_size = cfg.d_model ** 0.5

    def _avg_pool_by_positions(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Position-based spatial average pooling.

        Groups patches into ``k x k`` grid cells based on their (x, y) coords
        and averages within each cell.

        Returns:
            ``(pooled [B, output_length, D], mask [B, output_length] bool)``.
        """
        input_len = hidden_states.shape[1]
        k = int((input_len / output_length) ** 0.5)
        k_squared = k * k

        clamped = position_ids.clamp(min=0)  # [B, L, 2]
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1  # [B, 1]
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")  # [B, L, 2]
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]  # [B, L]

        weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared  # [B, L, output_length]
        pooled = weights.transpose(1, 2) @ hidden_states.float()  # [B, output_length, D]
        mask = ~(weights == 0).all(dim=1)  # [B, output_length]
        return pooled.to(hidden_states.dtype), mask

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            padding_mask: torch.Tensor,
            output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: ``[B, L, D]``.
            position_ids: ``[B, L, 2]``.
            padding_mask: ``[B, L]`` True=padding.
            output_length: target number of soft tokens.

        Returns:
            ``(hidden_states, pooler_mask)`` where pooler_mask is ``[B, output_length]``
            with True=valid, False=padding.
        """
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if hidden_states.shape[1] != output_length:
            hidden_states, valid_mask = self._avg_pool_by_positions(
                hidden_states, position_ids, output_length,
            )
        else:
            valid_mask = ~padding_mask

        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, valid_mask


# ---------------------------------------------------------------------------
# VisionEncoder (top-level)
# ---------------------------------------------------------------------------

class VisionEncoder(InitModule):
    """Full vision encoder: patches → blocks → pool.

    Does NOT include projection to text space — that is handled by
    MultimodalEmbedder in model.py.
    """

    def __init__(
            self,
            cfg: VisionConfig,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dd = factory_kwargs(device, dtype)
        self.cfg = cfg
        self.patch_embedder = VisionPatchEmbedder(cfg, **dd)
        self.layers = nn.ModuleList([
            VisionBlock(cfg, layer_idx=i, **dd) for i in range(cfg.num_layers)
        ])
        self.pooler = VisionPooler(cfg, **dd)

        if cfg.standardize:
            self.register_buffer("std_bias", torch.zeros(cfg.d_model, **dd))
            self.register_buffer("std_scale", torch.ones(cfg.d_model, **dd))

    def forward(
            self,
            pixel_values: torch.Tensor,
            position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: ``[B, max_patches, patch_dim]`` in [0, 1].
            position_ids: ``[B, max_patches, 2]`` — (x, y) coords, -1 for padding.

        Returns:
            ``(hidden_states, pooler_mask)`` where hidden_states is
            ``[B, output_length, D]`` and pooler_mask is ``[B, output_length]``
            bool (True = valid soft token).
        """
        pooling_kernel_size = self.cfg.pooling_kernel_size
        output_length = pixel_values.shape[1] // (pooling_kernel_size ** 2)

        padding_mask = (position_ids == -1).all(dim=-1)  # [B, L]

        # Bidirectional attention mask: mask both query and key padding positions.
        # JAX: attention_mask = input_mask[:, :, None] * input_mask[:, None, :]
        # Use finfo.min (large finite neg) instead of -inf to avoid NaN from
        # softmax on all-masked padding rows (JAX uses jnp.finfo(dtype).min).
        valid = ~padding_mask  # [B, L]
        attn_mask = valid.unsqueeze(2) & valid.unsqueeze(1)  # [B, L, L]
        big_neg = torch.finfo(pixel_values.dtype).min
        attn_mask = torch.where(
            attn_mask.unsqueeze(1),  # [B, 1, L, L]
            torch.tensor(0.0, device=pixel_values.device),
            torch.tensor(big_neg, device=pixel_values.device),
        )

        x = self.patch_embedder(pixel_values, position_ids, padding_mask)

        for layer in self.layers:
            x = layer(x, attn_mask, position_ids)

        hidden_states, pooler_mask = self.pooler(x, position_ids, padding_mask, output_length)

        if self.cfg.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        return hidden_states, pooler_mask

    def _init_weights(self, ctx) -> None:
        if self.cfg.standardize:
            self.std_bias.zero_()
            self.std_scale.fill_(1.0)
