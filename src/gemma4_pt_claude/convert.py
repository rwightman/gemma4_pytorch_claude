"""Convert Orbax (JAX) checkpoint → safetensors for our PyTorch model.

Usage::

    # As CLI entry point:
    gemma4-convert --checkpoint /path/to/orbax --variant e2b --output model.safetensors

    # As Python API:
    from gemma4_pt_claude.convert import convert_orbax
    convert_orbax("/path/to/orbax", "e2b", "model.safetensors")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from .config import AttentionType, make_attention_pattern
from .factory import gemma4_e2b, gemma4_e4b, gemma4_31b, gemma4_26b_a4b


# ---------------------------------------------------------------------------
# Variant → factory mapping
# ---------------------------------------------------------------------------

_VARIANT_FACTORIES = {
    "e2b": gemma4_e2b,
    "e4b": gemma4_e4b,
    "31b": gemma4_31b,
    "26b_a4b": gemma4_26b_a4b,
}


# ---------------------------------------------------------------------------
# JAX → PyTorch weight transforms
# ---------------------------------------------------------------------------

def _convert_q_einsum(w: np.ndarray) -> torch.Tensor:
    """``(N, D, H)`` → ``(N*H, D)`` Linear weight."""
    N, D, H = w.shape
    return torch.from_numpy(w.transpose(0, 2, 1).reshape(N * H, D))


def _convert_kv_einsum(w: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """``(2, K, D, H)`` → separate K and V as ``(K*H, D)`` each."""
    k_w = w[0]  # (K, D, H)
    v_w = w[1]  # (K, D, H)
    K, D, H = k_w.shape
    k_out = torch.from_numpy(k_w.transpose(0, 2, 1).reshape(K * H, D))
    v_out = torch.from_numpy(v_w.transpose(0, 2, 1).reshape(K * H, D))
    return k_out, v_out


def _convert_attn_vec(w: np.ndarray) -> torch.Tensor:
    """``(N, H, D)`` → ``(D, N*H)`` Linear weight (transposed)."""
    N, H, D = w.shape
    return torch.from_numpy(w.reshape(N * H, D).T.copy())


def _convert_gating_einsum(w: np.ndarray) -> torch.Tensor:
    """``(2, F, D)`` → ``(2*F, D)`` fused gate+up weight.

    JAX stores as ``(2, hidden_dim, embed_dim)`` with einsum ``'td,fd->tf'``.
    PyTorch ``nn.Linear(embed, 2*hidden)`` weight is ``(2*hidden, embed)``.
    Both use ``(F, D)`` layout, so just reshape — no transpose.
    """
    two, F, D = w.shape
    return torch.from_numpy(w.reshape(two * F, D).copy())


def _convert_linear(w: np.ndarray) -> torch.Tensor:
    """``(F, D)`` → ``(D, F)`` transposed."""
    return torch.from_numpy(w.T.copy())


def _convert_scale(w: np.ndarray) -> torch.Tensor:
    """Norm scale — no transform needed."""
    return torch.from_numpy(w.copy())


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def _load_orbax_checkpoint(checkpoint_path: str) -> dict:
    """Load an Orbax checkpoint and return flat param dict."""
    try:
        import orbax.checkpoint as ocp
    except ImportError:
        raise ImportError(
            "orbax-checkpoint is required for conversion. "
            "Install with: pip install 'gemma4-pt-claude[convert]'"
        )

    # Support both local paths and GCS URIs (gs://...)
    ckpt_path: str | Path = checkpoint_path
    if not checkpoint_path.startswith("gs://"):
        ckpt_path = Path(checkpoint_path)
    checkpointer = ocp.PyTreeCheckpointer()
    raw = checkpointer.restore(ckpt_path)

    # Flatten the nested dict to slash-separated keys
    flat = {}

    def _flatten(d: dict, prefix: str = ""):
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                arr = np.array(v)
                # JAX may store weights as bfloat16 (ml_dtypes) which
                # torch.from_numpy doesn't support — cast to float32.
                if arr.dtype.name == "bfloat16":
                    arr = arr.astype(np.float32)
                flat[key] = arr

    _flatten(raw)
    return flat


def _map_text_layer(
        jax_params: dict,
        layer_idx: int,
        attn_type: AttentionType,
        prefix: str = "text_decoder",
        has_moe: bool = False,
        k_eq_v: bool = False,
) -> dict[str, torch.Tensor]:
    """Map one transformer layer from JAX to our PyTorch naming."""
    jax_prefix = f"layer_{layer_idx}"
    pt_prefix = f"{prefix}.blocks.{layer_idx}"
    out: dict[str, torch.Tensor] = {}

    # --- Attention ---
    q_key = f"{jax_prefix}/attn/q_einsum/w"
    kv_key = f"{jax_prefix}/attn/kv_einsum/w"
    o_key = f"{jax_prefix}/attn/attn_vec_einsum/w"

    if q_key in jax_params:
        out[f"{pt_prefix}.attn.q_proj.weight"] = _convert_q_einsum(jax_params[q_key])

    if kv_key in jax_params:
        k_w, v_w = _convert_kv_einsum(jax_params[kv_key])
        out[f"{pt_prefix}.attn.k_proj.weight"] = k_w
        if not k_eq_v:
            out[f"{pt_prefix}.attn.v_proj.weight"] = v_w

    if o_key in jax_params:
        out[f"{pt_prefix}.attn.o_proj.weight"] = _convert_attn_vec(jax_params[o_key])

    # QK norms — JAX uses query_norm/key_norm, we use q_norm/k_norm
    for jax_name, pt_name in [("query_norm", "q_norm"), ("key_norm", "k_norm")]:
        jax_norm = f"{jax_prefix}/attn/{jax_name}/scale"
        if jax_norm in jax_params:
            out[f"{pt_prefix}.attn.{pt_name}.weight"] = _convert_scale(jax_params[jax_norm])

    # --- Pre/post attention norms ---
    pre_attn = f"{jax_prefix}/pre_attention_norm/scale"
    if pre_attn in jax_params:
        out[f"{pt_prefix}.pre_attn_norm.weight"] = _convert_scale(jax_params[pre_attn])

    post_attn = f"{jax_prefix}/post_attention_norm/scale"
    if post_attn in jax_params:
        out[f"{pt_prefix}.post_attn_norm.weight"] = _convert_scale(jax_params[post_attn])

    # --- Feed-forward ---
    if has_moe:
        # MoE layers have separate handling
        # Pre/post FFW norms for MoE branch
        for jax_name, pt_name in [
            (f"{jax_prefix}/pre_ffw_norm/scale", f"{pt_prefix}.pre_ffw_norm.weight"),
            (f"{jax_prefix}/post_ffw1_norm/scale", f"{pt_prefix}.post_ffw1_norm.weight"),
            (f"{jax_prefix}/pre_ffw2_norm/scale", f"{pt_prefix}.pre_ffw2_norm.weight"),
            (f"{jax_prefix}/post_ffw2_norm/scale", f"{pt_prefix}.post_ffw2_norm.weight"),
            (f"{jax_prefix}/post_ffw_norm/scale", f"{pt_prefix}.post_ffw_norm.weight"),
        ]:
            if jax_name in jax_params:
                out[pt_name] = _convert_scale(jax_params[jax_name])
    else:
        # Dense MLP
        gate_key = f"{jax_prefix}/mlp/gating_einsum/w"
        linear_key = f"{jax_prefix}/mlp/linear/w"

        if gate_key in jax_params:
            out[f"{pt_prefix}.ffw.gate_up_proj.weight"] = _convert_gating_einsum(jax_params[gate_key])
        if linear_key in jax_params:
            out[f"{pt_prefix}.ffw.down_proj.weight"] = _convert_linear(jax_params[linear_key])

        # Pre/post FFW norms
        pre_ffw = f"{jax_prefix}/pre_ffw_norm/scale"
        if pre_ffw in jax_params:
            out[f"{pt_prefix}.pre_ffw_norm.weight"] = _convert_scale(jax_params[pre_ffw])
        post_ffw = f"{jax_prefix}/post_ffw_norm/scale"
        if post_ffw in jax_params:
            out[f"{pt_prefix}.post_ffw_norm.weight"] = _convert_scale(jax_params[post_ffw])

    # --- Skip scale ---
    skip_key = f"{jax_prefix}/skip_scale"
    if skip_key in jax_params:
        out[f"{pt_prefix}.skip_scale"] = torch.from_numpy(jax_params[skip_key].copy())

    # --- Per-layer input (PLI) ---
    pli_gate_key = f"{jax_prefix}/per_layer_input_gate/w"
    if pli_gate_key in jax_params:
        out[f"{pt_prefix}.pli_mapping.gate.weight"] = _convert_linear(jax_params[pli_gate_key])
    pli_proj_key = f"{jax_prefix}/per_layer_projection/w"
    if pli_proj_key in jax_params:
        out[f"{pt_prefix}.pli_mapping.proj.weight"] = _convert_linear(jax_params[pli_proj_key])
    pli_norm_key = f"{jax_prefix}/post_per_layer_input_norm/scale"
    if pli_norm_key in jax_params:
        out[f"{pt_prefix}.pli_mapping.norm.weight"] = _convert_scale(jax_params[pli_norm_key])

    return out


def _convert_vision_orbax(
        jax_params: dict,
        state_dict: dict[str, torch.Tensor],
        vision_cfg,
) -> None:
    """Map vision encoder JAX params to our naming.

    JAX checkpoint layout::

        vision_encoder/entry/input_projection/w      → patch_embedder.input_proj.weight
        vision_encoder/entry/pos_emb                  → patch_embedder.position_embedding_table
        vision_encoder/transformer/stacked_layers/block/...  → per-layer (stacked in dim 0)
        embedder/mm_input_projection/w                → embed_vision.proj.weight

    Vision layers are **stacked**: weights have an extra leading dimension
    ``num_layers`` that must be sliced per layer.
    """
    prefix = "vision_encoder"
    nl = vision_cfg.num_layers
    jax_block = "vision_encoder/transformer/stacked_layers/block"

    # --- Patch embedder ---
    proj_key = "vision_encoder/entry/input_projection/w"
    if proj_key in jax_params:
        state_dict[f"{prefix}.patch_embedder.input_proj.weight"] = _convert_linear(
            jax_params[proj_key]
        )
    pos_key = "vision_encoder/entry/pos_emb"
    if pos_key in jax_params:
        # JAX: (pos_size, 2, d_model) → our: (2, pos_size, d_model)
        w = jax_params[pos_key]
        state_dict[f"{prefix}.patch_embedder.position_embedding_table"] = torch.from_numpy(
            w.transpose(1, 0, 2).copy()
        )

    # --- Stacked layers (dim 0 = num_layers) ---
    # Attention: q_einsum, kv_einsum, attn_vec_einsum
    q_key = f"{jax_block}/attn/q_einsum/w"
    if q_key in jax_params:
        # Stacked (nl, N, D, H) → per-layer (N, D, H) → _convert_q_einsum → (N*H, D)
        stacked = jax_params[q_key]
        for i in range(nl):
            state_dict[f"{prefix}.layers.{i}.attn.q_proj.linear.weight"] = (
                _convert_q_einsum(stacked[i])
            )

    kv_key = f"{jax_block}/attn/kv_einsum/w"
    if kv_key in jax_params:
        # Stacked (nl, 2, K, D, H) → per-layer (2, K, D, H)
        stacked = jax_params[kv_key]
        for i in range(nl):
            k_w, v_w = _convert_kv_einsum(stacked[i])
            state_dict[f"{prefix}.layers.{i}.attn.k_proj.linear.weight"] = k_w
            state_dict[f"{prefix}.layers.{i}.attn.v_proj.linear.weight"] = v_w

    o_key = f"{jax_block}/attn/attn_vec_einsum/w"
    if o_key in jax_params:
        # Stacked (nl, N, H, D) → per-layer (N, H, D) → _convert_attn_vec → (D, N*H)
        stacked = jax_params[o_key]
        for i in range(nl):
            state_dict[f"{prefix}.layers.{i}.attn.o_proj.linear.weight"] = (
                _convert_attn_vec(stacked[i])
            )

    # QK norms — stacked (nl, head_dim)
    for jax_name, pt_name in [("query_norm", "q_norm"), ("key_norm", "k_norm")]:
        s_key = f"{jax_block}/attn/{jax_name}/scale"
        if s_key in jax_params:
            stacked = jax_params[s_key]
            for i in range(nl):
                state_dict[f"{prefix}.layers.{i}.attn.{pt_name}.weight"] = (
                    _convert_scale(stacked[i])
                )

    # MLP: gating_einsum (fused gate+up) → separate gate_proj, up_proj
    gate_key = f"{jax_block}/mlp/gating_einsum/w"
    if gate_key in jax_params:
        # Stacked (nl, 2, ffw_hidden, d_model) → per-layer (2, F, D)
        # Our vision MLP has separate gate_proj, up_proj (not fused)
        stacked = jax_params[gate_key]
        for i in range(nl):
            gate_w = stacked[i][0]  # (F, D) — no transpose needed
            up_w = stacked[i][1]    # (F, D)
            state_dict[f"{prefix}.layers.{i}.mlp.gate_proj.linear.weight"] = (
                torch.from_numpy(gate_w.copy())
            )
            state_dict[f"{prefix}.layers.{i}.mlp.up_proj.linear.weight"] = (
                torch.from_numpy(up_w.copy())
            )

    linear_key = f"{jax_block}/mlp/linear/w"
    if linear_key in jax_params:
        # Stacked (nl, F, D) — down projection, needs transpose
        stacked = jax_params[linear_key]
        for i in range(nl):
            state_dict[f"{prefix}.layers.{i}.mlp.down_proj.linear.weight"] = (
                _convert_linear(stacked[i])
            )

    # Layer norms — stacked (nl, d_model)
    for jax_name, pt_name in [
        ("pre_attention_norm/scale", "pre_attn_norm.weight"),
        ("post_attention_norm/scale", "post_attn_norm.weight"),
        ("pre_ffw_norm/scale", "pre_ffw_norm.weight"),
        ("post_ffw_norm/scale", "post_ffw_norm.weight"),
    ]:
        s_key = f"{jax_block}/{jax_name}"
        if s_key in jax_params:
            stacked = jax_params[s_key]
            for i in range(nl):
                state_dict[f"{prefix}.layers.{i}.{pt_name}"] = _convert_scale(stacked[i])

    # ClippedLinear bounds — stacked (nl,) scalars
    # kv_einsum bounds apply to both k_proj and v_proj; gating_einsum bounds
    # apply to both gate_proj and up_proj.
    _clip_map = [
        ("attn/q_einsum", ["attn.q_proj"]),
        ("attn/kv_einsum", ["attn.k_proj", "attn.v_proj"]),
        ("attn/attn_vec_einsum", ["attn.o_proj"]),
        ("mlp/gating_einsum", ["mlp.gate_proj", "mlp.up_proj"]),
        ("mlp/linear", ["mlp.down_proj"]),
    ]
    buf_map = {
        "clip_input_max": "input_max",
        "clip_input_min": "input_min",
        "clip_output_max": "output_max",
        "clip_output_min": "output_min",
    }
    for proj_path, pt_projs in _clip_map:
        for clip_name, buf_name in buf_map.items():
            c_key = f"{jax_block}/{proj_path}/{clip_name}"
            if c_key in jax_params:
                stacked = jax_params[c_key]
                for i in range(nl):
                    for pt_proj in pt_projs:
                        state_dict[f"{prefix}.layers.{i}.{pt_proj}.{buf_name}"] = (
                            torch.tensor(float(stacked[i]))
                        )

    # Standardize buffers
    for buf_name in ("std_bias", "std_scale"):
        key = f"vision_encoder/{buf_name}"
        if key in jax_params:
            state_dict[f"{prefix}.{buf_name}"] = torch.from_numpy(jax_params[key].copy())

    # Vision embedder projection (in embedder, not vision_encoder)
    proj_key = "embedder/mm_input_projection/w"
    if proj_key in jax_params:
        state_dict["embed_vision.proj.weight"] = _convert_linear(jax_params[proj_key])


def _convert_audio_orbax(
        jax_params: dict,
        state_dict: dict[str, torch.Tensor],
        audio_cfg,
) -> None:
    """Map audio encoder JAX params to our naming.

    JAX checkpoint layout::

        audio_encoder/feature/subsampling_0/kernel   → subsample.conv1.weight
        audio_encoder/feature/subsampling_1/kernel   → subsample.conv2.weight
        audio_encoder/feature/norm_0/scale           → subsample.norm1.weight
        audio_encoder/feature/norm_1/scale           → subsample.norm2.weight
        audio_encoder/feature/input_proj/kernel       → subsample.proj.weight
        audio_encoder/conformer/stacked_layers_N/... → conformer.N.{submodule}
        audio_encoder/output_projection/{kernel,bias} → output_proj.{weight,bias}
    """
    prefix = "audio_encoder"

    # --- SubSampling ---
    _sub_map = {
        "audio_encoder/feature/subsampling_0/kernel": f"{prefix}.subsample.conv1.weight",
        "audio_encoder/feature/subsampling_1/kernel": f"{prefix}.subsample.conv2.weight",
        "audio_encoder/feature/input_proj/kernel": f"{prefix}.subsample.proj.weight",
    }
    for jax_k, pt_k in _sub_map.items():
        if jax_k in jax_params:
            w = jax_params[jax_k]
            if w.ndim == 4:
                # Conv2d: JAX (H,W,Cin,Cout) → PyTorch (Cout,Cin,H,W)
                state_dict[pt_k] = torch.from_numpy(w.transpose(3, 2, 0, 1).copy())
            elif w.ndim == 3:
                # input_proj: JAX (c2, f2, D) → reshape to (c2*f2, D) → Linear
                w_flat = w.reshape(-1, w.shape[-1])
                state_dict[pt_k] = _convert_linear(w_flat)
            elif w.ndim == 2:
                state_dict[pt_k] = _convert_linear(w)
            else:
                state_dict[pt_k] = torch.from_numpy(w.copy())

    # SubSampling norms (LayerNorm weight, no bias)
    for jax_norm, pt_norm in [
        ("audio_encoder/feature/norm_0/scale", f"{prefix}.subsample.norm1.weight"),
        ("audio_encoder/feature/norm_1/scale", f"{prefix}.subsample.norm2.weight"),
    ]:
        if jax_norm in jax_params:
            state_dict[pt_norm] = _convert_scale(jax_params[jax_norm])

    # --- Conformer layers ---
    for i in range(audio_cfg.num_layers):
        jax_pfx = f"audio_encoder/conformer/stacked_layers_{i}"
        pt_pfx = f"{prefix}.conformer.{i}"

        # FFN blocks (fflayer_start → ffw_start, fflayer_end → ffw_end)
        for jax_ffw, pt_ffw in [("fflayer_start", "ffw_start"), ("fflayer_end", "ffw_end")]:
            # up/down projections (ClippedLinear → .linear.weight)
            for jax_proj, pt_proj in [("ffn_layer1/kernel", "up.linear.weight"), ("ffn_layer2/kernel", "down.linear.weight")]:
                k = f"{jax_pfx}/{jax_ffw}/{jax_proj}"
                if k in jax_params:
                    state_dict[f"{pt_pfx}.{pt_ffw}.{pt_proj}"] = _convert_linear(jax_params[k])
            # Norms
            for jax_n, pt_n in [("pre_layer_norm/scale", "pre_norm.weight"), ("post_layer_norm/scale", "post_norm.weight")]:
                k = f"{jax_pfx}/{jax_ffw}/{jax_n}"
                if k in jax_params:
                    state_dict[f"{pt_pfx}.{pt_ffw}.{pt_n}"] = _convert_scale(jax_params[k])

        # Attention (trans_atten → attn)
        attn_pfx = f"{jax_pfx}/trans_atten"
        # Q/K/V projections (ClippedLinear → .linear.weight)
        for jax_proj, pt_proj in [("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")]:
            k = f"{attn_pfx}/self_atten/{jax_proj}/kernel"
            if k in jax_params:
                state_dict[f"{pt_pfx}.attn.attn.{pt_proj}.linear.weight"] = _convert_linear(jax_params[k])
        # per_dim_scale
        k = f"{attn_pfx}/self_atten/per_dim_scale"
        if k in jax_params:
            state_dict[f"{pt_pfx}.attn.attn.per_dim_scale"] = _convert_scale(jax_params[k])
        # Relative position projection — JAX kernel is (D, N, H) → reshape to (D, N*H) then transpose
        k = f"{attn_pfx}/self_atten/relative_position_embedding/pos_proj/kernel"
        if k in jax_params:
            w = jax_params[k]
            if w.ndim == 3:
                D, N_h, H_h = w.shape
                w = w.reshape(D, N_h * H_h)
            state_dict[f"{pt_pfx}.attn.attn.rel_pos_emb.pos_proj.weight"] = _convert_linear(w)
        # O projection — JAX kernel is (N, H, D), same as attn_vec_einsum (ClippedLinear → .linear.weight)
        k = f"{attn_pfx}/post/kernel"
        if k in jax_params:
            w = jax_params[k]
            if w.ndim == 3:
                state_dict[f"{pt_pfx}.attn.o_proj.linear.weight"] = _convert_attn_vec(w)
            else:
                state_dict[f"{pt_pfx}.attn.o_proj.linear.weight"] = _convert_linear(w)
        # Pre/post norms
        for jax_n, pt_n in [("pre_norm/scale", "attn.pre_norm.weight"), ("post_norm/scale", "attn.post_norm.weight")]:
            k = f"{attn_pfx}/{jax_n}"
            if k in jax_params:
                state_dict[f"{pt_pfx}.{pt_n}"] = _convert_scale(jax_params[k])

        # Lightweight conv (lconv)
        lconv_pfx = f"{jax_pfx}/lconv"
        for jax_k, pt_k in [
            ("ln/scale", "lconv.pre_norm.weight"),
            ("linear_start/kernel", "lconv.linear_start.linear.weight"),
            ("conv_norm/scale", "lconv.conv_norm.weight"),
            ("linear_end/kernel", "lconv.linear_end.linear.weight"),
        ]:
            k = f"{lconv_pfx}/{jax_k}"
            if k in jax_params:
                w = jax_params[k]
                if "kernel" in jax_k and w.ndim == 2:
                    state_dict[f"{pt_pfx}.{pt_k}"] = _convert_linear(w)
                else:
                    state_dict[f"{pt_pfx}.{pt_k}"] = _convert_scale(w)

        # Depthwise conv1d: JAX (W, Cin, Cout) → PyTorch (Cout, 1, W) for groups=Cin
        dwconv_key = f"{lconv_pfx}/depthwise_conv1d/kernel"
        if dwconv_key in jax_params:
            w = jax_params[dwconv_key]  # (W, Cin, 1) for depthwise
            state_dict[f"{pt_pfx}.lconv.dwconv.weight"] = torch.from_numpy(
                w.transpose(2, 1, 0).copy()
            )

        # Final layer norm
        k = f"{jax_pfx}/final_ln/scale"
        if k in jax_params:
            state_dict[f"{pt_pfx}.norm.weight"] = _convert_scale(jax_params[k])

        # ClippedLinear bounds for all clipped projections
        _audio_clip_map = [
            # FFN
            ("fflayer_start/ffn_layer1", ["ffw_start.up"]),
            ("fflayer_start/ffn_layer2", ["ffw_start.down"]),
            ("fflayer_end/ffn_layer1", ["ffw_end.up"]),
            ("fflayer_end/ffn_layer2", ["ffw_end.down"]),
            # Attention Q/K/V
            ("trans_atten/self_atten/query", ["attn.attn.q_proj"]),
            ("trans_atten/self_atten/key", ["attn.attn.k_proj"]),
            ("trans_atten/self_atten/value", ["attn.attn.v_proj"]),
            # Attention O
            ("trans_atten/post", ["attn.o_proj"]),
            # LConv
            ("lconv/linear_start", ["lconv.linear_start"]),
            ("lconv/linear_end", ["lconv.linear_end"]),
        ]
        _audio_buf_map = {
            "clip_input_max": "input_max",
            "clip_input_min": "input_min",
            "clip_output_max": "output_max",
            "clip_output_min": "output_min",
        }
        for proj_path, pt_projs in _audio_clip_map:
            for clip_name, buf_name in _audio_buf_map.items():
                c_key = f"{jax_pfx}/{proj_path}/{clip_name}"
                if c_key in jax_params:
                    for pt_proj in pt_projs:
                        state_dict[f"{pt_pfx}.{pt_proj}.{buf_name}"] = (
                            torch.tensor(float(jax_params[c_key]))
                        )

    # --- Output projection ---
    out_key = "audio_encoder/output_projection/kernel"
    if out_key in jax_params:
        state_dict[f"{prefix}.output_proj.weight"] = _convert_linear(jax_params[out_key])
    out_bias = "audio_encoder/output_projection/bias"
    if out_bias in jax_params:
        state_dict[f"{prefix}.output_proj.bias"] = _convert_scale(jax_params[out_bias])

    # Audio embedder projection (in embedder module, not audio_encoder)
    audio_proj_key = "embedder/audio_input_projection/w"
    if audio_proj_key in jax_params:
        state_dict["embed_audio.proj.weight"] = _convert_linear(jax_params[audio_proj_key])


def convert_orbax(
        checkpoint_path: str,
        variant: str,
        output_path: str,
) -> None:
    """Convert an Orbax checkpoint to our safetensors format.

    Args:
        checkpoint_path: Path to Orbax checkpoint directory.
        variant: One of 'e2b', 'e4b', '31b', '26b_a4b'.
        output_path: Output ``.safetensors`` file path.
    """
    if variant not in _VARIANT_FACTORIES:
        raise ValueError(
            f"Unknown variant: {variant}. Choose from: {list(_VARIANT_FACTORIES)}"
        )

    factory = _VARIANT_FACTORIES[variant]
    model = factory(text_only=False)
    cfg = model.cfg.text

    attn_types = make_attention_pattern(cfg.attention_pattern, cfg.num_layers)
    has_moe = cfg.moe is not None

    print(f"Loading Orbax checkpoint from {checkpoint_path}...")
    jax_params = _load_orbax_checkpoint(checkpoint_path)

    print(f"Converting {variant} ({cfg.num_layers} layers)...")
    state_dict: dict[str, torch.Tensor] = {}

    # Embedder
    emb_key = "embedder/input_embedding"
    if emb_key in jax_params:
        state_dict["text_decoder.embedder.token_embedding.weight"] = torch.from_numpy(
            jax_params[emb_key].copy()
        )

    # PLI embedder weights
    # JAX: (vocab, num_layers, pli_dim) → our nn.Embedding: (vocab, num_layers * pli_dim)
    pli_emb_key = "embedder/per_layer_embeddings"
    if pli_emb_key in jax_params:
        w = jax_params[pli_emb_key]  # (vocab, num_layers, pli_dim)
        state_dict["text_decoder.embedder.pli_embedding.weight"] = torch.from_numpy(
            w.reshape(w.shape[0], -1).copy()
        )
    # JAX: (embed, num_layers, pli_dim) → our nn.Linear: (num_layers*pli_dim, embed)
    pli_proj_key = "embedder/per_layer_model_projection/w"
    if pli_proj_key in jax_params:
        w = jax_params[pli_proj_key]  # (embed, num_layers, pli_dim)
        w_flat = w.reshape(w.shape[0], -1)  # (embed, num_layers*pli_dim)
        state_dict["text_decoder.embedder.pli_proj.weight"] = _convert_linear(w_flat)
    pli_proj_norm_key = "embedder/per_layer_projection_norm/scale"
    if pli_proj_norm_key in jax_params:
        state_dict["text_decoder.embedder.pli_proj_norm.weight"] = (
            _convert_scale(jax_params[pli_proj_norm_key])
        )

    # Per-layer blocks
    for i in range(cfg.num_layers):
        layer_params = _map_text_layer(
            jax_params, i, attn_types[i],
            has_moe=has_moe,
            k_eq_v=cfg.k_eq_v,
        )
        state_dict.update(layer_params)

    # Final norm
    final_norm_key = "final_norm/scale"
    if final_norm_key in jax_params:
        state_dict["text_decoder.final_norm.weight"] = _convert_scale(jax_params[final_norm_key])

    # --- Vision encoder ---
    vision_cfg = model.cfg.vision
    if vision_cfg is not None:
        _convert_vision_orbax(jax_params, state_dict, vision_cfg)

    # --- Audio encoder ---
    audio_cfg = model.cfg.audio
    if audio_cfg is not None:
        _convert_audio_orbax(jax_params, state_dict, audio_cfg)

    # Save — ensure all tensors are contiguous (stacked layer slicing can
    # produce views that safetensors rejects)
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    print(f"Saving {len(state_dict)} tensors to {output_path}...")
    save_file(state_dict, output_path)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Orbax (JAX) checkpoint to safetensors for gemma4-pytorch"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to Orbax checkpoint directory"
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=list(_VARIANT_FACTORIES),
        help="Model variant",
    )
    parser.add_argument(
        "--output", required=True, help="Output .safetensors file path"
    )
    args = parser.parse_args()
    convert_orbax(args.checkpoint, args.variant, args.output)


if __name__ == "__main__":
    main()
