"""Convert Orbax (JAX) checkpoint → safetensors for our PyTorch model.

Usage::

    # As CLI entry point:
    gemma4-convert --checkpoint /path/to/orbax --variant e2b --output model.safetensors

    # As Python API:
    from gemma4_pytorch_claude.convert import convert_orbax
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
    """``(2, D, F)`` → ``(2*F, D)`` fused gate+up weight."""
    return torch.from_numpy(w.reshape(2 * w.shape[1], w.shape[2]).T.copy())


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
            "Install with: pip install 'gemma4-pytorch[convert]'"
        )

    ckpt_path = Path(checkpoint_path)
    checkpointer = ocp.PyTreeCheckpointer()
    raw = checkpointer.restore(ckpt_path)

    # Flatten the nested dict to dot-separated keys
    flat = {}

    def _flatten(d: dict, prefix: str = ""):
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                flat[key] = np.array(v)

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

    # QK norms
    for norm_name in ("q_norm", "k_norm"):
        jax_norm = f"{jax_prefix}/attn/{norm_name}/scale"
        if jax_norm in jax_params:
            out[f"{pt_prefix}.attn.{norm_name}.scale"] = _convert_scale(jax_params[jax_norm])

    # --- Pre/post attention norms ---
    pre_attn = f"{jax_prefix}/pre_attention_norm/scale"
    if pre_attn in jax_params:
        out[f"{pt_prefix}.pre_attn_norm.scale"] = _convert_scale(jax_params[pre_attn])

    post_attn = f"{jax_prefix}/post_attention_norm/scale"
    if post_attn in jax_params:
        out[f"{pt_prefix}.post_attn_norm.scale"] = _convert_scale(jax_params[post_attn])

    # --- Feed-forward ---
    if has_moe:
        # MoE layers have separate handling
        # Pre/post FFW norms for MoE branch
        for jax_name, pt_name in [
            (f"{jax_prefix}/pre_ffw_norm/scale", f"{pt_prefix}.pre_ffw_norm.scale"),
            (f"{jax_prefix}/post_ffw1_norm/scale", f"{pt_prefix}.post_ffw1_norm.scale"),
            (f"{jax_prefix}/pre_ffw2_norm/scale", f"{pt_prefix}.pre_ffw2_norm.scale"),
            (f"{jax_prefix}/post_ffw2_norm/scale", f"{pt_prefix}.post_ffw2_norm.scale"),
            (f"{jax_prefix}/post_ffw_norm/scale", f"{pt_prefix}.post_ffw_norm.scale"),
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
            out[f"{pt_prefix}.pre_ffw_norm.scale"] = _convert_scale(jax_params[pre_ffw])
        post_ffw = f"{jax_prefix}/post_ffw_norm/scale"
        if post_ffw in jax_params:
            out[f"{pt_prefix}.post_ffw_norm.scale"] = _convert_scale(jax_params[post_ffw])

    # --- Skip scale ---
    skip_key = f"{jax_prefix}/skip_scale"
    if skip_key in jax_params:
        out[f"{pt_prefix}.skip_scale"] = torch.from_numpy(jax_params[skip_key].copy())

    return out


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
        raise ValueError(f"Unknown variant: {variant}. Choose from: {list(_VARIANT_FACTORIES)}")

    factory = _VARIANT_FACTORIES[variant]
    model = factory(text_only=True)
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
    pli_emb_key = "embedder/pli_embedding"
    if pli_emb_key in jax_params:
        state_dict["text_decoder.embedder.pli_embedding.weight"] = torch.from_numpy(
            jax_params[pli_emb_key].copy()
        )
    pli_proj_key = "embedder/pli_proj/w"
    if pli_proj_key in jax_params:
        state_dict["text_decoder.embedder.pli_proj.weight"] = _convert_linear(jax_params[pli_proj_key])
    pli_proj_norm_key = "embedder/pli_proj_norm/scale"
    if pli_proj_norm_key in jax_params:
        state_dict["text_decoder.embedder.pli_proj_norm.scale"] = _convert_scale(jax_params[pli_proj_norm_key])

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
        state_dict["text_decoder.final_norm.scale"] = _convert_scale(jax_params[final_norm_key])

    # Save
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
