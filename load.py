"""Load weights into a Gemma4Model from safetensors (our format or HuggingFace).

Usage::

    from gemma4_pytorch_claude import gemma4_e2b
    from gemma4_pytorch_claude.load import load_weights

    model = gemma4_e2b(text_only=True)

    # From our converted safetensors:
    load_weights(model, "model.safetensors")

    # From HuggingFace safetensors directory:
    load_weights(model, "/path/to/hf/model", format="hf")
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from .model import Gemma4Model


# ---------------------------------------------------------------------------
# HuggingFace → our naming
# ---------------------------------------------------------------------------

def _hf_key_to_ours(hf_key: str, num_layers: int) -> str | None:
    """Map a single HF key to our naming convention.

    Returns None if the key should be skipped (not part of text decoder).
    """
    # Strip leading "model.language_model." or "model." prefix
    k = hf_key
    if k.startswith("model.language_model."):
        k = k[len("model.language_model."):]
    elif k.startswith("model."):
        k = k[len("model."):]

    # --- Embedding ---
    if k == "embed_tokens.weight":
        return "text_decoder.embedder.token_embedding.weight"

    # --- Final norm ---
    if k == "norm.weight":
        return "text_decoder.final_norm.scale"

    # --- Layer params ---
    if k.startswith("layers."):
        parts = k.split(".", 2)  # ["layers", "N", "rest"]
        if len(parts) < 3:
            return None
        layer_idx = parts[1]
        rest = parts[2]
        prefix = f"text_decoder.blocks.{layer_idx}"

        # Attention projections
        _attn_map = {
            "self_attn.q_proj.weight": "attn.q_proj.weight",
            "self_attn.k_proj.weight": "attn.k_proj.weight",
            "self_attn.v_proj.weight": "attn.v_proj.weight",
            "self_attn.o_proj.weight": "attn.o_proj.weight",
            "self_attn.q_norm.weight": "attn.q_norm.scale",
            "self_attn.k_norm.weight": "attn.k_norm.scale",
        }
        if rest in _attn_map:
            return f"{prefix}.{_attn_map[rest]}"

        # MLP
        _mlp_map = {
            "mlp.gate_up_proj.weight": "ffw.gate_up_proj.weight",
            "mlp.down_proj.weight": "ffw.down_proj.weight",
        }
        if rest in _mlp_map:
            return f"{prefix}.{_mlp_map[rest]}"

        # Norms
        _norm_map = {
            "input_layernorm.weight": "pre_attn_norm.scale",
            "post_attention_layernorm.weight": "post_attn_norm.scale",
            "pre_feedforward_layernorm.weight": "pre_ffw_norm.scale",
            "post_feedforward_layernorm.weight": "post_ffw_norm.scale",
        }
        if rest in _norm_map:
            return f"{prefix}.{_norm_map[rest]}"

        # Skip scale (scalar)
        if rest == "layer_scalar":
            return f"{prefix}.skip_scale"

        # PLI mapping
        _pli_map = {
            "per_layer_input.gate.weight": "pli_mapping.gate.weight",
            "per_layer_input.proj.weight": "pli_mapping.proj.weight",
            "per_layer_input.norm.weight": "pli_mapping.norm.scale",
        }
        if rest in _pli_map:
            return f"{prefix}.{_pli_map[rest]}"

    return None


def _load_safetensors_files(path: str | Path) -> dict[str, torch.Tensor]:
    """Load tensors from one or more safetensors files."""
    p = Path(path)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {p}")
    else:
        raise FileNotFoundError(f"Path does not exist: {p}")

    tensors: dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                tensors[key] = sf.get_tensor(key)
    return tensors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_weights(
    model: Gemma4Model,
    path: str | Path,
    *,
    format: str = "auto",
    strict: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[list[str], list[str]]:
    """Load weights into a model from safetensors.

    Args:
        model: A ``Gemma4Model`` instance.
        path: Path to a ``.safetensors`` file, or a directory containing them.
        format: ``"auto"`` (detect), ``"ours"`` (our naming), or ``"hf"``
            (HuggingFace naming).
        strict: If True, raise on missing/unexpected keys.
        dtype: Cast all loaded tensors to this dtype (e.g. ``torch.bfloat16``).

    Returns:
        (missing_keys, unexpected_keys) from ``load_state_dict``.
    """
    raw = _load_safetensors_files(path)

    # Auto-detect format
    if format == "auto":
        sample_key = next(iter(raw))
        if sample_key.startswith("model.") or sample_key.startswith("language_model."):
            format = "hf"
        else:
            format = "ours"

    if format == "hf":
        num_layers = model.cfg.text.num_layers
        mapped: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for hf_key, tensor in raw.items():
            our_key = _hf_key_to_ours(hf_key, num_layers)
            if our_key is not None:
                mapped[our_key] = tensor
            else:
                skipped.append(hf_key)
        if skipped:
            print(f"Skipped {len(skipped)} HF keys not in text decoder: {skipped[:5]}...")
        raw = mapped

    # Optional dtype cast
    if dtype is not None:
        raw = {k: v.to(dtype) for k, v in raw.items()}

    result = model.load_state_dict(raw, strict=strict)
    missing = list(result.missing_keys) if hasattr(result, "missing_keys") else []
    unexpected = list(result.unexpected_keys) if hasattr(result, "unexpected_keys") else []
    return missing, unexpected
