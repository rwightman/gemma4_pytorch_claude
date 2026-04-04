"""Load weights into a Gemma4Model from safetensors (our format or HuggingFace).

Usage::

    from gemma4_pt_claude import gemma4_e2b
    from gemma4_pt_claude.load import load_weights

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

def _hf_key_to_ours(hf_key: str, num_layers: int, has_vision: bool = False) -> str | None:
    """Map a single HF key to our naming convention.

    Returns None if the key should be skipped.
    Keys that need special merge handling (gate_proj, up_proj) return None
    and are handled in ``_hf_convert_weights``.
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

    # --- Global PLI (embedder-level) ---
    _global_pli_map = {
        "embed_tokens_per_layer.weight": "text_decoder.embedder.pli_embedding.weight",
        "per_layer_model_projection.weight": "text_decoder.embedder.pli_proj.weight",
        "per_layer_projection_norm.weight": "text_decoder.embedder.pli_proj_norm.scale",
    }
    if k in _global_pli_map:
        return _global_pli_map[k]

    # --- Multimodal embedder (vision) ---
    if has_vision:
        if k == "embed_vision.embedding_projection.weight":
            return "embed_vision.proj.weight"
        if k == "embed_vision.embedding_pre_projection_norm.weight":
            return None  # with_scale=False → no weight parameter

    # --- Vision tower ---
    if k.startswith("vision_tower.") and has_vision:
        return _hf_vision_key_to_ours(k)

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

        # MLP — fused gate_up_proj (our format) or separate (handled by merge)
        _mlp_map = {
            "mlp.gate_up_proj.weight": "ffw.gate_up_proj.weight",
            "mlp.down_proj.weight": "ffw.down_proj.weight",
        }
        if rest in _mlp_map:
            return f"{prefix}.{_mlp_map[rest]}"

        # MLP — separate gate_proj / up_proj need merge (handled in _hf_convert_weights)
        if rest in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            return None  # handled specially

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

        # Per-layer input (PLI) mappings
        _pli_map = {
            "per_layer_input_gate.weight": "pli_mapping.gate.weight",
            "per_layer_projection.weight": "pli_mapping.proj.weight",
            "post_per_layer_input_norm.weight": "pli_mapping.norm.scale",
        }
        if rest in _pli_map:
            return f"{prefix}.{_pli_map[rest]}"

    return None


def _hf_vision_key_to_ours(k: str) -> str | None:
    """Map a single HF vision_tower key (prefix already stripped to ``vision_tower.``)."""
    # --- Patch embedder ---
    if k == "vision_tower.patch_embedder.input_proj.weight":
        return "vision_encoder.patch_embedder.input_proj.weight"
    if k == "vision_tower.patch_embedder.position_embedding_table":
        return "vision_encoder.patch_embedder.position_embedding_table"

    # --- Encoder rotary_emb (skip — recomputed) ---
    if "rotary_emb" in k:
        return None

    # --- Encoder layers ---
    if k.startswith("vision_tower.encoder.layers."):
        # vision_tower.encoder.layers.{i}.{rest}
        parts = k.split(".", 4)  # ["vision_tower", "encoder", "layers", "N", "rest"]
        if len(parts) < 5:
            return None
        layer_idx = parts[3]
        rest = parts[4]
        prefix = f"vision_encoder.layers.{layer_idx}"

        # Attention projs — ClippedLinear subkeys: linear.weight, input_min, etc.
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            attn_prefix = f"self_attn.{proj_name}."
            if rest.startswith(attn_prefix):
                suffix = rest[len(attn_prefix):]
                return f"{prefix}.attn.{proj_name}.{suffix}"

        # QK/V norms
        _norm_attn_map = {
            "self_attn.q_norm.weight": f"{prefix}.attn.q_norm.scale",
            "self_attn.k_norm.weight": f"{prefix}.attn.k_norm.scale",
        }
        if rest in _norm_attn_map:
            return _norm_attn_map[rest]

        # MLP projs — ClippedLinear subkeys
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            mlp_prefix = f"mlp.{proj_name}."
            if rest.startswith(mlp_prefix):
                suffix = rest[len(mlp_prefix):]
                return f"{prefix}.mlp.{proj_name}.{suffix}"

        # Layer norms
        _norm_map = {
            "input_layernorm.weight": f"{prefix}.pre_attn_norm.scale",
            "post_attention_layernorm.weight": f"{prefix}.post_attn_norm.scale",
            "pre_feedforward_layernorm.weight": f"{prefix}.pre_ffw_norm.scale",
            "post_feedforward_layernorm.weight": f"{prefix}.post_ffw_norm.scale",
        }
        if rest in _norm_map:
            return _norm_map[rest]

    return None


def _hf_convert_weights(
        raw: dict[str, torch.Tensor],
        num_layers: int,
        has_vision: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert HF safetensors keys to our naming, merging gate/up projections."""
    mapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    # Collect gate_proj and up_proj for merging (text decoder only)
    gate_projs: dict[str, torch.Tensor] = {}
    up_projs: dict[str, torch.Tensor] = {}

    for hf_key, tensor in raw.items():
        # Check for text decoder gate_proj / up_proj that need merging
        k = hf_key
        if k.startswith("model.language_model."):
            k = k[len("model.language_model."):]
        elif k.startswith("model."):
            k = k[len("model."):]

        # Only merge text decoder MLP gate/up; vision MLPs stay separate
        if k.startswith("layers.") and k.endswith(".mlp.gate_proj.weight"):
            layer_idx = k.split(".")[1]
            gate_projs[layer_idx] = tensor
            continue
        if k.startswith("layers.") and k.endswith(".mlp.up_proj.weight"):
            layer_idx = k.split(".")[1]
            up_projs[layer_idx] = tensor
            continue

        our_key = _hf_key_to_ours(hf_key, num_layers, has_vision=has_vision)
        if our_key is not None:
            mapped[our_key] = tensor
        else:
            skipped.append(hf_key)

    # Merge gate_proj + up_proj → gate_up_proj (text decoder)
    for layer_idx in gate_projs:
        gate = gate_projs[layer_idx]
        up = up_projs.get(layer_idx)
        if up is None:
            skipped.append(f"layers.{layer_idx}.mlp.gate_proj.weight (missing up_proj)")
            continue
        fused = torch.cat([gate, up], dim=0)
        our_key = f"text_decoder.blocks.{layer_idx}.ffw.gate_up_proj.weight"
        mapped[our_key] = fused

    if skipped:
        print(f"Skipped {len(skipped)} HF keys: {skipped[:10]}...")

    return mapped


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
        has_vision = model.vision_encoder is not None
        raw = _hf_convert_weights(raw, num_layers, has_vision=has_vision)

    # Optional dtype cast
    if dtype is not None:
        raw = {k: v.to(dtype) for k, v in raw.items()}

    result = model.load_state_dict(raw, strict=strict)
    missing = list(result.missing_keys) if hasattr(result, "missing_keys") else []
    unexpected = list(result.unexpected_keys) if hasattr(result, "unexpected_keys") else []
    return missing, unexpected
