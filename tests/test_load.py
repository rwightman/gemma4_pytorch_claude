"""Unit tests for weight loading."""

import tempfile
from pathlib import Path

import torch
import pytest
from safetensors.torch import save_file

from gemma4.config import AttentionType, Gemma4Config, TextConfig
from gemma4.model import Gemma4Model
from gemma4.load import load_weights, _hf_key_to_ours, _hf_convert_weights


class TestHFKeyMapping:
    def test_embed_tokens(self):
        assert _hf_key_to_ours("model.language_model.embed_tokens.weight", 4) == (
            "text_decoder.embedder.token_embedding.weight"
        )

    def test_final_norm(self):
        assert _hf_key_to_ours("model.language_model.norm.weight", 4) == (
            "text_decoder.final_norm.scale"
        )

    def test_attn_proj(self):
        result = _hf_key_to_ours(
            "model.language_model.layers.3.self_attn.q_proj.weight", 4
        )
        assert result == "text_decoder.blocks.3.attn.q_proj.weight"

    def test_mlp_fused(self):
        """Our fused gate_up_proj format is directly mapped."""
        result = _hf_key_to_ours(
            "model.language_model.layers.0.mlp.gate_up_proj.weight", 4
        )
        assert result == "text_decoder.blocks.0.ffw.gate_up_proj.weight"

    def test_mlp_separate_returns_none(self):
        """Separate gate/up are handled by _hf_convert_weights, not _hf_key_to_ours."""
        assert _hf_key_to_ours(
            "model.language_model.layers.0.mlp.gate_proj.weight", 4
        ) is None
        assert _hf_key_to_ours(
            "model.language_model.layers.0.mlp.up_proj.weight", 4
        ) is None

    def test_norms(self):
        result = _hf_key_to_ours(
            "model.language_model.layers.1.input_layernorm.weight", 4
        )
        assert result == "text_decoder.blocks.1.pre_attn_norm.scale"

    def test_qk_norm(self):
        result = _hf_key_to_ours(
            "model.language_model.layers.0.self_attn.q_norm.weight", 4
        )
        assert result == "text_decoder.blocks.0.attn.q_norm.scale"

    def test_skip_scale(self):
        result = _hf_key_to_ours(
            "model.language_model.layers.2.layer_scalar", 4
        )
        assert result == "text_decoder.blocks.2.skip_scale"

    def test_unknown_key_returns_none(self):
        assert _hf_key_to_ours("model.some_unknown_thing", 4) is None

    def test_model_prefix_stripped(self):
        """Also works with just 'model.' prefix (no 'language_model.')."""
        result = _hf_key_to_ours("model.embed_tokens.weight", 4)
        assert result == "text_decoder.embedder.token_embedding.weight"

    def test_pli_embedder_keys(self):
        assert _hf_key_to_ours(
            "model.language_model.embed_tokens_per_layer.weight", 4
        ) == "text_decoder.embedder.pli_embedding.weight"
        assert _hf_key_to_ours(
            "model.language_model.per_layer_model_projection.weight", 4
        ) == "text_decoder.embedder.pli_proj.weight"
        assert _hf_key_to_ours(
            "model.language_model.per_layer_projection_norm.weight", 4
        ) == "text_decoder.embedder.pli_proj_norm.scale"

    def test_pli_layer_keys(self):
        assert _hf_key_to_ours(
            "model.language_model.layers.0.per_layer_input_gate.weight", 4
        ) == "text_decoder.blocks.0.pli_mapping.gate.weight"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.per_layer_projection.weight", 4
        ) == "text_decoder.blocks.0.pli_mapping.proj.weight"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.post_per_layer_input_norm.weight", 4
        ) == "text_decoder.blocks.0.pli_mapping.norm.scale"


class TestHFConvertWeights:
    def test_gate_up_merge(self):
        """gate_proj + up_proj are concatenated into gate_up_proj."""
        raw = {
            "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(4, 2),
            "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(4, 2),
            "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(2, 4),
        }
        mapped = _hf_convert_weights(raw, num_layers=1)
        fused = mapped["text_decoder.blocks.0.ffw.gate_up_proj.weight"]
        assert fused.shape == (8, 2)
        assert torch.equal(fused[:4], raw["model.language_model.layers.0.mlp.gate_proj.weight"])
        assert torch.equal(fused[4:], raw["model.language_model.layers.0.mlp.up_proj.weight"])


class TestLoadWeights:
    @pytest.fixture
    def tiny_model(self):
        cfg = Gemma4Config(
            text=TextConfig(
                vocab_size=32,
                embed_dim=16,
                hidden_dim=32,
                num_heads=2,
                head_dim=8,
                num_kv_heads=2,
                num_layers=1,
                sliding_window_size=8,
                attention_pattern=(AttentionType.GLOBAL,),
                use_qk_norm=False,
                use_value_norm=False,
                use_post_attn_norm=False,
                use_post_ffw_norm=False,
            ),
        )
        return Gemma4Model(cfg)

    def test_load_our_format(self, tiny_model):
        """Save and reload in our native format."""
        state = tiny_model.state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            save_file(state, str(path))
            missing, unexpected = load_weights(tiny_model, path)
            assert missing == []
            assert unexpected == []

    def test_load_from_directory(self, tiny_model):
        """Load from a directory containing safetensors shards."""
        state = tiny_model.state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model-00001-of-00001.safetensors"
            save_file(state, str(path))
            missing, unexpected = load_weights(tiny_model, tmp)
            assert missing == []
            assert unexpected == []

    def test_auto_detect_our_format(self, tiny_model):
        """Auto-detection picks 'ours' when keys start with text_decoder."""
        state = tiny_model.state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            save_file(state, str(path))
            missing, unexpected = load_weights(tiny_model, path, format="auto")
            assert missing == []
