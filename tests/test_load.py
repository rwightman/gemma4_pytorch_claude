"""Unit tests for weight loading."""

import tempfile
from pathlib import Path

import torch
import pytest
from safetensors.torch import save_file

from gemma4_pt_claude.config import AttentionType, AudioConfig, Gemma4Config, MoEConfig, TextConfig
from gemma4_pt_claude.model import Gemma4Model
from gemma4_pt_claude.load import (
    load_weights,
    load_weights_streaming,
    _hf_key_to_ours,
    _hf_convert_weights,
    _hf_vision_key_to_ours,
    _hf_audio_key_to_ours,
)


class TestHFKeyMapping:
    def test_embed_tokens(self):
        assert _hf_key_to_ours("model.language_model.embed_tokens.weight", 4) == (
            "text_decoder.embedder.token_embedding.weight"
        )

    def test_final_norm(self):
        assert _hf_key_to_ours("model.language_model.norm.weight", 4) == (
            "text_decoder.final_norm.weight"
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
        assert result == "text_decoder.blocks.1.pre_attn_norm.weight"

    def test_qk_norm(self):
        result = _hf_key_to_ours(
            "model.language_model.layers.0.self_attn.q_norm.weight", 4
        )
        assert result == "text_decoder.blocks.0.attn.q_norm.weight"

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
        ) == "text_decoder.embedder.pli_proj_norm.weight"

    def test_pli_layer_keys(self):
        assert _hf_key_to_ours(
            "model.language_model.layers.0.per_layer_input_gate.weight", 4
        ) == "text_decoder.blocks.0.pli_mapping.gate.weight"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.per_layer_projection.weight", 4
        ) == "text_decoder.blocks.0.pli_mapping.proj.weight"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.post_per_layer_input_norm.weight", 4
        ) == "text_decoder.blocks.0.pli_mapping.norm.weight"

    def test_moe_dense_branch_mapping(self):
        assert _hf_key_to_ours(
            "model.language_model.layers.0.mlp.down_proj.weight", 4, has_moe=True
        ) == "text_decoder.blocks.0.mlp2.down_proj.weight"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.mlp.gate_up_proj.weight", 4, has_moe=True
        ) == "text_decoder.blocks.0.mlp2.gate_up_proj.weight"

    def test_moe_router_and_expert_mapping(self):
        assert _hf_key_to_ours(
            "model.language_model.layers.0.router.proj.weight", 4, has_moe=True
        ) == "text_decoder.blocks.0.moe.router.gate.weight"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.router.scale", 4, has_moe=True
        ) == "text_decoder.blocks.0.moe.router.router_scale"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.router.per_expert_scale", 4, has_moe=True
        ) == "text_decoder.blocks.0.moe.experts.per_expert_scale"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.experts.gate_up_proj", 4, has_moe=True
        ) == "text_decoder.blocks.0.moe.experts.gate_up"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.experts.down_proj", 4, has_moe=True
        ) == "text_decoder.blocks.0.moe.experts.down"

    def test_moe_norm_mapping(self):
        assert _hf_key_to_ours(
            "model.language_model.layers.0.pre_feedforward_layernorm.weight", 4, has_moe=True
        ) == "text_decoder.blocks.0.pre_ffw2_norm.weight"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.pre_feedforward_layernorm_2.weight", 4, has_moe=True
        ) == "text_decoder.blocks.0.pre_ffw_norm.weight"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.post_feedforward_layernorm_1.weight", 4, has_moe=True
        ) == "text_decoder.blocks.0.post_ffw2_norm.weight"
        assert _hf_key_to_ours(
            "model.language_model.layers.0.post_feedforward_layernorm_2.weight", 4, has_moe=True
        ) == "text_decoder.blocks.0.post_ffw1_norm.weight"


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

    def test_moe_paths_and_transposes(self):
        raw = {
            "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(4, 2),
            "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(4, 2),
            "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(2, 4),
            "model.language_model.layers.0.experts.gate_up_proj": torch.randn(3, 8, 2),
            "model.language_model.layers.0.experts.down_proj": torch.randn(3, 2, 8),
            "model.language_model.layers.0.router.proj.weight": torch.randn(3, 2),
            "model.language_model.layers.0.router.scale": torch.randn(2),
            "model.language_model.layers.0.router.per_expert_scale": torch.randn(3),
            "model.language_model.layers.0.pre_feedforward_layernorm.weight": torch.randn(2),
            "model.language_model.layers.0.pre_feedforward_layernorm_2.weight": torch.randn(2),
            "model.language_model.layers.0.post_feedforward_layernorm_1.weight": torch.randn(2),
            "model.language_model.layers.0.post_feedforward_layernorm_2.weight": torch.randn(2),
        }
        mapped = _hf_convert_weights(raw, num_layers=1, has_moe=True)

        assert "text_decoder.blocks.0.mlp2.gate_up_proj.weight" in mapped
        assert "text_decoder.blocks.0.mlp2.down_proj.weight" in mapped
        assert "text_decoder.blocks.0.moe.router.gate.weight" in mapped
        assert "text_decoder.blocks.0.moe.router.router_scale" in mapped
        assert "text_decoder.blocks.0.moe.experts.per_expert_scale" in mapped
        assert "text_decoder.blocks.0.pre_ffw_norm.weight" in mapped
        assert "text_decoder.blocks.0.pre_ffw2_norm.weight" in mapped
        assert "text_decoder.blocks.0.post_ffw1_norm.weight" in mapped
        assert "text_decoder.blocks.0.post_ffw2_norm.weight" in mapped

        fused = mapped["text_decoder.blocks.0.mlp2.gate_up_proj.weight"]
        assert fused.shape == (8, 2)
        assert torch.equal(fused[:4], raw["model.language_model.layers.0.mlp.gate_proj.weight"])
        assert torch.equal(fused[4:], raw["model.language_model.layers.0.mlp.up_proj.weight"])

        expected_gate_up = raw["model.language_model.layers.0.experts.gate_up_proj"].transpose(1, 2).contiguous()
        expected_down = raw["model.language_model.layers.0.experts.down_proj"].transpose(1, 2).contiguous()
        assert torch.equal(mapped["text_decoder.blocks.0.moe.experts.gate_up"], expected_gate_up)
        assert torch.equal(mapped["text_decoder.blocks.0.moe.experts.down"], expected_down)


class TestLoadWeights:
    @staticmethod
    def _runtime_buffer_config(with_audio: bool = False) -> Gemma4Config:
        text = TextConfig(
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
            per_layer_input_dim=4,
        )
        audio = None
        if with_audio:
            audio = AudioConfig(
                hidden_size=16,
                num_layers=1,
                num_heads=4,
                chunk_size=4,
                context_left=5,
                context_right=0,
                conv_kernel_size=3,
                input_feat_size=8,
                sscp_channels=(8, 4),
                lm_model_dims=16,
            )
        return Gemma4Config(text=text, audio=audio)

    @pytest.fixture
    def tiny_model(self):
        cfg = self._runtime_buffer_config()
        return Gemma4Model(cfg)

    @staticmethod
    def _runtime_moe_config() -> Gemma4Config:
        text = TextConfig(
            vocab_size=32,
            embed_dim=16,
            hidden_dim=12,
            num_heads=2,
            head_dim=8,
            num_kv_heads=2,
            num_layers=1,
            sliding_window_size=8,
            attention_pattern=(AttentionType.GLOBAL,),
            use_qk_norm=False,
            use_value_norm=False,
            use_post_attn_norm=False,
            use_post_ffw_norm=True,
            per_layer_input_dim=0,
            moe=MoEConfig(
                num_experts=4,
                top_k=2,
                expert_dim=6,
                dense_hidden_dim=12,
            ),
        )
        return Gemma4Config(text=text)

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

    def test_init_non_persistent_buffers_recurses(self):
        model = Gemma4Model(self._runtime_buffer_config(with_audio=True))
        embedder = model.text_decoder.embedder
        audio_attn = model.audio_encoder.conformer[0].attn.attn

        embedder.pli_proj_scale = 0.0
        audio_attn.q_scale_base = 0.0
        audio_attn.key_scale = 0.0
        audio_attn.local_causal_mask = torch.zeros_like(audio_attn.local_causal_mask)

        model.init_non_persistent_buffers()

        assert embedder.pli_proj_scale == 0.0
        assert audio_attn.q_scale_base == 0.0
        assert audio_attn.key_scale == 0.0
        assert audio_attn.local_causal_mask.any()

    def test_load_meta_model_rebuilds_runtime_buffers(self):
        cfg = self._runtime_buffer_config(with_audio=True)
        reference = Gemma4Model(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            save_file(reference.state_dict(), str(path))
            with torch.device("meta"):
                meta_model = Gemma4Model(cfg)
            missing, unexpected = load_weights(meta_model, path, device="cpu")

        assert missing == []
        assert unexpected == []
        assert not meta_model.text_decoder.embedder.token_embedding.weight.is_meta
        assert isinstance(meta_model.text_decoder.embedder.pli_proj_scale, float)
        assert meta_model.text_decoder.embedder.pli_proj_scale > 0.0
        assert not meta_model.audio_encoder.conformer[0].attn.attn.local_causal_mask.is_meta

    def test_load_streaming_meta_model_rebuilds_runtime_buffers(self):
        cfg = self._runtime_buffer_config(with_audio=True)
        reference = Gemma4Model(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            save_file(reference.state_dict(), str(path))
            with torch.device("meta"):
                meta_model = Gemma4Model(cfg)
            missing, unexpected = load_weights_streaming(meta_model, path, device="cpu")

        assert missing == []
        assert unexpected == []
        assert not meta_model.text_decoder.embedder.token_embedding.weight.is_meta
        assert isinstance(meta_model.text_decoder.embedder.pli_proj_scale, float)
        assert meta_model.text_decoder.embedder.pli_proj_scale > 0.0
        assert not meta_model.audio_encoder.conformer[0].attn.attn.local_causal_mask.is_meta

    def test_streaming_hf_dense_merge(self):
        cfg = self._runtime_buffer_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            gate = torch.randn(32, 16)
            up = torch.randn(32, 16)
            down = torch.randn(16, 32)
            save_file({
                "model.language_model.layers.0.mlp.gate_proj.weight": gate,
                "model.language_model.layers.0.mlp.up_proj.weight": up,
                "model.language_model.layers.0.mlp.down_proj.weight": down,
            }, str(path))

            with torch.device("meta"):
                meta_model = Gemma4Model(cfg)
            missing, unexpected = load_weights_streaming(meta_model, path, format="hf", device="cpu")

        assert unexpected == []
        fused = meta_model.state_dict()["text_decoder.blocks.0.ffw.gate_up_proj.weight"]
        assert torch.equal(fused[:32], gate)
        assert torch.equal(fused[32:], up)
        assert torch.equal(meta_model.state_dict()["text_decoder.blocks.0.ffw.down_proj.weight"], down)
        assert "text_decoder.blocks.0.ffw.gate_up_proj.weight" not in missing
        assert "text_decoder.blocks.0.ffw.down_proj.weight" not in missing

    def test_streaming_hf_moe_mapping(self):
        cfg = self._runtime_moe_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            gate = torch.randn(12, 16)
            up = torch.randn(12, 16)
            down = torch.randn(16, 12)
            moe_gate_up = torch.randn(4, 12, 16)
            moe_down = torch.randn(4, 16, 6)
            router = torch.randn(4, 16)
            router_scale = torch.randn(16)
            per_expert_scale = torch.randn(4)
            pre_ffw = torch.randn(16)
            pre_ffw2 = torch.randn(16)
            post_ffw1 = torch.randn(16)
            post_ffw2 = torch.randn(16)
            post_ffw = torch.randn(16)
            save_file({
                "model.language_model.layers.0.mlp.gate_proj.weight": gate,
                "model.language_model.layers.0.mlp.up_proj.weight": up,
                "model.language_model.layers.0.mlp.down_proj.weight": down,
                "model.language_model.layers.0.experts.gate_up_proj": moe_gate_up,
                "model.language_model.layers.0.experts.down_proj": moe_down,
                "model.language_model.layers.0.router.proj.weight": router,
                "model.language_model.layers.0.router.scale": router_scale,
                "model.language_model.layers.0.router.per_expert_scale": per_expert_scale,
                "model.language_model.layers.0.pre_feedforward_layernorm.weight": pre_ffw2,
                "model.language_model.layers.0.pre_feedforward_layernorm_2.weight": pre_ffw,
                "model.language_model.layers.0.post_feedforward_layernorm_1.weight": post_ffw2,
                "model.language_model.layers.0.post_feedforward_layernorm_2.weight": post_ffw1,
                "model.language_model.layers.0.post_feedforward_layernorm.weight": post_ffw,
            }, str(path))

            with torch.device("meta"):
                meta_model = Gemma4Model(cfg)
            missing, unexpected = load_weights_streaming(meta_model, path, format="hf", device="cpu")

        assert unexpected == []
        sd = meta_model.state_dict()
        assert torch.equal(sd["text_decoder.blocks.0.mlp2.gate_up_proj.weight"][:12], gate)
        assert torch.equal(sd["text_decoder.blocks.0.mlp2.gate_up_proj.weight"][12:], up)
        assert torch.equal(sd["text_decoder.blocks.0.mlp2.down_proj.weight"], down)
        assert torch.equal(sd["text_decoder.blocks.0.moe.router.gate.weight"], router)
        assert torch.equal(sd["text_decoder.blocks.0.moe.router.router_scale"], router_scale)
        assert torch.equal(sd["text_decoder.blocks.0.moe.experts.per_expert_scale"], per_expert_scale)
        assert torch.equal(sd["text_decoder.blocks.0.moe.experts.gate_up"], moe_gate_up.transpose(1, 2))
        assert torch.equal(sd["text_decoder.blocks.0.moe.experts.down"], moe_down.transpose(1, 2))
        assert torch.equal(sd["text_decoder.blocks.0.pre_ffw_norm.weight"], pre_ffw)
        assert torch.equal(sd["text_decoder.blocks.0.pre_ffw2_norm.weight"], pre_ffw2)
        assert torch.equal(sd["text_decoder.blocks.0.post_ffw1_norm.weight"], post_ffw1)
        assert torch.equal(sd["text_decoder.blocks.0.post_ffw2_norm.weight"], post_ffw2)
        assert torch.equal(sd["text_decoder.blocks.0.post_ffw_norm.weight"], post_ffw)
        assert "text_decoder.blocks.0.moe.router.gate.weight" not in missing
        assert "text_decoder.blocks.0.moe.experts.gate_up" not in missing

    def test_materialize_initializes_meta_model_and_runtime_buffers(self):
        cfg = self._runtime_buffer_config(with_audio=True)
        with torch.device("meta"):
            model = Gemma4Model(cfg)

        model.materialize(device="cpu", dtype=torch.float32, init_weights=True)

        assert not model.text_decoder.embedder.token_embedding.weight.is_meta
        assert model.text_decoder.embedder.token_embedding.weight.dtype == torch.float32
        assert not model.audio_encoder.conformer[0].attn.attn.local_causal_mask.is_meta


class TestVisionKeyMapping:
    def test_patch_embedder(self):
        assert _hf_vision_key_to_ours(
            "vision_tower.patch_embedder.input_proj.weight"
        ) == "vision_encoder.patch_embedder.input_proj.weight"

    def test_position_embedding_table(self):
        assert _hf_vision_key_to_ours(
            "vision_tower.patch_embedder.position_embedding_table"
        ) == "vision_encoder.patch_embedder.position_embedding_table"

    def test_encoder_layer_attn(self):
        assert _hf_vision_key_to_ours(
            "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"
        ) == "vision_encoder.layers.0.attn.q_proj.linear.weight"

    def test_encoder_layer_attn_without_clipped_linear(self):
        assert _hf_vision_key_to_ours(
            "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
            use_clipped_linear=False,
        ) == "vision_encoder.layers.0.attn.q_proj.weight"

    def test_encoder_layer_norm(self):
        assert _hf_vision_key_to_ours(
            "vision_tower.encoder.layers.5.input_layernorm.weight"
        ) == "vision_encoder.layers.5.pre_attn_norm.weight"

    def test_encoder_layer_mlp(self):
        assert _hf_vision_key_to_ours(
            "vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight"
        ) == "vision_encoder.layers.0.mlp.gate_proj.linear.weight"

    def test_encoder_layer_mlp_without_clipped_linear(self):
        assert _hf_vision_key_to_ours(
            "vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight",
            use_clipped_linear=False,
        ) == "vision_encoder.layers.0.mlp.gate_proj.weight"

    def test_rotary_emb_skipped(self):
        assert _hf_vision_key_to_ours(
            "vision_tower.encoder.rotary_emb.inv_freq"
        ) is None

    def test_qk_norm(self):
        assert _hf_vision_key_to_ours(
            "vision_tower.encoder.layers.0.self_attn.q_norm.weight"
        ) == "vision_encoder.layers.0.attn.q_norm.weight"

    def test_standardize_buffers(self):
        assert _hf_vision_key_to_ours("vision_tower.std_bias") == "vision_encoder.std_bias"
        assert _hf_vision_key_to_ours("vision_tower.std_scale") == "vision_encoder.std_scale"


class TestAudioKeyMapping:
    def test_subsample(self):
        assert _hf_audio_key_to_ours(
            "audio_tower.subsampling.conv1.weight"
        ) == "audio_encoder.subsample.conv1.weight"

    def test_current_hf_subsample(self):
        assert _hf_audio_key_to_ours(
            "audio_tower.subsample_conv_projection.layer0.conv.weight"
        ) == "audio_encoder.subsample.conv1.weight"

    def test_subsample_norm(self):
        # LayerNorm: weight only, no bias
        assert _hf_audio_key_to_ours(
            "audio_tower.subsampling.norm1.weight"
        ) == "audio_encoder.subsample.norm1.weight"

    def test_output_proj_weight(self):
        assert _hf_audio_key_to_ours(
            "audio_tower.output_proj.weight"
        ) == "audio_encoder.output_proj.weight"

    def test_output_proj_bias(self):
        assert _hf_audio_key_to_ours(
            "audio_tower.output_proj.bias"
        ) == "audio_encoder.output_proj.bias"

    def test_non_audio_returns_none(self):
        assert _hf_audio_key_to_ours("vision_tower.something") is None

    def test_conformer_layer(self):
        result = _hf_audio_key_to_ours(
            "audio_tower.conformer.layers.0.ffw_start.up.linear.weight"
        )
        assert result == "audio_encoder.conformer.0.ffw_start.up.linear.weight"

    def test_current_hf_conformer_layer(self):
        result = _hf_audio_key_to_ours(
            "audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight"
        )
        assert result == "audio_encoder.conformer.0.ffw_start.up.linear.weight"

    def test_current_hf_relative_position(self):
        result = _hf_audio_key_to_ours(
            "audio_tower.layers.0.self_attn.relative_k_proj.weight"
        )
        assert result == "audio_encoder.conformer.0.attn.attn.rel_pos_emb.pos_proj.weight"

    def test_audio_embedder_key(self):
        result = _hf_key_to_ours(
            "model.embed_audio.embedding_projection.weight", 4,
            has_audio=True,
        )
        assert result == "embed_audio.proj.weight"
