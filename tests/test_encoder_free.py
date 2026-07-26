"""Tests for the encoder-free (12B / gemma4_unified) multimodal path.

These variants have no vision tower and no conformer: images arrive as merged
raw pixel patches and audio as raw waveform frames, both projected straight into
text space.
"""

import pytest
import torch

from gemma4_pt_claude.composer import Composer
from gemma4_pt_claude.config import (
    AttentionType,
    EncoderFreeAudioConfig,
    EncoderFreeVisionConfig,
    Gemma4Config,
    TextConfig,
    VisionConfig,
    make_attention_pattern,
)
from gemma4_pt_claude.encoder_free import EncoderFreeAudioEmbedder, EncoderFreeVisionEmbedder
from gemma4_pt_claude.audio_processing import frame_waveform
from gemma4_pt_claude.factory import gemma4_12b
from gemma4_pt_claude.image_processing import merge_patches, preprocess_images
from gemma4_pt_claude.load import _detect_variant, _hf_key_to_ours
from gemma4_pt_claude.model import Gemma4Model

from tests.test_composer import StubTokenizer


def _vision_cfg(**kw) -> EncoderFreeVisionConfig:
    defaults = dict(
        patch_size=4, pooling_kernel_size=3, output_length=4,
        mm_embed_dim=32, output_proj_dims=32, position_embedding_size=16,
        text_embed_dim=32,
    )
    defaults.update(kw)
    return EncoderFreeVisionConfig(**defaults)


class TestGemma4_12BConfig:
    """Values checked against google/gemma-4-12B-it config.json."""

    def test_text_config(self):
        with torch.device("meta"):
            m = gemma4_12b(text_only=True)
        cfg = m.cfg.text
        assert cfg.num_layers == 48
        assert cfg.embed_dim == 3840
        assert cfg.hidden_dim == 15360
        assert cfg.num_heads == 16
        assert cfg.num_kv_heads == 8
        assert cfg.num_global_kv_heads == 1
        assert cfg.head_dim == 256
        assert cfg.global_head_dim == 512
        assert cfg.sliding_window_size == 1024
        assert cfg.final_logit_softcap == 30.0
        assert cfg.k_eq_v_global is True
        assert cfg.bidirectional_vision is True
        assert cfg.per_layer_input_dim == 0   # no PLI
        assert cfg.kv_sharing is None         # no KV sharing
        assert cfg.moe is None

    def test_global_layers_match_checkpoint(self):
        """The checkpoint omits v_proj on exactly these layers."""
        with torch.device("meta"):
            m = gemma4_12b(text_only=True)
        types = make_attention_pattern(m.cfg.text.attention_pattern, m.cfg.text.num_layers)
        globals_ = [i for i, t in enumerate(types) if t == AttentionType.GLOBAL]
        assert globals_ == [5, 11, 17, 23, 29, 35, 41, 47]
        assert types[-1] == AttentionType.GLOBAL

    def test_has_no_towers(self):
        with torch.device("meta"):
            m = gemma4_12b()
        assert m.vision_encoder is None and m.audio_encoder is None
        assert m.vision_embedder is not None and m.audio_embedder is not None

    def test_text_only_skips_embedders(self):
        with torch.device("meta"):
            m = gemma4_12b(text_only=True)
        assert m.vision_embedder is None and m.audio_embedder is None

    def test_variant_detection(self):
        assert _detect_variant({"text_config": {"num_hidden_layers": 48, "hidden_size": 3840}}) == "gemma4_12b"


class TestMergePatches:
    def test_shapes_and_positions(self):
        k, p, out_len = 3, 4, 4
        n = out_len * k * k
        grid_w = 6
        patches = torch.arange(n * p * p * 3, dtype=torch.float32).reshape(n, p * p * 3)
        pos = torch.stack(torch.meshgrid(
            torch.arange(grid_w), torch.arange(n // grid_w), indexing="xy",
        ), dim=-1).reshape(n, 2)
        merged, merged_pos = merge_patches(patches, pos, out_len)
        assert merged.shape == (out_len, (k * p) ** 2 * 3)
        assert merged_pos.shape == (out_len, 2)
        # Merged positions are exactly the distinct kernel indices (order aside).
        expected = torch.div(pos, k, rounding_mode="floor").unique(dim=0)
        assert torch.equal(merged_pos.unique(dim=0), expected)
        assert merged_pos.min() >= 0

    def test_preserves_pixels(self):
        """Merging must be a pure regrouping — no pixel is lost or duplicated."""
        k, p, out_len = 3, 2, 1
        n = out_len * k * k
        patches = torch.randn(n, p * p * 3)
        pos = torch.stack(torch.meshgrid(
            torch.arange(k), torch.arange(k), indexing="xy",
        ), dim=-1).reshape(n, 2)
        merged, _ = merge_patches(patches, pos, out_len)
        assert torch.allclose(merged.sum(), patches.sum())
        assert sorted(merged.flatten().tolist()) == pytest.approx(sorted(patches.flatten().tolist()))

    def test_rejects_bad_length(self):
        patches = torch.randn(10, 48)
        pos = torch.zeros(10, 2, dtype=torch.long)
        with pytest.raises(ValueError, match="Cannot merge"):
            merge_patches(patches, pos, 3)


class TestEncoderFreeVisionEmbedder:
    def test_output_shape_and_mask(self):
        cfg = _vision_cfg()
        emb = EncoderFreeVisionEmbedder(cfg)
        emb.init_weights()
        B, L = 2, 4
        pixel_values = torch.rand(B, L, cfg.patch_dim)
        pos = torch.zeros(B, L, 2, dtype=torch.long)
        pos[:, -1] = -1  # padding
        out, mask = emb(pixel_values, pos)
        assert out.shape == (B, L, cfg.text_embed_dim)
        assert mask.shape == (B, L)
        assert not mask[:, -1].any() and mask[:, :-1].all()

    def test_patch_dim_is_merged_size(self):
        cfg = _vision_cfg(patch_size=16, pooling_kernel_size=3)
        assert cfg.model_patch_size == 48
        assert cfg.patch_dim == 48 * 48 * 3 == 6912

    def test_padding_positions_get_no_position_embedding(self):
        cfg = _vision_cfg()
        emb = EncoderFreeVisionEmbedder(cfg)
        emb.init_weights()
        with torch.no_grad():
            emb.pos_embedding.normal_(std=1.0)
        pixel_values = torch.zeros(1, 2, cfg.patch_dim)
        pad = torch.full((1, 2, 2), -1, dtype=torch.long)
        # With all-padding positions the pos embedding contributes nothing, so
        # both rows must be identical.
        out, mask = emb(pixel_values, pad)
        assert torch.allclose(out[0, 0], out[0, 1])
        assert not mask.any()


class TestEncoderFreeAudioEmbedder:
    def test_output_shape(self):
        cfg = EncoderFreeAudioConfig(text_embed_dim=32, output_proj_dims=640)
        emb = EncoderFreeAudioEmbedder(cfg)
        emb.init_weights()
        frames = torch.randn(2, 5, 640)
        out, mask = emb(frames)
        assert out.shape == (2, 5, 32)
        assert mask.shape == (2, 5) and mask.all()

    def test_respects_frame_mask(self):
        cfg = EncoderFreeAudioConfig(text_embed_dim=32, output_proj_dims=640)
        emb = EncoderFreeAudioEmbedder(cfg)
        emb.init_weights()
        frames = torch.randn(1, 4, 640)
        m = torch.tensor([[True, True, False, False]])
        _, mask = emb(frames, m)
        assert torch.equal(mask, m)


class TestFrameWaveform:
    def test_frame_count_and_padding(self):
        wav = torch.randn(16000)
        frames = frame_waveform(wav, 640)
        assert frames.shape == (25, 640)          # 16000 / 640 exactly
        assert torch.equal(frames.reshape(-1), wav)

    def test_pads_partial_frame(self):
        frames = frame_waveform(torch.ones(700), 640)
        assert frames.shape == (2, 640)
        assert frames[1, 60:].abs().sum() == 0    # tail zero-padded

    def test_rejects_multichannel(self):
        with pytest.raises(ValueError, match="1-D mono"):
            frame_waveform(torch.randn(2, 1000), 640)


class TestEncoderFreePreprocessing:
    def test_preprocess_images_merges_and_pads(self):
        cfg = EncoderFreeVisionConfig()   # real 12B geometry
        img = torch.randint(0, 255, (3, 480, 480), dtype=torch.uint8)
        out = preprocess_images([img], cfg)
        assert out["pixel_values"].shape == (1, cfg.output_length, cfg.patch_dim)
        assert out["image_position_ids"].shape == (1, cfg.output_length, 2)
        assert out["num_soft_tokens_per_image"][0] <= cfg.output_length

    def test_tower_config_still_unmerged(self):
        cfg = VisionConfig(text_embed_dim=32)
        img = torch.randint(0, 255, (3, 480, 480), dtype=torch.uint8)
        out = preprocess_images([img], cfg)
        # Tower path keeps 16px patches padded to max_patches.
        assert out["pixel_values"].shape == (1, cfg.max_patches, 3 * cfg.patch_size ** 2)


class TestEncoderFreeComposer:
    def _composer(self):
        cfg = Gemma4Config(
            text=TextConfig(vocab_size=64, embed_dim=32),
            vision=_vision_cfg(patch_size=16, output_length=280, position_embedding_size=1120),
            audio=EncoderFreeAudioConfig(text_embed_dim=32),
        )
        return Composer(StubTokenizer(), cfg)

    def test_audio_produces_frames_not_mel(self):
        comp = self._composer()
        composed = comp.compose_chat("transcribe", audios=[torch.zeros(16000)])
        assert composed.audio_frames is not None
        assert composed.audio_mel is None
        assert composed.audio_frames.shape == (1, 25, 640)
        assert int(composed.audio_num_soft_tokens[0]) == 25
        kwargs = composed.to_model_kwargs()
        assert "audio_frames" in kwargs and "audio_mel" not in kwargs

    def test_audio_soft_token_count_matches_mask(self):
        comp = self._composer()
        composed = comp.compose_chat("transcribe", audios=[torch.zeros(16000)])
        assert int(composed.audio_mask.sum()) == composed.audio_frames.shape[1]

    def test_image_soft_token_count_matches_patches(self):
        """One placeholder token per merged patch, and one merged patch per token.

        The image is resized to fill the soft-token budget, so the count is
        driven by ``output_length``, not by the input resolution.
        """
        comp = self._composer()
        img = torch.randint(0, 255, (3, 240, 240), dtype=torch.uint8)
        composed = comp.compose_chat("describe", images=[img])
        n_soft = int(composed.image_mask.sum())
        assert 0 < n_soft <= comp.config.vision.output_length
        # Each merged patch is one soft token, so the padded tensor is at least as long.
        assert composed.pixel_values.shape[1] == comp.config.vision.output_length
        assert composed.pixel_values.shape[-1] == comp.config.vision.patch_dim
        # Non-padding position ids must match the soft-token count exactly.
        valid = (composed.image_position_ids != -1).all(dim=-1).sum()
        assert int(valid) == n_soft


class TestEncoderFreeWeightMapping:
    """Keys taken from the real google/gemma-4-12B-it checkpoint."""

    @pytest.mark.parametrize("hf_key,ours", [
        ("model.vision_embedder.patch_ln1.weight", "vision_embedder.patch_ln1.weight"),
        ("model.vision_embedder.patch_ln1.bias", "vision_embedder.patch_ln1.bias"),
        ("model.vision_embedder.patch_dense.weight", "vision_embedder.patch_dense.weight"),
        ("model.vision_embedder.patch_dense.bias", "vision_embedder.patch_dense.bias"),
        ("model.vision_embedder.patch_ln2.weight", "vision_embedder.patch_ln2.weight"),
        ("model.vision_embedder.pos_embedding", "vision_embedder.pos_embedding"),
        ("model.vision_embedder.pos_norm.weight", "vision_embedder.pos_norm.weight"),
        ("model.embed_vision.embedding_projection.weight",
         "vision_embedder.multimodal_projection.proj.weight"),
        ("model.embed_audio.embedding_projection.weight",
         "audio_embedder.multimodal_projection.proj.weight"),
    ])
    def test_encoder_free_keys(self, hf_key, ours):
        assert _hf_key_to_ours(
            hf_key, 48, has_vision=True, has_audio=True, encoder_free=True,
        ) == ours

    def test_tower_mapping_unaffected(self):
        assert _hf_key_to_ours(
            "model.embed_vision.embedding_projection.weight", 35, has_vision=True,
        ) == "embed_vision.proj.weight"

    def test_text_keys_shared_with_tower_variants(self):
        """12B text keys use the same naming, so the existing mapping applies."""
        for hf_key, ours in [
            ("model.language_model.layers.5.self_attn.q_proj.weight",
             "text_decoder.blocks.5.attn.q_proj.weight"),
            ("model.language_model.layers.5.layer_scalar",
             "text_decoder.blocks.5.skip_scale"),
            ("model.language_model.norm.weight", "text_decoder.final_norm.weight"),
        ]:
            assert _hf_key_to_ours(hf_key, 48, encoder_free=True) == ours


class TestEncoderFreeModelForward:
    def _model(self):
        text = TextConfig(
            vocab_size=64, embed_dim=32, hidden_dim=64, num_heads=4, head_dim=16,
            num_kv_heads=2, num_layers=2, sliding_window_size=32,
            attention_pattern=(AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL),
            use_value_norm=False, bidirectional_vision=True,
        )
        cfg = Gemma4Config(
            text=text,
            vision=_vision_cfg(),
            audio=EncoderFreeAudioConfig(text_embed_dim=32, output_proj_dims=640),
        )
        torch.manual_seed(0)
        return Gemma4Model(cfg).eval()

    def test_image_tokens_are_injected(self):
        m = self._model()
        tokens = torch.randint(4, 64, (1, 10))
        image_mask = torch.zeros(1, 10, dtype=torch.bool)
        image_mask[0, 2:6] = True
        pixel_values = torch.rand(1, 4, m.cfg.vision.patch_dim)
        pos = torch.zeros(1, 4, 2, dtype=torch.long)
        with torch.no_grad():
            with_img, _ = m(tokens, pixel_values=pixel_values,
                            image_position_ids=pos, image_mask=image_mask)
            without, _ = m(tokens)
        assert with_img.shape == (1, 10, 64)
        assert not torch.allclose(with_img, without)

    def test_audio_frames_are_injected(self):
        m = self._model()
        tokens = torch.randint(4, 64, (1, 8))
        audio_mask = torch.zeros(1, 8, dtype=torch.bool)
        audio_mask[0, 1:4] = True
        frames = torch.randn(1, 3, 640)
        with torch.no_grad():
            with_audio, _ = m(tokens, audio_frames=frames, audio_mask=audio_mask)
            without, _ = m(tokens)
        assert not torch.allclose(with_audio, without)

    def test_cached_decode_matches_full_forward(self):
        from gemma4_pt_claude.generate import init_cache

        m = self._model()
        tokens = torch.randint(4, 64, (1, 10))
        image_mask = torch.zeros(1, 10, dtype=torch.bool)
        image_mask[0, 2:6] = True
        pixel_values = torch.rand(1, 4, m.cfg.vision.patch_dim)
        pos = torch.zeros(1, 4, 2, dtype=torch.long)
        mm = dict(pixel_values=pixel_values, image_position_ids=pos, image_mask=image_mask)
        with torch.no_grad():
            full, _ = m(tokens, **mm)
            cache = init_cache(m.cfg, 1, 32, dtype=torch.float32)
            cached, _ = m(tokens, cache=cache, **mm)
        assert torch.allclose(full, cached, atol=1e-4)
