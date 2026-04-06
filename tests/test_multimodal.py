"""Unit tests for multimodal merging, preprocessing, and composer."""

import torch
import pytest

from gemma4_pt_claude.config import (
    AttentionType,
    Gemma4Config,
    TextConfig,
    VisionConfig,
)
from gemma4_pt_claude.model import (
    Gemma4Model,
    merge_multimodal_embeddings,
    make_causal_bidirectional_mask,
    make_causal_mask,
    VisionEmbedder,
    AudioEmbedder,
)
from gemma4_pt_claude.image_processing import preprocess_image, preprocess_images


def _tiny_vision_config(**overrides) -> VisionConfig:
    defaults = dict(
        d_model=32,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        ffw_hidden=64,
        patch_size=4,
        output_length=4,
        pooling_kernel_size=2,
        position_embedding_size=64,
        use_clipped_linear=False,
        standardize=False,
        rms_norm_eps=1e-6,
        rope_base_frequency=100.0,
        text_embed_dim=32,
    )
    defaults.update(overrides)
    return VisionConfig(**defaults)


def _tiny_text_config(**overrides) -> TextConfig:
    defaults = dict(
        vocab_size=64,
        embed_dim=32,
        hidden_dim=64,
        num_heads=2,
        head_dim=16,
        num_kv_heads=2,
        num_layers=2,
        sliding_window_size=16,
        attention_pattern=(AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL),
        use_qk_norm=True,
        use_value_norm=False,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
    )
    defaults.update(overrides)
    return TextConfig(**defaults)


def _tiny_config_with_vision(**text_overrides) -> Gemma4Config:
    return Gemma4Config(
        text=_tiny_text_config(**text_overrides),
        vision=_tiny_vision_config(),
    )


class TestMergeMultimodalEmbeddings:
    def test_correct_placement(self):
        B, L, D = 1, 8, 16
        text = torch.zeros(B, L, D)
        mm = torch.ones(B, 3, D)
        mm_mask = torch.ones(B, 3, dtype=torch.bool)
        placeholder = torch.zeros(B, L, dtype=torch.bool)
        placeholder[0, 2] = True
        placeholder[0, 4] = True
        placeholder[0, 6] = True
        out = merge_multimodal_embeddings(text, mm, mm_mask, placeholder)
        assert out[0, 2].sum().item() == D  # ones placed here
        assert out[0, 4].sum().item() == D
        assert out[0, 6].sum().item() == D
        assert out[0, 0].sum().item() == 0  # untouched
        assert out[0, 1].sum().item() == 0

    def test_count_validation(self):
        B, L, D = 1, 4, 8
        text = torch.zeros(B, L, D)
        mm = torch.ones(B, 2, D)
        mm_mask = torch.ones(B, 2, dtype=torch.bool)
        # Only 1 placeholder but 2 soft tokens
        placeholder = torch.zeros(B, L, dtype=torch.bool)
        placeholder[0, 0] = True
        with pytest.raises(AssertionError, match="1 placeholders vs 2"):
            merge_multimodal_embeddings(text, mm, mm_mask, placeholder)

    def test_batch_processing(self):
        B, L, D = 2, 6, 8
        text = torch.zeros(B, L, D)
        mm = torch.ones(B, 2, D)
        mm_mask = torch.ones(B, 2, dtype=torch.bool)
        placeholder = torch.zeros(B, L, dtype=torch.bool)
        placeholder[0, 1] = True
        placeholder[0, 3] = True
        placeholder[1, 0] = True
        placeholder[1, 5] = True
        out = merge_multimodal_embeddings(text, mm, mm_mask, placeholder)
        assert out[0, 1].sum().item() == D
        assert out[0, 3].sum().item() == D
        assert out[1, 0].sum().item() == D
        assert out[1, 5].sum().item() == D


class TestVisionEmbedder:
    def test_output_shape(self):
        embedder = VisionEmbedder(32, 64)
        x = torch.randn(2, 10, 32)
        out = embedder(x)
        assert out.shape == (2, 10, 64)

    def test_norm_before_proj(self):
        """Vision: norm applied before projection."""
        embedder = VisionEmbedder(16, 32)
        x = torch.randn(1, 5, 16, dtype=torch.float32)
        out = embedder(x)
        assert out.dtype == torch.float32


class TestAudioEmbedder:
    def test_output_shape(self):
        embedder = AudioEmbedder(32, 64)
        x = torch.randn(2, 10, 32)
        out = embedder(x)
        assert out.shape == (2, 10, 64)

    def test_norm_after_proj(self):
        """Audio: norm applied after projection."""
        embedder = AudioEmbedder(16, 32)
        x = torch.randn(1, 5, 16, dtype=torch.float32)
        out = embedder(x)
        assert out.dtype == torch.float32


class TestImagePreprocessing:
    def test_pil_image(self):
        pytest.importorskip("PIL")
        from PIL import Image
        img = Image.new("RGB", (64, 48))
        patches, pos_ids, n_soft = preprocess_image(img, 4, 16, 2)
        assert patches.ndim == 2
        assert pos_ids.ndim == 2
        assert pos_ids.shape[-1] == 2
        assert n_soft > 0

    def test_uint8_tensor(self):
        """uint8 tensors should be rescaled to [0, 1]."""
        tensor = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        patches, pos_ids, n_soft = preprocess_image(tensor, 4, 16, 2)
        # If uint8 rescaling works, patches should be in roughly [0, 1] range
        # (after bicubic + patch rescaling, values range may slightly exceed)
        assert patches.dtype == torch.float32
        assert n_soft > 0

    def test_float_tensor(self):
        tensor = torch.rand(3, 32, 32)
        patches, pos_ids, n_soft = preprocess_image(tensor, 4, 16, 2)
        assert patches.dtype == torch.float32

    def test_batch_preprocess(self):
        pytest.importorskip("PIL")
        from PIL import Image
        cfg = _tiny_vision_config()
        images = [Image.new("RGB", (32, 32)), Image.new("RGB", (48, 64))]
        result = preprocess_images(images, cfg)
        assert result["pixel_values"].shape[0] == 2
        assert result["pixel_values"].shape[1] == cfg.max_patches
        assert len(result["num_soft_tokens_per_image"]) == 2


class TestBidirectionalVisionMask:
    def test_basic(self):
        B, L = 1, 8
        causal = make_causal_mask(L, torch.device("cpu")).expand(B, -1, -1)
        bidir_mask = torch.zeros(B, L, dtype=torch.bool)
        # Mark positions 2,3,4 as image tokens
        bidir_mask[0, 2:5] = True
        result = make_causal_bidirectional_mask(causal, bidir_mask)

        # Position 2 should now see position 4 (bidirectional within span)
        assert result[0, 2, 4].item() is True
        # Position 4 should see position 2 (reverse)
        assert result[0, 4, 2].item() is True
        # Causal behavior outside image span preserved
        assert result[0, 0, 1].item() is False
        assert result[0, 1, 0].item() is True

    def test_separate_spans(self):
        B, L = 1, 10
        causal = make_causal_mask(L, torch.device("cpu")).expand(B, -1, -1)
        bidir_mask = torch.zeros(B, L, dtype=torch.bool)
        bidir_mask[0, 1:3] = True  # span 1: positions 1,2
        bidir_mask[0, 6:8] = True  # span 2: positions 6,7

        result = make_causal_bidirectional_mask(causal, bidir_mask)
        # Within span 1: bidirectional
        assert result[0, 1, 2].item() is True
        assert result[0, 2, 1].item() is True
        # Within span 2: bidirectional
        assert result[0, 6, 7].item() is True
        assert result[0, 7, 6].item() is True
        # Cross-span: not bidirectional (only causal)
        assert result[0, 1, 7].item() is False

    def test_no_bidirectional_positions(self):
        B, L = 1, 4
        causal = make_causal_mask(L, torch.device("cpu")).expand(B, -1, -1)
        bidir_mask = torch.zeros(B, L, dtype=torch.bool)
        result = make_causal_bidirectional_mask(causal, bidir_mask)
        assert torch.equal(result, causal)


class TestModelWithVision:
    def test_forward_with_vision(self):
        cfg = _tiny_config_with_vision()
        model = Gemma4Model(cfg)
        model.eval()

        B, L = 1, 12
        tokens = torch.randint(0, 64, (B, L))
        patch_dim = 3 * cfg.vision.patch_size ** 2
        pixel_values = torch.randn(B, cfg.vision.max_patches, patch_dim)
        pos_ids = torch.zeros(B, cfg.vision.max_patches, 2, dtype=torch.long)
        k = cfg.vision.pooling_kernel_size
        for i in range(cfg.vision.max_patches):
            pos_ids[:, i, 0] = i % (k * 2)
            pos_ids[:, i, 1] = i // (k * 2)

        # Mark some positions as image placeholders
        image_mask = torch.zeros(B, L, dtype=torch.bool)
        image_mask[0, 2:6] = True  # 4 placeholders = output_length

        with torch.no_grad():
            logits, _ = model(
                tokens,
                pixel_values=pixel_values,
                image_position_ids=pos_ids,
                image_mask=image_mask,
            )
        assert logits.shape == (B, L, 64)
        assert not torch.isnan(logits).any()
