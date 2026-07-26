"""Chat-template and modality-injection tests.

The exact prompt token sequence is not something a shape assertion can catch —
a missing BOS or a stray newline silently degrades generation quality — so these
pin it structurally against a stub tokenizer.
"""

import torch

from gemma4_pt_claude.composer import (
    _AUDIO_MARKER,
    _AUDIO_SOFT_TOKEN,
    _DOUBLE_NEWLINE_TOKEN,
    _IMAGE_MARKER,
    _IMAGE_SOFT_TOKEN,
    Composer,
)
from gemma4_pt_claude.config import Gemma4Config, TextConfig, VisionConfig
from gemma4_pt_claude.tokenizer import Gemma4Tokenizer


class StubTokenizer(Gemma4Tokenizer):
    """Deterministic tokenizer: each character becomes ``1000 + ord(c)``."""

    def __init__(self):  # noqa: D107 - deliberately skips the real backend
        pass

    def encode(self, text, *, add_bos=False, add_eos=False):
        ids = [1000 + ord(c) for c in text]
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids):
        return "".join(chr(i - 1000) for i in ids if i >= 1000)


def _vision_config() -> VisionConfig:
    return VisionConfig(
        d_model=32, num_layers=1, num_heads=2, head_dim=16, ffw_hidden=64,
        patch_size=4, output_length=4, pooling_kernel_size=2,
        position_embedding_size=64, text_embed_dim=32,
    )


def _composer(**kwargs) -> Composer:
    cfg = Gemma4Config(text=TextConfig(vocab_size=64), vision=_vision_config())
    return Composer(StubTokenizer(), cfg, **kwargs)


def _image(size=(8, 8)) -> torch.Tensor:
    return torch.rand(3, *size)


class TestChatTemplate:
    def test_text_only_prompt_structure(self):
        composer = _composer()
        tok = composer.tokenizer
        ids = composer.compose_chat("hi").input_ids[0].tolist()
        expected = (
            [tok.BOS, tok.START_OF_TURN]
            + tok.encode("user\nhi")
            + [tok.END_OF_TURN]
            + tok.encode("\n")
            + [tok.START_OF_TURN]
            + tok.encode("model\n")
        )
        assert ids == expected

    def test_prompt_starts_with_bos(self):
        # Regression guard: a missing BOS measurably degrades generation.
        ids = _composer().compose_chat("hello").input_ids[0].tolist()
        assert ids[0] == Gemma4Tokenizer.BOS

    def test_prompt_ends_with_model_turn(self):
        composer = _composer()
        ids = composer.compose_chat("hello").input_ids[0].tolist()
        assert ids[-len(composer.tokenizer.encode("model\n")) - 1] == Gemma4Tokenizer.START_OF_TURN


class TestImageInjection:
    def test_no_newline_wrap_by_default(self):
        """Gemma 4's processor emits {boi}{soft tokens}{eoi} with no newlines."""
        composer = _composer()
        ids = composer.compose_chat("describe", images=[_image()]).input_ids[0].tolist()
        boi = ids.index(Gemma4Tokenizer.START_OF_IMAGE)
        eoi = ids.index(Gemma4Tokenizer.END_OF_IMAGE)
        assert ids[boi - 1] != _DOUBLE_NEWLINE_TOKEN
        assert ids[eoi + 1] != _DOUBLE_NEWLINE_TOKEN
        assert set(ids[boi + 1 : eoi]) == {_IMAGE_SOFT_TOKEN}

    def test_newline_wrap_is_opt_in(self):
        composer = _composer(image_newline_wrap=True)
        ids = composer.compose_chat("describe", images=[_image()]).input_ids[0].tolist()
        boi = ids.index(Gemma4Tokenizer.START_OF_IMAGE)
        eoi = ids.index(Gemma4Tokenizer.END_OF_IMAGE)
        assert ids[boi - 1] == _DOUBLE_NEWLINE_TOKEN
        assert ids[eoi + 1] == _DOUBLE_NEWLINE_TOKEN

    def test_soft_token_count_matches_mask(self):
        composer = _composer()
        composed = composer.compose_chat("describe", images=[_image()])
        assert composed.image_mask is not None
        assert int(composed.image_mask.sum()) == (composed.input_ids == _IMAGE_SOFT_TOKEN).sum()
        assert composed.pixel_values.shape[0] == 1

    def test_marker_position_is_respected(self):
        composer = _composer()
        ids = composer.compose(f"before{_IMAGE_MARKER}after", images=[_image()])
        flat = ids.input_ids[0].tolist()
        boi = flat.index(Gemma4Tokenizer.START_OF_IMAGE)
        assert flat[:boi] == composer.tokenizer.encode("before")
        eoi = flat.index(Gemma4Tokenizer.END_OF_IMAGE)
        assert flat[eoi + 1 :] == composer.tokenizer.encode("after")


class TestAudioInjection:
    def _audio_composer(self):
        from gemma4_pt_claude.config import AudioConfig

        cfg = Gemma4Config(
            text=TextConfig(vocab_size=64),
            vision=_vision_config(),
            audio=AudioConfig(lm_model_dims=32),
        )
        return Composer(StubTokenizer(), cfg)

    def test_audio_block_has_no_newline_wrap(self):
        composer = self._audio_composer()
        waveform = torch.zeros(16000)
        ids = composer.compose_chat("transcribe", audios=[waveform]).input_ids[0].tolist()
        boa = ids.index(Gemma4Tokenizer.START_OF_AUDIO)
        eoa = ids.index(Gemma4Tokenizer.END_OF_AUDIO)
        assert ids[boa - 1] != _DOUBLE_NEWLINE_TOKEN
        assert ids[eoa + 1] != _DOUBLE_NEWLINE_TOKEN
        assert set(ids[boa + 1 : eoa]) == {_AUDIO_SOFT_TOKEN}

    def test_audio_marker_ordering_with_image(self):
        composer = self._audio_composer()
        text = f"{_IMAGE_MARKER} and {_AUDIO_MARKER}"
        ids = composer.compose(
            text, images=[_image()], audios=[torch.zeros(16000)],
        ).input_ids[0].tolist()
        assert ids.index(Gemma4Tokenizer.START_OF_IMAGE) < ids.index(Gemma4Tokenizer.START_OF_AUDIO)
