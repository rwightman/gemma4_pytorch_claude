"""Tests for audio_processing.py — mel spectrogram extraction and preprocessing."""

import math
import sys

import numpy as np
import pytest
import torch

from gemma4_pt_claude.audio_processing import (
    _HAS_TORCHAUDIO,
    _hann_window_nonzero,
    _mel_filterbank,
    to_float32,
)
from gemma4_pt_claude.composer import (
    AudioTransform,
    ComposedInput,
    PreparedAudio,
    _compute_audio_soft_tokens,
)

needs_torchaudio = pytest.mark.skipif(not _HAS_TORCHAUDIO, reason="torchaudio not installed")


# ---------------------------------------------------------------------------
# Hann window (no torchaudio needed)
# ---------------------------------------------------------------------------

class TestHannWindowNonzero:
    def test_shape(self):
        w = _hann_window_nonzero(320)
        assert w.shape == (320,)

    def test_nonzero_endpoints(self):
        w = _hann_window_nonzero(320)
        assert w[0].item() > 0, "Window should not touch zero at start"
        assert w[-1].item() > 0, "Window should not touch zero at end"

    def test_matches_jax_formula(self):
        """Verify: 0.5 - 0.5*cos(2*pi*(n+0.5)/N)."""
        N = 320
        n = torch.arange(N, dtype=torch.float64)
        expected = 0.5 - 0.5 * torch.cos(2.0 * math.pi * (n + 0.5) / N)
        actual = _hann_window_nonzero(N, dtype=torch.float64)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Mel filterbank (no torchaudio needed)
# ---------------------------------------------------------------------------

class TestMelFilterbank:
    def test_shape(self):
        fb = _mel_filterbank(128, 512, 16000, 0.0, 8000.0)
        assert fb.shape == (128, 257)

    def test_nonnegative(self):
        fb = _mel_filterbank(128, 512, 16000, 0.0, 8000.0)
        assert (fb >= 0).all()


# ---------------------------------------------------------------------------
# dtype conversion (no torchaudio needed)
# ---------------------------------------------------------------------------

class TestToFloat32:
    def test_int16(self):
        x = torch.tensor([0, 16384, -32768], dtype=torch.int16)
        y = to_float32(x)
        assert y.dtype == torch.float32
        assert y[0].item() == 0.0
        assert abs(y[1].item() - 0.5) < 1e-6
        assert y[2].item() == -1.0

    def test_uint8(self):
        x = torch.tensor([128, 0, 255], dtype=torch.uint8)
        y = to_float32(x)
        assert y.dtype == torch.float32
        assert abs(y[0].item()) < 1e-6  # 128 → center → 0

    def test_float_passthrough(self):
        x = torch.randn(10)
        y = to_float32(x)
        torch.testing.assert_close(y, x)

    def test_float64_to_float32(self):
        x = torch.randn(10, dtype=torch.float64)
        y = to_float32(x)
        assert y.dtype == torch.float32


# ---------------------------------------------------------------------------
# Import guard (no torchaudio needed)
# ---------------------------------------------------------------------------

class TestImportGuard:
    def test_error_without_torchaudio(self):
        """Calling extract_mel_spectrogram without torchaudio raises ImportError."""
        import gemma4_pt_claude.audio_processing as ap
        original = ap._HAS_TORCHAUDIO
        try:
            ap._HAS_TORCHAUDIO = False
            with pytest.raises(ImportError, match="torchaudio"):
                ap.extract_mel_spectrogram(torch.randn(16000))
        finally:
            ap._HAS_TORCHAUDIO = original

    def test_preprocess_error_without_torchaudio(self):
        """Calling preprocess_audio without torchaudio raises ImportError."""
        import gemma4_pt_claude.audio_processing as ap
        original = ap._HAS_TORCHAUDIO
        try:
            ap._HAS_TORCHAUDIO = False
            with pytest.raises(ImportError, match="torchaudio"):
                ap.preprocess_audio(torch.randn(16000))
        finally:
            ap._HAS_TORCHAUDIO = original


# ---------------------------------------------------------------------------
# Soft token count (no torchaudio needed)
# ---------------------------------------------------------------------------

class TestComputeAudioSoftTokens:
    def test_1s(self):
        """1s of 16 kHz → 25 soft tokens (matching JAX formula)."""
        assert _compute_audio_soft_tokens(16000) == 25

    def test_short(self):
        # 321 samples → 1 mel frame → subsampled twice → 1
        assert _compute_audio_soft_tokens(321) == 1

    def test_10s(self):
        """10s → reasonable count."""
        count = _compute_audio_soft_tokens(160000)
        assert 100 < count < 300

    def test_too_short(self):
        """Fewer than 321 samples → 0 tokens."""
        assert _compute_audio_soft_tokens(100) == 0


# ---------------------------------------------------------------------------
# ComposedInput audio fields (no torchaudio needed)
# ---------------------------------------------------------------------------

class TestComposedInputAudio:
    def test_audio_fields_in_kwargs(self):
        """ComposedInput has audio fields and to_model_kwargs includes them."""
        mel = torch.randn(1, 50, 128)
        mel_mask = torch.ones(1, 50, dtype=torch.bool)
        audio_mask = torch.zeros(1, 10, dtype=torch.bool)
        audio_mask[0, 3:6] = True

        ci = ComposedInput(
            input_ids=torch.zeros(1, 10, dtype=torch.long),
            pixel_values=None,
            image_position_ids=None,
            image_mask=None,
            audio_mel=mel,
            audio_mel_mask=mel_mask,
            audio_mask=audio_mask,
            audio_num_soft_tokens=torch.tensor([3], dtype=torch.long),
        )

        kwargs = ci.to_model_kwargs()
        assert "audio_mel" in kwargs
        assert "audio_mel_mask" in kwargs
        assert "audio_mask" in kwargs
        assert "audio_num_soft_tokens" in kwargs
        assert kwargs["audio_mel"].shape == (1, 50, 128)

    def test_no_audio_omits_fields(self):
        """When no audio, kwargs should not have audio keys."""
        ci = ComposedInput(
            input_ids=torch.zeros(1, 10, dtype=torch.long),
            pixel_values=None,
            image_position_ids=None,
            image_mask=None,
            audio_mel=None,
            audio_mel_mask=None,
            audio_mask=None,
            audio_num_soft_tokens=None,
        )
        kwargs = ci.to_model_kwargs()
        assert "audio_mel" not in kwargs
        assert "audio_mel_mask" not in kwargs
        assert "audio_mask" not in kwargs
        assert "audio_num_soft_tokens" not in kwargs


# ---------------------------------------------------------------------------
# Mel spectrogram extraction (requires torchaudio)
# ---------------------------------------------------------------------------

@needs_torchaudio
class TestExtractMelSpectrogram:
    def test_shape_1s(self):
        """1 second of 16 kHz → ~100 frames, 128 mel bins."""
        from gemma4_pt_claude.audio_processing import extract_mel_spectrogram

        waveform = torch.randn(16000)
        mel = extract_mel_spectrogram(waveform)
        assert mel.ndim == 3
        assert mel.shape[0] == 1  # batch
        assert mel.shape[2] == 128
        # 16000 + 160 + 159 = 16319 padded samples
        # frames = (16319 - 320) / 160 + 1 = 100
        assert mel.shape[1] == 100

    def test_shape_batched(self):
        from gemma4_pt_claude.audio_processing import extract_mel_spectrogram

        waveform = torch.randn(3, 16000)
        mel = extract_mel_spectrogram(waveform)
        assert mel.shape[0] == 3
        assert mel.shape[2] == 128

    def test_output_finite(self):
        from gemma4_pt_claude.audio_processing import extract_mel_spectrogram

        waveform = torch.randn(16000)
        mel = extract_mel_spectrogram(waveform)
        assert torch.isfinite(mel).all()

    def test_silence(self):
        """Silent audio should produce small (negative) log-mel values."""
        from gemma4_pt_claude.audio_processing import extract_mel_spectrogram

        waveform = torch.zeros(16000)
        mel = extract_mel_spectrogram(waveform)
        # log(0 + 0.001) ≈ -6.9
        expected_val = math.log(0.001)
        assert torch.allclose(mel, torch.full_like(mel, expected_val), atol=0.1)


# ---------------------------------------------------------------------------
# preprocess_audio (requires torchaudio)
# ---------------------------------------------------------------------------

@needs_torchaudio
class TestPreprocessAudio:
    def test_basic(self):
        from gemma4_pt_claude.audio_processing import preprocess_audio

        waveform = torch.randn(16000)
        result = preprocess_audio(waveform)
        assert "audio_mel" in result
        assert "audio_mel_mask" in result
        assert result["audio_mel"].ndim == 3
        assert result["audio_mel_mask"].ndim == 2
        assert result["audio_mel_mask"].dtype == torch.bool
        assert result["audio_mel_mask"].all()

    def test_numpy_input(self):
        from gemma4_pt_claude.audio_processing import preprocess_audio

        waveform = np.random.randn(16000).astype(np.float32)
        result = preprocess_audio(waveform)
        assert result["audio_mel"].ndim == 3

    def test_mask_with_sequence_lengths(self):
        """Sequence lengths should produce correct frame-level mask."""
        from gemma4_pt_claude.audio_processing import preprocess_audio

        B = 2
        waveform = torch.randn(B, 16000)
        seq_lens = torch.tensor([8000, 16000])
        result = preprocess_audio(waveform, sequence_lengths=seq_lens)

        mask = result["audio_mel_mask"]
        # Second element all valid
        assert mask[1].all()
        # First element: 8000/160 = 50 valid frames
        assert mask[0].sum().item() == 50
        assert not mask[0, -1].item()


# ---------------------------------------------------------------------------
# AudioTransform (requires torchaudio)
# ---------------------------------------------------------------------------

@needs_torchaudio
class TestAudioTransform:
    def test_prepared_audio_fields(self):
        from gemma4_pt_claude.config import AudioConfig

        cfg = AudioConfig()
        transform = AudioTransform(cfg)
        waveform = torch.randn(16000)
        prepared = transform(waveform)

        assert isinstance(prepared, PreparedAudio)
        assert prepared.audio_mel.ndim == 2  # [T, 128]
        assert prepared.audio_mel.shape[1] == 128
        assert prepared.audio_mel_mask.ndim == 1
        # 16000 samples → valid_mel_frames = (16000 - 321) // 160 + 1 = 98
        # STFT produces 100 total frames; last 2 frames from zero-padding are invalid.
        assert prepared.audio_mel_mask.shape[0] == 100
        assert prepared.audio_mel_mask.sum().item() == 98
        assert not prepared.audio_mel_mask[-1].item()
        assert not prepared.audio_mel_mask[-2].item()
        assert prepared.num_soft_tokens == 25

    def test_stereo_waveform_is_folded_to_mono(self):
        from gemma4_pt_claude.config import AudioConfig

        cfg = AudioConfig()
        transform = AudioTransform(cfg)
        waveform = torch.randn(2, 16000)
        prepared = transform(waveform)

        assert isinstance(prepared, PreparedAudio)
        assert prepared.audio_mel.ndim == 2
        assert prepared.audio_mel.shape[1] == 128
