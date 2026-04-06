"""Audio preprocessing — wav-to-log-mel spectrogram extraction.

Matches the JAX ``GemaxMelFilterbank`` exactly:
- sample_rate=16000, win_length=320, hop_length=160
- n_fft=512, periodic Hann window (nonzero variant), magnitude (power=1)
- HTK mel scale, 128 bins, 0–8 kHz
- Semi-causal padding: ``pad(waveform, (160, 159))`` then STFT with ``center=False``
- ``log(mel + 0.001)``

Requires ``torchaudio`` (install via ``pip install gemma4-pt-claude[audio]``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass

try:
    import torchaudio  # noqa: F401
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False


def _require_torchaudio() -> None:
    if not _HAS_TORCHAUDIO:
        raise ImportError(
            "torchaudio is required for audio processing but not installed. "
            "Install it with: pip install gemma4-pt-claude[audio]"
        )


# ---------------------------------------------------------------------------
# Hann window (JAX nonzero variant)
# ---------------------------------------------------------------------------

def _hann_window_nonzero(length: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Periodic Hann window matching JAX's nonzero variant.

    JAX formula: ``0.5 - 0.5 * cos(2*pi*(n + 0.5) / N)``
    This ensures the window never touches zero at endpoints.
    """
    n = torch.arange(length, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * (n + 0.5) / length)


# ---------------------------------------------------------------------------
# HTK mel scale helpers
# ---------------------------------------------------------------------------

def _hertz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hertz(mels: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def _mel_filterbank(
        n_mels: int,
        n_fft: int,
        sample_rate: int,
        f_min: float,
        f_max: float,
) -> torch.Tensor:
    """Triangular mel filterbank matrix ``[n_mels, n_fft // 2 + 1]``."""
    n_freqs = n_fft // 2 + 1

    # Mel-spaced center frequencies
    mel_min = _hertz_to_mel(torch.tensor(f_min))
    mel_max = _hertz_to_mel(torch.tensor(f_max))
    mel_points = torch.linspace(mel_min.item(), mel_max.item(), n_mels + 2)
    hz_points = _mel_to_hertz(mel_points)

    # FFT bin frequencies
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)

    # Build triangular filters
    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        lo, center, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        # Rising slope
        up = (fft_freqs - lo) / (center - lo + 1e-10)
        # Falling slope
        down = (hi - fft_freqs) / (hi - center + 1e-10)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)

    return fb


# ---------------------------------------------------------------------------
# dtype normalisation
# ---------------------------------------------------------------------------

def to_float32(audio: torch.Tensor) -> torch.Tensor:
    """Normalise audio to float32.

    Handles int16 (scale by 1/32768) and uint8 (scale by 1/128, center).
    Float tensors are returned as-is (cast to float32 if needed).
    """
    if audio.dtype == torch.int16:
        return audio.float() / 32768.0
    if audio.dtype == torch.uint8:
        return (audio.float() - 128.0) / 128.0
    return audio.float()


def to_mono_waveform(audio: torch.Tensor) -> torch.Tensor:
    """Fold a 1-D or 2-D waveform tensor down to mono.

    For 2-D inputs, accepts either ``[channels, samples]`` or
    ``[samples, channels]`` and averages over the likely channel axis.
    """
    if audio.dim() == 1:
        return audio
    if audio.dim() != 2:
        raise ValueError(f"Expected 1D or 2D waveform tensor, got shape {tuple(audio.shape)}")

    if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
        return audio.mean(dim=0)
    if audio.shape[1] <= 8 and audio.shape[0] > audio.shape[1]:
        return audio.mean(dim=1)
    return audio.mean(dim=0)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_mel_spectrogram(
        waveform: torch.Tensor,
        sample_rate: int = 16000,
) -> torch.Tensor:
    """Extract log-mel spectrogram matching JAX ``GemaxMelFilterbank``.

    Args:
        waveform: ``[B, samples]`` or ``[samples]`` float tensor.
        sample_rate: Expected sample rate (default 16 kHz).

    Returns:
        ``[B, frames, 128]`` log-mel spectrogram.
    """
    _require_torchaudio()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    waveform = to_float32(waveform)

    win_length = 320
    hop_length = 160
    n_fft = 512
    n_mels = 128
    f_min = 0.0
    f_max = float(sample_rate) / 2.0  # 8000 Hz for 16 kHz
    constant = 0.001

    # Semi-causal padding: pad_left = win_length - hop_length, pad_right = hop_length - 1
    pad_left = win_length - hop_length   # 160
    pad_right = hop_length - 1           # 159
    waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), value=0.0)

    # Pre-compute JAX-matching window
    window = _hann_window_nonzero(win_length, dtype=waveform.dtype).to(waveform.device)

    # STFT — align_to_window=True makes torch step by win_length (320) instead
    # of n_fft (512), matching JAX's sl_signal_frame framing exactly.
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
        align_to_window=True,
    )
    # stft: [B, n_fft//2+1, frames]

    # Magnitude spectrogram (power=1)
    magnitude = stft.abs()  # [B, n_fft//2+1, frames]

    # Mel filterbank: [n_mels, n_fft//2+1]
    mel_fb = _mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)
    mel_fb = mel_fb.to(dtype=magnitude.dtype, device=magnitude.device)

    # Apply filterbank: [B, n_mels, frames]
    mel = torch.matmul(mel_fb, magnitude)

    # Transpose to [B, frames, n_mels]
    mel = mel.transpose(1, 2)

    # Log compression
    mel = torch.log(mel + constant)

    return mel


# ---------------------------------------------------------------------------
# High-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_audio(
        waveform: torch.Tensor | np.ndarray,
        sample_rate: int = 16000,
        sequence_lengths: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Preprocess audio waveform(s) into mel spectrograms with masks.

    Args:
        waveform: ``[B, samples]`` or ``[samples]`` audio data.
            Accepts torch Tensor or numpy array.
        sample_rate: Sample rate in Hz (default 16000).
        sequence_lengths: ``[B]`` int tensor of valid sample counts per batch
            element (for masking padded regions). If None, all frames are valid.

    Returns:
        Dict with:
        - ``"audio_mel"``: ``[B, T, 128]`` log-mel spectrogram
        - ``"audio_mel_mask"``: ``[B, T]`` bool — True for valid frames
    """
    _require_torchaudio()

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    waveform = to_float32(waveform)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel = extract_mel_spectrogram(waveform, sample_rate=sample_rate)
    B, T, _ = mel.shape

    if sequence_lengths is not None:
        # Convert sample lengths to frame counts
        # Each frame covers hop_length=160 samples, with semi-causal padding
        # frame_count = ceil(num_samples / hop_length)
        hop_length = 160
        frame_lengths = (sequence_lengths + hop_length - 1) // hop_length  # [B]
        frame_lengths = frame_lengths.clamp(max=T)
        # Build mask: True for valid frames
        frame_idx = torch.arange(T, device=mel.device).unsqueeze(0)  # [1, T]
        mask = frame_idx < frame_lengths.unsqueeze(1)  # [B, T]
    else:
        mask = torch.ones(B, T, dtype=torch.bool, device=mel.device)

    return {"audio_mel": mel, "audio_mel_mask": mask}
