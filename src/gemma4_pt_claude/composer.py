"""Multimodal Composer — coordinates modality transforms and token injection.

Usage::

    from gemma4_pt_claude import gemma4_e2b, Gemma4Tokenizer
    from gemma4_pt_claude.composer import Composer

    model = gemma4_e2b()
    tokenizer = Gemma4Tokenizer("/path/to/hf-model-dir/")
    composer = Composer(tokenizer, model.cfg)

    # Single-turn text + image
    composed = composer.compose_chat("Describe this image", images=[pil_image])
    logits, cache = model(**composed.to_model_kwargs(device="cuda"))

    # Single-turn text + audio
    composed = composer.compose_chat("Transcribe this audio", audios=[waveform_tensor])
    logits, cache = model(**composed.to_model_kwargs(device="cuda"))
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore[assignment, misc]

from .config import AudioConfig, Gemma4Config, VisionConfig
from .image_processing import pad_to_max_patches, preprocess_image
from .tokenizer import Gemma4Tokenizer


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PreparedImage:
    """Result of transforming a single image for the vision encoder."""

    pixel_values: torch.Tensor    # [max_patches, patch_dim]
    position_ids: torch.Tensor    # [max_patches, 2]
    num_soft_tokens: int


@dataclass
class PreparedAudio:
    """Result of transforming a single audio waveform for the audio encoder."""

    audio_mel: torch.Tensor       # [T, 128] log-mel spectrogram
    audio_mel_mask: torch.Tensor  # [T] bool — True for valid frames
    num_soft_tokens: int          # number of placeholder tokens to insert


@dataclass
class ComposedInput:
    """Ready-to-forward batch — pass directly to ``model.forward()``."""

    input_ids: torch.Tensor                   # [B, L]
    pixel_values: torch.Tensor | None         # [B, max_patches, patch_dim]
    image_position_ids: torch.Tensor | None   # [B, max_patches, 2]
    image_mask: torch.Tensor | None           # [B, L] — True at soft-token placeholder positions
    audio_mel: torch.Tensor | None            # [B, T, 128]
    audio_mel_mask: torch.Tensor | None       # [B, T] bool
    audio_mask: torch.Tensor | None           # [B, L] bool — True at audio placeholder positions
    audio_num_soft_tokens: torch.Tensor | None  # [B_audio] int

    def to_model_kwargs(
            self,
            device: torch.device | str | None = None,
    ) -> dict:
        """Return a kwargs dict suitable for ``model.forward(**kwargs)``."""
        out: dict = {"tokens": self.input_ids}
        if self.pixel_values is not None:
            out["pixel_values"] = self.pixel_values
        if self.image_position_ids is not None:
            out["image_position_ids"] = self.image_position_ids
        if self.image_mask is not None:
            out["image_mask"] = self.image_mask
        if self.audio_mel is not None:
            out["audio_mel"] = self.audio_mel
        if self.audio_mel_mask is not None:
            out["audio_mel_mask"] = self.audio_mel_mask
        if self.audio_mask is not None:
            out["audio_mask"] = self.audio_mask
        if self.audio_num_soft_tokens is not None:
            out["audio_num_soft_tokens"] = self.audio_num_soft_tokens
        if device is not None:
            out = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in out.items()}
        return out


# ---------------------------------------------------------------------------
# ImageTransform
# ---------------------------------------------------------------------------

class ImageTransform:
    """Transform a single image into patches + position IDs.

    Accepts a PIL Image or ``[C, H, W]`` tensor and returns a ``PreparedImage``.
    Handles uint8 rescaling internally (via ``preprocess_image``).
    """

    def __init__(self, config: VisionConfig):
        self.patch_size = config.patch_size
        self.max_patches = config.max_patches
        self.pooling_kernel_size = config.pooling_kernel_size

    def __call__(self, image: "PILImage.Image | torch.Tensor") -> PreparedImage:
        patches, position_ids, num_soft_tokens = preprocess_image(
            image, self.patch_size, self.max_patches, self.pooling_kernel_size,
        )
        patches, position_ids = pad_to_max_patches(patches, position_ids, self.max_patches)
        return PreparedImage(
            pixel_values=patches,
            position_ids=position_ids,
            num_soft_tokens=num_soft_tokens,
        )


# ---------------------------------------------------------------------------
# AudioTransform
# ---------------------------------------------------------------------------

def _compute_audio_soft_tokens(num_samples: int, sample_rate: int = 16000) -> int:
    """Compute the number of audio soft tokens from waveform length.

    Mirrors the JAX ``_gemma4_sampler.py`` formula:
    mel frames → 2× subsampling (stride 2, kernel 3, padding 1 each side).
    """
    frame_length = int(round(sample_rate * 20.0 / 1000.0))  # 320
    hop_length = int(round(sample_rate * 10.0 / 1000.0))    # 160
    frame_size_for_unfold = frame_length + 1                 # 321
    num_mel_frames = (num_samples - frame_size_for_unfold) // hop_length + 1

    t = num_mel_frames
    for _ in range(2):
        t_padded = t + 2
        t = (t_padded - 3) // 2 + 1
    return max(t, 0)


class AudioTransform:
    """Transform a single audio waveform into mel spectrogram + mask.

    Accepts a 1-D or 2-D tensor (``[samples]`` or ``[channels, samples]``)
    and returns a ``PreparedAudio``.  If the input sample rate differs from
    the target (16 kHz), the waveform is resampled automatically.
    """

    TARGET_SAMPLE_RATE = 16_000

    def __init__(self, config: AudioConfig, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_seq_length = 750  # cap from JAX audio_seq_length

    def __call__(
            self,
            waveform: torch.Tensor,
            sample_rate: int | None = None,
    ) -> PreparedAudio:
        """Transform waveform to ``PreparedAudio``.

        Args:
            waveform: ``[samples]`` or ``[channels, samples]`` audio tensor.
            sample_rate: Actual sample rate of *waveform*.  When provided and
                different from ``TARGET_SAMPLE_RATE`` (16 kHz), the waveform
                is resampled before mel extraction.  Defaults to the rate
                passed at construction time.
        """
        from .audio_processing import extract_mel_spectrogram, to_float32, to_mono_waveform

        if sample_rate is None:
            sample_rate = self.sample_rate

        waveform = to_float32(waveform)
        if waveform.dim() == 2:
            waveform = to_mono_waveform(waveform)

        # Resample to 16 kHz if needed
        if sample_rate != self.TARGET_SAMPLE_RATE:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sample_rate, self.TARGET_SAMPLE_RATE,
            ).squeeze(0)

        num_samples = waveform.shape[0]

        # Compute soft token count from 16 kHz sample count
        num_soft_tokens = min(
            _compute_audio_soft_tokens(num_samples, self.TARGET_SAMPLE_RATE),
            self.max_seq_length,
        )

        # Extract mel spectrogram: [1, T, 128] → [T, 128]
        mel = extract_mel_spectrogram(
            waveform.unsqueeze(0), sample_rate=self.TARGET_SAMPLE_RATE,
        )
        mel = mel.squeeze(0)  # [T, 128]

        # Derive valid mel frame count (STFT zero-padding can produce extra frames)
        frame_size_for_unfold = 321  # frame_length (320) + 1
        hop = 160
        valid_mel_frames = max((num_samples - frame_size_for_unfold) // hop + 1, 0)
        mask = torch.zeros(mel.shape[0], dtype=torch.bool)
        mask[:valid_mel_frames] = True

        return PreparedAudio(
            audio_mel=mel,
            audio_mel_mask=mask,
            num_soft_tokens=num_soft_tokens,
        )


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

_IMAGE_MARKER = "<|image|>"
_AUDIO_MARKER = "<|audio|>"

# Internal soft-token placeholders (negative, not in vocab).
# Positions where encoder embeddings will be injected.
# Matches JAX _token_utils.SOFT_TOKEN_PLACEHOLDER / AUDIO_SOFT_TOKEN_PLACEHOLDER.
_IMAGE_SOFT_TOKEN = -2
_AUDIO_SOFT_TOKEN = -4
_DOUBLE_NEWLINE_TOKEN = 108


def _broadcast_sample_rates(
        rates: list[int] | int | None,
        n: int,
) -> list[int]:
    """Expand *rates* to a list of length *n*, defaulting to 16000."""
    if rates is None:
        return [16_000] * n
    if isinstance(rates, int):
        return [rates] * n
    if len(rates) != n:
        raise ValueError(
            f"audio_sample_rates has {len(rates)} entries but {n} audios were given"
        )
    return list(rates)


class Composer:
    """Coordinate tokenization, modality transforms, and placeholder injection.

    Args:
        tokenizer: A ``Gemma4Tokenizer`` instance.
        config: A ``Gemma4Config`` instance.
    """

    def __init__(self, tokenizer: Gemma4Tokenizer, config: Gemma4Config):
        self.tokenizer = tokenizer
        self.config = config
        self.image_transform = ImageTransform(config.vision) if config.vision else None
        self.audio_transform = AudioTransform(config.audio) if config.audio else None

    def compose(
            self,
            text: str,
            *,
            images: "list[PILImage.Image] | None" = None,
            audios: list[torch.Tensor] | None = None,
            audio_sample_rates: list[int] | int | None = None,
    ) -> ComposedInput:
        """Tokenize text, transform images/audio, inject boundary + placeholder tokens.

        If ``text`` contains ``<|image|>`` or ``<|audio|>`` markers, each is
        expanded into boundary + placeholder tokens using the corresponding
        entry from ``images`` or ``audios``.

        Args:
            text: Already-formatted text (with chat template if desired).
            images: Optional list of PIL images, one per ``<|image|>`` marker.
            audios: Optional list of waveform tensors, one per ``<|audio|>`` marker.
            audio_sample_rates: Sample rate(s) of the audio waveforms.  A single
                int applies to all clips; a list must match ``len(audios)``.
                Defaults to 16000 (no resampling).

        Returns:
            A ``ComposedInput`` ready for ``model.forward()``.
        """
        images = images or []
        audios = audios or []

        if images and self.image_transform is None:
            raise ValueError("Cannot compose images — model has no vision config")
        if audios and self.audio_transform is None:
            raise ValueError("Cannot compose audio — model has no audio config")

        rates = _broadcast_sample_rates(audio_sample_rates, len(audios))

        # Transform modalities
        prepared_images = [self.image_transform(img) for img in images]
        prepared_audios = [
            self.audio_transform(aud, sample_rate=sr)
            for aud, sr in zip(audios, rates)
        ]

        # Tokenize, expanding markers
        token_ids = self._tokenize_with_modalities(text, prepared_images, prepared_audios)
        return self._build_output(token_ids, prepared_images, prepared_audios)

    def compose_chat(
            self,
            prompt: str,
            *,
            images: "list[PILImage.Image] | None" = None,
            audios: list[torch.Tensor] | None = None,
            audio_sample_rates: list[int] | int | None = None,
    ) -> ComposedInput:
        """Wrap prompt in Gemma4 chat template, then compose.

        For image/audio inputs, prepend markers to the user message
        automatically unless markers are already present.

        Args:
            prompt: User message text.
            images: Optional list of PIL images.
            audios: Optional list of waveform tensors.
            audio_sample_rates: Sample rate(s) of the audio waveforms.  A single
                int applies to all clips; a list must match ``len(audios)``.
                Defaults to 16000 (no resampling).
        """
        images = images or []
        audios = audios or []

        if images and self.image_transform is None:
            raise ValueError("Cannot compose images — model has no vision config")
        if audios and self.audio_transform is None:
            raise ValueError("Cannot compose audio — model has no audio config")

        rates = _broadcast_sample_rates(audio_sample_rates, len(audios))

        # Build user turn text, inserting markers if not present
        user_text = prompt
        if audios and _AUDIO_MARKER not in prompt:
            markers = "\n".join([_AUDIO_MARKER] * len(audios))
            user_text = markers + "\n" + user_text
        if images and _IMAGE_MARKER not in prompt:
            markers = "\n".join([_IMAGE_MARKER] * len(images))
            user_text = markers + "\n" + user_text

        # Transform modalities
        prepared_images = [self.image_transform(img) for img in images]
        prepared_audios = [
            self.audio_transform(aud, sample_rate=sr)
            for aud, sr in zip(audios, rates)
        ]

        # Tokenize with chat template
        tok = self.tokenizer
        formatted = f"user\n{user_text}"
        ids: list[int] = [tok.BOS, tok.START_OF_TURN]

        # Tokenize segment-by-segment, expanding both image and audio markers
        ids.extend(self._expand_markers(formatted, prepared_images, prepared_audios))

        ids.extend([tok.END_OF_TURN])
        ids.extend(tok.encode("\n"))
        ids.extend([tok.START_OF_TURN])
        ids.extend(tok.encode("model\n"))

        return self._build_output(ids, prepared_images, prepared_audios)

    def _build_output(
            self,
            token_ids: list[int],
            prepared_images: list[PreparedImage],
            prepared_audios: list[PreparedAudio],
    ) -> ComposedInput:
        """Build ``ComposedInput`` from token IDs and prepared modalities."""
        input_ids = torch.tensor([token_ids], dtype=torch.long)

        pixel_values = None
        image_position_ids = None
        image_mask = None

        if prepared_images:
            pixel_values = torch.stack([p.pixel_values for p in prepared_images])
            image_position_ids = torch.stack([p.position_ids for p in prepared_images])
            image_mask = (input_ids == _IMAGE_SOFT_TOKEN)

        audio_mel = None
        audio_mel_mask = None
        audio_mask = None
        audio_num_soft_tokens = None

        if prepared_audios:
            # Pad mel spectrograms to same length for batching
            max_t = max(p.audio_mel.shape[0] for p in prepared_audios)
            padded_mels = []
            padded_masks = []
            for p in prepared_audios:
                t = p.audio_mel.shape[0]
                pad_len = max_t - t
                padded_mels.append(torch.nn.functional.pad(p.audio_mel, (0, 0, 0, pad_len)))
                padded_masks.append(torch.nn.functional.pad(p.audio_mel_mask, (0, pad_len), value=False))
            audio_mel = torch.stack(padded_mels)           # [N_audio, T, 128]
            audio_mel_mask = torch.stack(padded_masks)     # [N_audio, T]
            audio_mask = (input_ids == _AUDIO_SOFT_TOKEN)
            audio_num_soft_tokens = torch.tensor(
                [p.num_soft_tokens for p in prepared_audios],
                dtype=torch.long,
            )

        return ComposedInput(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            image_mask=image_mask,
            audio_mel=audio_mel,
            audio_mel_mask=audio_mel_mask,
            audio_mask=audio_mask,
            audio_num_soft_tokens=audio_num_soft_tokens,
        )

    def _expand_markers(
            self,
            text: str,
            prepared_images: list[PreparedImage],
            prepared_audios: list[PreparedAudio],
    ) -> list[int]:
        """Tokenize text, expanding ``<|image|>`` and ``<|audio|>`` markers."""
        tok = self.tokenizer
        ids: list[int] = []
        img_idx = 0
        aud_idx = 0

        # Split on both marker types while preserving order
        pos = 0
        markers: list[tuple[int, str]] = []
        for marker in (_IMAGE_MARKER, _AUDIO_MARKER):
            start = 0
            while True:
                idx = text.find(marker, start)
                if idx == -1:
                    break
                markers.append((idx, marker))
                start = idx + len(marker)
        markers.sort(key=lambda x: x[0])

        for marker_pos, marker_type in markers:
            # Tokenize text before this marker
            segment = text[pos:marker_pos]
            if segment:
                ids.extend(tok.encode(segment))
            pos = marker_pos + len(marker_type)

            if marker_type == _IMAGE_MARKER and img_idx < len(prepared_images):
                ids.append(_DOUBLE_NEWLINE_TOKEN)
                ids.append(tok.START_OF_IMAGE)
                ids.extend([_IMAGE_SOFT_TOKEN] * prepared_images[img_idx].num_soft_tokens)
                ids.append(tok.END_OF_IMAGE)
                ids.append(_DOUBLE_NEWLINE_TOKEN)
                img_idx += 1
            elif marker_type == _AUDIO_MARKER and aud_idx < len(prepared_audios):
                ids.append(tok.START_OF_AUDIO)
                ids.extend([_AUDIO_SOFT_TOKEN] * prepared_audios[aud_idx].num_soft_tokens)
                ids.append(tok.END_OF_AUDIO)
                aud_idx += 1

        # Remaining text after last marker
        if pos < len(text):
            remaining = text[pos:]
            if remaining:
                ids.extend(tok.encode(remaining))

        return ids

    def _tokenize_with_modalities(
            self,
            text: str,
            prepared_images: list[PreparedImage],
            prepared_audios: list[PreparedAudio],
    ) -> list[int]:
        """Tokenize text, expanding image and audio markers."""
        return self._expand_markers(text, prepared_images, prepared_audios)
