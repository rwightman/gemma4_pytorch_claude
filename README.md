# gemma4-pt-claude

A clean, standalone PyTorch implementation of Google's **Gemma 4** model family — ported layer-by-layer from the official JAX reference with numerical verification at every step.

Text, vision, and audio. No mystery abstractions. Just PyTorch and matrix multiplies.

## Supported Models

| Variant | Layers | Embed | Heads | KV Heads | Attention | Notable Features |
|---------|--------|-------|-------|----------|-----------|-----------------|
| **E2B** | 35 | 1536 | 8 | 1 | 4:1 local:global | PLI, KV sharing, vision, audio |
| **E4B** | 42 | 2560 | 8 | 2 | 5:1 local:global | PLI, KV sharing, V-norm, vision, audio |
| **31B** | 60 | 5376 | 32 | 16 (4 global) | 5:1 local:global | K=V global, bidirectional vision |
| **26B-A4B** | 30 | 2816 | 16 | 8 (2 global) | 5:1 local:global | MoE (128 experts, top-8) |

## Installation

```bash
pip install -e .
```

Core dependencies are `torch`, `sentencepiece`, and `safetensors`.

Optional extras:

```bash
pip install -e ".[audio]"    # adds torchaudio (for audio resampling)
pip install -e ".[convert]"  # adds jax + orbax (for JAX checkpoint conversion)
```

## Quickstart

### Build a model and load weights

```python
import torch
import gemma4_pt_claude as gemma4

model = gemma4.gemma4_e2b()
gemma4.load_weights(model, "/path/to/weights/", format="hf")
model.to("cuda", dtype=torch.bfloat16).eval()
```

Use `text_only=True` to skip vision/audio encoders if you only need text:

```python
model = gemma4.gemma4_e2b(text_only=True)
```

### Text generation

```python
tokenizer = gemma4.Gemma4Tokenizer("/path/to/weights/")
tokens = tokenizer.encode("The meaning of life is", add_bos=True)

input_ids = torch.tensor([tokens], device="cuda")
output = gemma4.generate(model, input_ids, max_new_tokens=128, temperature=0.0)
print(tokenizer.decode(output[0].tolist()))
```

### Chat (single-turn)

```python
response = gemma4.chat(model, tokenizer, "Explain RoPE embeddings like I'm five.")
print(response)
```

`chat()` wraps your prompt in the Gemma 4 chat template and handles generation + decoding.

## Multimodal: The Composer

The `Composer` class is the recommended way to do multimodal inference. It handles image/audio preprocessing, placeholder token injection, and chat template formatting in one call.

```python
from gemma4_pt_claude import Composer, Gemma4Tokenizer, gemma4_e2b, load_weights, generate

model = gemma4_e2b()
load_weights(model, "/path/to/weights/", format="hf")
model.to("cuda", dtype=torch.bfloat16).eval()

tokenizer = Gemma4Tokenizer("/path/to/weights/")
composer = Composer(tokenizer, model.cfg)
```

### Image captioning

```python
from PIL import Image

img = Image.open("photo.jpg")
composed = composer.compose_chat("Describe this image in detail.", images=[img])
kwargs = composed.to_model_kwargs(device="cuda")

output = generate(
    model,
    kwargs.pop("tokens"),
    max_new_tokens=256,
    temperature=0.0,
    **kwargs,
)

# Decode generated tokens
prompt_len = composed.input_ids.shape[1]
gen_ids = output[0, prompt_len:].tolist()
for i, tid in enumerate(gen_ids):
    if tid in {tokenizer.EOS, tokenizer.END_OF_TURN}:
        gen_ids = gen_ids[:i]
        break
print(tokenizer.decode(gen_ids))
```

The Composer automatically:
- Resizes and patchifies the image for the vision encoder
- Inserts `<|image|>` markers with the correct number of soft-token placeholders
- Builds the image mask for the model's multimodal embedding merge

### Audio transcription

Requires `torchaudio` for resampling (install with `pip install -e ".[audio]"`).

```python
import torchaudio

waveform, sample_rate = torchaudio.load("recording.wav")
waveform = waveform.mean(dim=0)  # stereo to mono

composed = composer.compose_chat(
    "Transcribe this audio clip exactly.",
    audios=[waveform],
    audio_sample_rates=sample_rate,
)
kwargs = composed.to_model_kwargs(device="cuda")

output = generate(
    model,
    kwargs.pop("tokens"),
    max_new_tokens=256,
    temperature=0.0,
    **kwargs,
)

prompt_len = composed.input_ids.shape[1]
gen_ids = output[0, prompt_len:].tolist()
for i, tid in enumerate(gen_ids):
    if tid in {tokenizer.EOS, tokenizer.END_OF_TURN}:
        gen_ids = gen_ids[:i]
        break
print(tokenizer.decode(gen_ids))
```

The Composer handles mel spectrogram extraction, resampling to 16 kHz, and audio soft-token placeholder injection.

### Multiple modalities in one prompt

```python
composed = composer.compose_chat(
    "Describe what you see and hear.",
    images=[img],
    audios=[waveform],
    audio_sample_rates=16000,
)
```

### Manual image pipeline (without Composer)

If you need lower-level control, you can preprocess images directly:

```python
from PIL import Image
import gemma4_pt_claude as gemma4

img = Image.open("photo.jpg")
data = gemma4.preprocess_images([img], model.cfg.vision)
n = data["num_soft_tokens_per_image"][0]

tok = gemma4.Gemma4Tokenizer
prompt_ids = [tok.BOS, tok.START_OF_TURN]
prompt_ids += tokenizer.encode("user\n")
prompt_ids += [tok.START_OF_IMAGE] + [tok.IMAGE_PLACEHOLDER] * n + [tok.END_OF_IMAGE]
prompt_ids += tokenizer.encode("\nDescribe this image in detail.")
prompt_ids += [tok.END_OF_TURN]
prompt_ids += tokenizer.encode("\n")
prompt_ids += [tok.START_OF_TURN]
prompt_ids += tokenizer.encode("model\n")

tokens = torch.tensor([prompt_ids], device="cuda")
image_mask = (tokens == tok.IMAGE_PLACEHOLDER)

output = gemma4.generate(
    model, tokens,
    max_new_tokens=256,
    temperature=0.0,
    pixel_values=data["pixel_values"].to("cuda", dtype=torch.bfloat16),
    image_position_ids=data["image_position_ids"].to("cuda"),
    image_mask=image_mask,
)
```

## Sampling Options

```python
output = gemma4.generate(
    model, tokens,
    max_new_tokens=256,
    temperature=0.7,     # 0 = greedy
    top_k=50,            # 0 = disabled
    top_p=0.95,          # 1.0 = disabled
)
```

## Weight Loading

Supports HuggingFace safetensors directories and JAX Orbax checkpoints (after conversion):

```python
# HuggingFace model directory (auto-detects safetensors files)
gemma4.load_weights(model, "/path/to/hf-model/", format="hf")

# Single safetensors file (native format)
gemma4.load_weights(model, "model.safetensors")
```

HuggingFace weight conversion handles gate/up projection merging, key remapping, and everything else automatically.

### Converting JAX checkpoints

Convert Orbax (JAX) checkpoints to our safetensors format:

```bash
gemma4-convert --checkpoint gs://gemma-data/checkpoints/gemma4-e2b-it \
               --variant e2b \
               --output weights/gemma4-e2b.safetensors
```

## Attention Backend

Two attention implementations, controlled by `attn_impl` in the text config:

- **`"sdpa"`** (default) — `F.scaled_dot_product_attention`. Faster, less memory, benefits from FlashAttention/cuDNN when available.
- **`"eager"`** — Manual einsum path. Required when using `attn_logits_soft_cap` (falls back automatically). Useful for debugging.

## Architecture Overview

```
Gemma4Model
├── text_decoder (TextDecoder)
│   ├── embedder (Embedder)
│   │   ├── token_embedding
│   │   └── per_layer_input (PLI) — small per-token-per-layer embeddings
│   ├── blocks[] (TransformerBlock)
│   │   ├── pre_attn_norm → Attention → post_attn_norm
│   │   ├── pre_ffw_norm → GatedMLP (or MoE + dense branch) → post_ffw_norm
│   │   ├── pli_mapping — gates + projects PLI back to embed_dim
│   │   └── skip_scale — learned scalar residual multiplier
│   └── final_norm
├── vision_encoder (VisionEncoder) — patch embed, 2-D RoPE, spatial pooling
│   └── embed_vision (VisionEmbedder) — RMSNorm + projection to text dim
├── audio_encoder (AudioEncoder) — conformer with SSCP frontend
│   └── embed_audio (AudioEmbedder) — RMSNorm + projection to text dim
└── composer (Composer) — tokenization + modality transforms + placeholder injection
```

Key architectural features:

- **GQA** — Grouped-query attention with configurable head counts per layer type
- **Sliding window + global** — Most layers use local sliding window; every Nth layer is global
- **QK-norm** — RMSNorm on queries and keys replaces 1/sqrt(d) scaling
- **Per-layer input (PLI)** — Lightweight per-token-per-layer embeddings (E2B, E4B)
- **KV sharing** — Later layers reuse KV from earlier layers, no redundant cache allocation (E2B, E4B)
- **K=V** — Key and value projections share weights for global layers (31B, 26B-A4B)
- **MoE** — 128 experts with top-8 routing + parallel dense branch (26B-A4B)
- **Skip scale** — Learned scalar multiplied at the end of each block
- **Vision encoder** — ViT-style with factorised 2-D positional embeddings and bidirectional attention
- **Audio encoder** — Conformer stack with mel spectrogram frontend and relative position embeddings

## Project Structure

```
src/gemma4_pt_claude/
├── config.py            # All dataclass configs (TextConfig, VisionConfig, AudioConfig, etc.)
├── layers.py            # RMSNorm, GatedMLP, RoPE, ClippedLinear
├── attention.py         # GQA attention with SDPA/eager backends, sliding window, KV cache
├── moe.py               # Router + experts + MoE layer
├── transformer.py       # TransformerBlock, Embedder, PLI, TextDecoder
├── vision_encoder.py    # Vision encoder (patch embed, 2-D RoPE, spatial pooling)
├── image_processing.py  # Image preprocessing (resize, patchify, pad)
├── audio_encoder.py     # Conformer audio encoder
├── audio_processing.py  # Mel spectrogram extraction (matching JAX GemaxMelFilterbank)
├── model.py             # Top-level Gemma4Model + multimodal embedding merge
├── generate.py          # Autoregressive generation with KV cache
├── composer.py          # Multimodal Composer (tokenize + transform + inject)
├── tokenizer.py         # SentencePiece + HuggingFace tokenizer backends
├── load.py              # Weight loading (HF safetensors + native)
├── factory.py           # gemma4_e2b(), gemma4_e4b(), gemma4_31b(), gemma4_26b_a4b()
└── convert.py           # JAX Orbax → safetensors checkpoint conversion
```

## Tests

```bash
pytest tests/                    # full suite (~160 tests)
pytest tests/ -k "vision"       # filter by keyword
pytest tests/test_model.py -v   # single file, verbose
```

## Numerical Verification

Every module was verified against the JAX reference implementation with tight tolerances:

| Module | Max Abs Error |
|--------|--------------|
| RMSNorm | 4.77e-07 |
| RoPE | 2.38e-07 |
| GatedMLP | 4.86e-09 |
| Attention (non-GQA) | 1.01e-07 |
| Attention (GQA) | 2.53e-07 |
| Attention (K=V) | 8.94e-08 |
| Embedder | 4.66e-10 |
| PLI | 4.17e-07 |
| TransformerBlock (dense) | 1.12e-05 |
| TransformerBlock (GQA+sliding) | 2.31e-05 |

End-to-end generation verified on E2B with both HuggingFace and Orbax weights (text, image captioning, and audio transcription all produce correct output).

## License

Apache 2.0
