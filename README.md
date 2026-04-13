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

### Fresh model vs checkpoint load

Top-level factory functions behave differently depending on whether you construct
them normally or under a meta-device context:

- normal construction: creates a fully initialized model
- `with torch.device("meta")`: creates an unmaterialized model, intended for checkpoint loading

If you want a fresh random-init model:

```python
import gemma4_pt_claude as gemma4

model = gemma4.gemma4_e2b(text_only=True)
model.eval()
```

If you want to load a real checkpoint, the recommended path for large HuggingFace
checkpoints is meta construction plus streaming load:

```python
import torch
import gemma4_pt_claude as gemma4

weights_dir = "/path/to/hf-model-dir"

with torch.device("meta"):
    model = gemma4.gemma4_e2b(text_only=False)

gemma4.load_weights_streaming(
    model,
    weights_dir,
    format="hf",
    dtype=torch.bfloat16,
    device="cuda",
)
model.eval()
```

Use `text_only=True` to skip vision/audio encoders when you only need text.

### Text generation

For instruction-tuned checkpoints, prefer `chat()` for single-turn prompts:

```python
import torch
import gemma4_pt_claude as gemma4

weights_dir = "/path/to/hf-model-dir"
tokenizer = gemma4.Gemma4Tokenizer(weights_dir)

with torch.device("meta"):
    model = gemma4.gemma4_e2b(text_only=True)
gemma4.load_weights_streaming(
    model,
    weights_dir,
    format="hf",
    dtype=torch.bfloat16,
    device="cuda",
)
model.eval()

response = gemma4.chat(
    model,
    tokenizer,
    "Explain rotary position embeddings in plain English.",
    max_new_tokens=192,
    temperature=0.0,
)
print(response)
```

For lower-level control, build the chat prompt yourself and call `generate()`:

```python
prompt_ids = [tokenizer.BOS, tokenizer.START_OF_TURN]
prompt_ids += tokenizer.encode("user\nSummarize grouped-query attention.")
prompt_ids += [tokenizer.END_OF_TURN]
prompt_ids += tokenizer.encode("\n")
prompt_ids += [tokenizer.START_OF_TURN]
prompt_ids += tokenizer.encode("model\n")

tokens = torch.tensor([prompt_ids], device="cuda")
output = gemma4.generate(model, tokens, max_new_tokens=128, temperature=0.0)

gen_ids = output[0, len(prompt_ids):].tolist()
for i, tid in enumerate(gen_ids):
    if tid in {tokenizer.EOS, tokenizer.END_OF_TURN}:
        gen_ids = gen_ids[:i]
        break
print(tokenizer.decode(gen_ids))
```

## Multimodal: The Composer

The `Composer` class is the recommended multimodal entrypoint. It handles:

- image/audio preprocessing
- placeholder token injection
- Gemma 4 chat formatting

Prefer `compose_chat(...)` unless you already have a fully formatted prompt and
need the lower-level `compose(...)` path.

```python
import torch
from gemma4_pt_claude import Composer, Gemma4Tokenizer, gemma4_e2b, load_weights_streaming, generate

weights_dir = "/path/to/hf-model-dir"

with torch.device("meta"):
    model = gemma4_e2b()
load_weights_streaming(model, weights_dir, format="hf", dtype=torch.bfloat16, device="cuda")
model.eval()

tokenizer = Gemma4Tokenizer(weights_dir)
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

### Low-level image path (without Composer)

If you need full manual control, you can preprocess images directly and build
the multimodal prompt yourself:

```python
from PIL import Image
import torch
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

Practical note:

- `temperature=0.0` is greedy and is the safest setting for comparisons/regression tests
- very high temperatures with `top_k=0` and `top_p=1.0` will produce degenerate sampling quickly

## Weight Loading

Supports HuggingFace safetensors directories and native safetensors files.

For large HF checkpoints, prefer the streaming loader:

```python
# HuggingFace model directory
with torch.device("meta"):
    model = gemma4.gemma4_31b(text_only=False)
gemma4.load_weights_streaming(
    model,
    "/path/to/hf-model/",
    format="hf",
    dtype=torch.bfloat16,
    device="cuda",
)
model.eval()
```

The non-streaming loader remains available:

```python
# HuggingFace model directory (remaps weights through a temporary CPU dict)
gemma4.load_weights(model, "/path/to/hf-model/", format="hf")

# Single safetensors file (native format)
gemma4.load_weights(model, "model.safetensors")
```

Both loaders handle:

- HF key remapping
- gate/up projection merging
- MoE routing/expert remaps
- vision/audio tensor name conversion

### Best Practices

- Use meta construction plus `load_weights_streaming(...)` for large HF checkpoints.
- Use normal construction only when you want a fresh randomly initialized model.
- Call `model.eval()` after loading.
- Use `compose_chat(...)` for multimodal prompts unless you already have a fully templated prompt.
- Use `text_only=True` if you do not need vision/audio towers.
- Keep `temperature=0.0` for deterministic eval and debugging.

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
