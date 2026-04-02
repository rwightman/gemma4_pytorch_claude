"""Autoregressive generation with KV cache."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import Attention
from .config import Gemma4Config, make_attention_pattern
from .model import Gemma4Model


def _sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample a single token from logits ``[B, V]``."""
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k > 0:
        vals, _ = logits.topk(top_k)
        logits[logits < vals[:, -1:]] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(descending=True)
        cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = cumprobs - sorted_logits.softmax(dim=-1) > top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def init_cache(
    cfg: Gemma4Config,
    batch_size: int,
    cache_length: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
) -> dict:
    """Initialise KV cache for all layers."""
    text = cfg.text
    attn_types = make_attention_pattern(text.attention_pattern, text.num_layers)
    cache = {}
    for i in range(text.num_layers):
        cache[f"layer_{i}"] = Attention.init_cache(
            cache_length=cache_length,
            num_kv_heads=text.num_kv_heads,
            head_dim=text.head_dim,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
    return cache


@torch.no_grad()
def generate(
    model: Gemma4Model,
    tokens: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    cache_length: int | None = None,
    # Optional multimodal inputs (only for prefill)
    image_patches: torch.Tensor | None = None,
    image_mask: torch.Tensor | None = None,
    audio_mel: torch.Tensor | None = None,
    audio_mel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autoregressive generation with KV cache.

    Args:
        model: A ``Gemma4Model`` instance.
        tokens: ``[B, L]`` — prompt token ids.
        max_new_tokens: number of tokens to generate.
        temperature: sampling temperature (0 = greedy).
        top_k: top-k filtering (0 = disabled).
        top_p: nucleus sampling threshold (1.0 = disabled).
        cache_length: KV cache length (defaults to prompt + max_new_tokens).

    Returns:
        ``[B, L + max_new_tokens]`` — prompt + generated tokens.
    """
    B, L = tokens.shape
    device = tokens.device
    cfg = model.cfg

    if cache_length is None:
        cache_length = L + max_new_tokens

    # --- Initialise cache ---
    cache = init_cache(cfg, B, cache_length, dtype=torch.bfloat16, device=device)

    # --- Prefill phase ---
    logits, cache = model(
        tokens,
        cache=cache,
        image_patches=image_patches,
        image_mask=image_mask,
        audio_mel=audio_mel,
        audio_mel_mask=audio_mel_mask,
    )

    # Take logits of last position
    next_logits = logits[:, -1, :]
    next_token = _sample_token(next_logits, temperature, top_k, top_p)
    generated = [next_token]

    # --- Decode loop ---
    for _ in range(max_new_tokens - 1):
        inp = next_token.unsqueeze(1)  # [B, 1]
        logits, cache = model(inp, cache=cache)
        next_logits = logits[:, -1, :]
        next_token = _sample_token(next_logits, temperature, top_k, top_p)
        generated.append(next_token)

    return torch.cat([tokens, torch.stack(generated, dim=1)], dim=1)


@torch.no_grad()
def chat(
    model: Gemma4Model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    device: torch.device | str | None = None,
) -> str:
    """Single-turn chat: wrap prompt in Gemma4 chat template, generate, decode.

    Wraps ``prompt`` as::

        <bos><start_of_turn>user
        {prompt}<end_of_turn>
        <start_of_turn>model

    and generates until EOS or ``max_new_tokens``.

    Args:
        model: A ``Gemma4Model`` instance.
        tokenizer: A ``Gemma4Tokenizer`` (or any object with ``.encode()``
            / ``.decode()`` and special token attributes).
        prompt: User message text.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering (0 = disabled).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        device: Device for input tensors (inferred from model if None).

    Returns:
        Decoded model response (without the chat template wrapper).
    """
    if device is None:
        device = next(model.parameters()).device

    # Build chat-formatted token sequence
    ids: list[int] = [tokenizer.BOS, tokenizer.START_OF_TURN]
    ids += tokenizer.encode("user\n" + prompt)
    ids += [tokenizer.END_OF_TURN, tokenizer.START_OF_TURN]
    ids += tokenizer.encode("model\n")

    prompt_len = len(ids)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    # Generate
    output = generate(
        model,
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Decode only the newly generated tokens, strip EOS if present
    gen_ids = output[0, prompt_len:].tolist()
    if tokenizer.EOS in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(tokenizer.EOS)]
    return tokenizer.decode(gen_ids)
