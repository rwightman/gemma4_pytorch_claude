"""Generation, KV-cache, and attention-backend equivalence tests.

These pin the properties that are easy to break silently: a cached decode must
match a full forward pass, the two attention backends must agree, batched
generation must stop per-sequence, and a cache overflow must raise rather than
quietly evict tokens the model still needs.
"""

import pytest
import torch

from gemma4_pt_claude.config import (
    AttentionType,
    Gemma4Config,
    KVCacheSharingConfig,
    TextConfig,
)
from gemma4_pt_claude.generate import generate, init_cache
from gemma4_pt_claude.model import Gemma4Model

VOCAB = 64
PARITY_TOL = 1e-4


def _config(
        attn_impl: str = "sdpa",
        sliding_window_size: int = 16,
        num_layers: int = 4,
        kv_sharing: KVCacheSharingConfig | None = None,
) -> Gemma4Config:
    text = TextConfig(
        vocab_size=VOCAB,
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
        head_dim=16,
        num_kv_heads=2,
        num_layers=num_layers,
        sliding_window_size=sliding_window_size,
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        use_qk_norm=True,
        use_value_norm=True,
        kv_sharing=kv_sharing,
        attn_impl=attn_impl,
    )
    return Gemma4Config(text=text)


def _model(cfg: Gemma4Config, seed: int = 0) -> Gemma4Model:
    torch.manual_seed(seed)
    return Gemma4Model(cfg).eval()


def _decode_one_at_a_time(model, tokens, cache_length=64, prefill_len=None):
    """Feed *tokens* one at a time through a KV cache, returning all logits."""
    cache = init_cache(
        model.cfg, tokens.shape[0], cache_length,
        dtype=torch.float32, prefill_len=prefill_len,
    )
    steps = []
    for i in range(tokens.shape[1]):
        logits, cache = model(tokens[:, i:i + 1], cache=cache)
        steps.append(logits[:, -1])
    return torch.stack(steps, dim=1)


class TestCacheParity:
    """A cached decode must reproduce the uncached forward pass."""

    def test_incremental_decode_matches_full_forward(self):
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (1, 12))
        with torch.no_grad():
            full, _ = model(tokens)
            incremental = _decode_one_at_a_time(model, tokens)
        assert torch.allclose(full, incremental, atol=PARITY_TOL)

    def test_prefill_then_decode_matches_full_forward(self):
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (1, 12))
        with torch.no_grad():
            full, _ = model(tokens)
            cache = init_cache(model.cfg, 1, 32, dtype=torch.float32)
            _, cache = model(tokens[:, :-1], cache=cache)
            last, _ = model(tokens[:, -1:], cache=cache)
        assert torch.allclose(full[:, -1], last[:, -1], atol=PARITY_TOL)

    def test_decode_past_sliding_window(self):
        """Local layers must still be correct once the window has scrolled."""
        model = _model(_config(sliding_window_size=4))
        tokens = torch.randint(0, VOCAB, (1, 20))
        with torch.no_grad():
            full, _ = model(tokens)
            incremental = _decode_one_at_a_time(model, tokens)
        assert torch.allclose(full, incremental, atol=PARITY_TOL)

    def test_kv_sharing_cache_parity(self):
        sharing = KVCacheSharingConfig(frac_shared_layers=0.5, share_global=True, share_local=True)
        model = _model(_config(num_layers=8, kv_sharing=sharing))
        tokens = torch.randint(0, VOCAB, (1, 10))
        with torch.no_grad():
            full, _ = model(tokens)
            incremental = _decode_one_at_a_time(model, tokens)
        assert torch.allclose(full, incremental, atol=PARITY_TOL)

    def test_batched_cache_parity(self):
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (3, 9))
        with torch.no_grad():
            full, _ = model(tokens)
            incremental = _decode_one_at_a_time(model, tokens)
        assert torch.allclose(full, incremental, atol=PARITY_TOL)

    def test_logits_to_keep_matches_full_logits(self):
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (2, 7))
        with torch.no_grad():
            full, _ = model(tokens)
            last_two, _ = model(tokens, logits_to_keep=2)
        assert last_two.shape == (2, 2, VOCAB)
        assert torch.allclose(full[:, -2:], last_two, atol=PARITY_TOL)


class TestRollingWindowCache:
    """Sliding-window layers may use a smaller, rolling cache."""

    def test_window_cache_is_smaller(self):
        cfg = _config(sliding_window_size=4)
        plain = init_cache(cfg, 1, 64, dtype=torch.float32)
        rolled = init_cache(cfg, 1, 64, dtype=torch.float32, prefill_len=2)
        plain_slots = sum(c["k"].shape[1] for c in plain.values())
        rolled_slots = sum(c["k"].shape[1] for c in rolled.values())
        assert rolled_slots < plain_slots
        assert any(c["rolling"] for c in rolled.values())

    def test_rolling_cache_matches_full_forward(self):
        """Wrapping around the buffer must not change the result."""
        model = _model(_config(sliding_window_size=4))
        tokens = torch.randint(0, VOCAB, (1, 24))
        with torch.no_grad():
            full, _ = model(tokens)
            incremental = _decode_one_at_a_time(model, tokens, cache_length=64, prefill_len=1)
        assert torch.allclose(full, incremental, atol=PARITY_TOL)

    def test_generate_with_rolling_cache_matches_plain_cache(self):
        model = _model(_config(sliding_window_size=4))
        prompt = torch.randint(0, VOCAB, (1, 6))
        with torch.no_grad():
            out = generate(model, prompt, max_new_tokens=16, temperature=0.0)
        # Same tokens must come out of a plain (non-rolling) full-length cache.
        with torch.no_grad():
            cache = init_cache(model.cfg, 1, 6 + 16, dtype=torch.float32)
            logits, cache = model(prompt, cache=cache)
            ref = [logits[:, -1].argmax(-1)]
            for _ in range(15):
                logits, cache = model(ref[-1].unsqueeze(1), cache=cache)
                ref.append(logits[:, -1].argmax(-1))
        assert torch.equal(out[0, 6:], torch.stack(ref, dim=1)[0])


class TestCacheOverflow:
    """Overflow must raise, not silently evict."""

    def test_prompt_longer_than_cache_raises(self):
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (1, 12))
        with pytest.raises(ValueError, match="cache_length"):
            generate(model, tokens, max_new_tokens=2, cache_length=8, temperature=0.0)

    def test_decode_past_non_rolling_cache_raises(self):
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (1, 4))
        cache = init_cache(model.cfg, 1, 6, dtype=torch.float32)
        with torch.no_grad():
            _, cache = model(tokens, cache=cache)
            _, cache = model(tokens[:, :1], cache=cache)
            _, cache = model(tokens[:, :1], cache=cache)
            with pytest.raises(ValueError, match="overflow"):
                model(tokens[:, :1], cache=cache)

    def test_write_larger_than_cache_raises(self):
        model = _model(_config())
        cache = init_cache(model.cfg, 1, 4, dtype=torch.float32)
        with pytest.raises(ValueError, match="overflow"), torch.no_grad():
            model(torch.randint(0, VOCAB, (1, 8)), cache=cache)


class TestAttentionBackendParity:
    def test_sdpa_matches_eager(self):
        sdpa = _model(_config("sdpa"), seed=7)
        eager = _model(_config("eager"), seed=7)
        eager.load_state_dict(sdpa.state_dict())
        tokens = torch.randint(0, VOCAB, (2, 10))
        with torch.no_grad():
            a, _ = sdpa(tokens)
            b, _ = eager(tokens)
        assert torch.allclose(a, b, atol=PARITY_TOL)

    def test_sdpa_matches_eager_with_cache(self):
        sdpa = _model(_config("sdpa"), seed=7)
        eager = _model(_config("eager"), seed=7)
        eager.load_state_dict(sdpa.state_dict())
        tokens = torch.randint(0, VOCAB, (1, 8))
        with torch.no_grad():
            a = _decode_one_at_a_time(sdpa, tokens)
            b = _decode_one_at_a_time(eager, tokens)
        assert torch.allclose(a, b, atol=PARITY_TOL)


class TestBatchedGeneration:
    def test_batch_generate_shape(self):
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (3, 5))
        out = generate(model, tokens, max_new_tokens=4, temperature=0.0)
        assert out.shape == (3, 9)
        assert torch.equal(out[:, :5], tokens)

    def test_no_cross_batch_leakage(self):
        """A row's logits must not depend on what its neighbours contain.

        This is the invariant that actually matters, and unlike exact
        token equality it holds regardless of dtype: batching changes GEMM
        tiling, so B=1 and B=2 results differ by rounding even when correct.
        """
        model = _model(_config())
        row = torch.randint(0, VOCAB, (1, 8))
        other_a = torch.randint(0, VOCAB, (1, 8))
        other_b = torch.randint(0, VOCAB, (1, 8))
        with torch.no_grad():
            with_a, _ = model(torch.cat([row, other_a]))
            with_b, _ = model(torch.cat([row, other_b]))
        assert torch.equal(with_a[0], with_b[0])

    def test_no_cross_batch_leakage_through_cache(self):
        model = _model(_config())
        row = torch.randint(0, VOCAB, (1, 6))
        other_a = torch.randint(0, VOCAB, (1, 6))
        other_b = torch.randint(0, VOCAB, (1, 6))
        outs = []
        for other in (other_a, other_b):
            tokens = torch.cat([row, other])
            cache = init_cache(model.cfg, 2, 16, dtype=torch.float32)
            with torch.no_grad():
                _, cache = model(tokens, cache=cache)
                logits, _ = model(tokens[:, :1], cache=cache)
            outs.append(logits[0])
        assert torch.equal(outs[0], outs[1])

    def test_batch_rows_match_individual_generation(self):
        # Exact on CPU float32; batching changes GEMM tiling, so this is a
        # determinism check rather than a promise that it holds in bfloat16.
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (3, 5))
        batched = generate(model, tokens, max_new_tokens=6, temperature=0.0, stop_tokens=set())
        for i in range(3):
            single = generate(
                model, tokens[i : i + 1], max_new_tokens=6, temperature=0.0, stop_tokens=set(),
            )
            assert torch.equal(batched[i], single[0])

    def test_finished_sequences_are_padded(self):
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (2, 4))
        # Stop on whatever the first sampled token happens to be, per row.
        first = generate(model, tokens, max_new_tokens=1, temperature=0.0)[:, -1]
        out = generate(
            model, tokens,
            max_new_tokens=5,
            temperature=0.0,
            stop_tokens={int(first[0].item())},
            pad_token_id=0,
        )
        # Row 0 stops immediately, so everything after the stop token is padding.
        assert torch.equal(out[0, 5:], torch.zeros(4, dtype=out.dtype))

    def test_stop_token_ends_generation(self):
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (1, 4))
        first = generate(model, tokens, max_new_tokens=1, temperature=0.0)[0, -1].item()
        out = generate(model, tokens, max_new_tokens=10, temperature=0.0, stop_tokens={first})
        assert out.shape == (1, 5)


class TestGenerationCallback:
    def test_callback_sees_every_token(self):
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (1, 4))
        seen = []
        out = generate(
            model, tokens, max_new_tokens=5, temperature=0.0,
            stop_tokens=set(), callback=lambda t: seen.append(t.clone()),
        )
        assert len(seen) == 5
        assert torch.equal(torch.stack(seen, dim=1)[0], out[0, 4:])

    def test_callback_can_stop_early(self):
        model = _model(_config())
        tokens = torch.randint(0, VOCAB, (1, 4))
        out = generate(
            model, tokens, max_new_tokens=10, temperature=0.0,
            stop_tokens=set(), callback=lambda t: True,
        )
        assert out.shape == (1, 5)


class TestBidirectionalVisionWithCache:
    """Bidirectional image spans must survive the cached prefill path."""

    def _bidir_model(self):
        text = TextConfig(
            vocab_size=VOCAB,
            embed_dim=32,
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
            num_kv_heads=2,
            num_layers=2,
            sliding_window_size=64,
            attention_pattern=(AttentionType.GLOBAL,),
            use_value_norm=False,
            bidirectional_vision=True,
        )
        torch.manual_seed(3)
        return Gemma4Model(Gemma4Config(text=text)).eval()

    def test_cached_prefill_applies_bidirectional_mask(self):
        model = self._bidir_model()
        tokens = torch.randint(4, VOCAB, (1, 10))
        image_mask = torch.zeros(1, 10, dtype=torch.bool)
        image_mask[0, 3:7] = True

        with torch.no_grad():
            uncached, _ = model(tokens, image_mask=image_mask)
            cache = init_cache(model.cfg, 1, 16, dtype=torch.float32)
            cached, _ = model(tokens, cache=cache, image_mask=image_mask)
            plain, _ = model(tokens)

        # Cached prefill must match the uncached bidirectional result...
        assert torch.allclose(uncached, cached, atol=PARITY_TOL)
        # ...and must actually differ from the purely causal result.
        assert not torch.allclose(uncached, plain, atol=1e-3)
