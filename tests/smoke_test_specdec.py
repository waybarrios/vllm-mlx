#!/usr/bin/env python3
"""
Smoke test for speculative decoding with real models.

Usage: python tests/smoke_test_specdec.py

Uses Qwen3.5-35B-A3B-8bit as target, Qwen3.5-4B-4bit as draft.
Tests the SimpleEngine path (mlx_lm.stream_generate with draft_model).
"""

import os
import sys
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = os.path.expanduser("~/ai-models/mlx_models/Qwen3.5-35B-A3B-8bit")
DRAFT = os.path.expanduser("~/ai-models/mlx_models/Qwen3.5-4B-4bit")
PROMPT = "What is the capital of France? Answer in one sentence."
MAX_TOKENS = 64
NUM_DRAFT = 3


def test_without_draft():
    """Baseline: generate without speculative decoding."""
    from mlx_lm import load, stream_generate

    print("=" * 60)
    print("Loading target model (no draft)...")
    model, tokenizer = load(TARGET)
    print(f"Target loaded. Generating {MAX_TOKENS} tokens...")

    tokens = []
    t0 = time.perf_counter()
    for resp in stream_generate(model, tokenizer, prompt=PROMPT, max_tokens=MAX_TOKENS):
        tokens.append(resp.token)
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(tokens)
    print(f"Output ({len(tokens)} tokens, {len(tokens)/elapsed:.1f} tok/s):")
    print(f"  {text}")
    print()
    return len(tokens), elapsed


def test_with_draft():
    """Speculative: generate with draft model."""
    from mlx_lm import load, stream_generate

    print("=" * 60)
    print("Loading target + draft model...")
    model, tokenizer = load(TARGET)
    draft_model, _ = load(DRAFT)

    # Verify vocab match — walk model structure to find embed_tokens
    def _get_vocab_size(m):
        for attr in ["model", "language_model"]:
            sub = getattr(m, attr, None)
            if sub is not None:
                et = getattr(sub, "embed_tokens", None)
                if et is not None:
                    return et.weight.shape[0]
        return None

    target_vocab = _get_vocab_size(model)
    draft_vocab = _get_vocab_size(draft_model)
    print(f"Target vocab: {target_vocab}, Draft vocab: {draft_vocab}")
    if target_vocab and draft_vocab:
        assert target_vocab == draft_vocab, "Vocab size mismatch!"

    print(f"Generating {MAX_TOKENS} tokens with num_draft_tokens={NUM_DRAFT}...")

    tokens = []
    from_draft_count = 0
    t0 = time.perf_counter()
    for resp in stream_generate(
        model,
        tokenizer,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
        draft_model=draft_model,
        num_draft_tokens=NUM_DRAFT,
    ):
        tokens.append(resp.token)
        if resp.from_draft:
            from_draft_count += 1
    elapsed = time.perf_counter() - t0

    text = tokenizer.decode(tokens)
    accept_rate = from_draft_count / len(tokens) * 100 if tokens else 0
    print(f"Output ({len(tokens)} tokens, {len(tokens)/elapsed:.1f} tok/s):")
    print(f"  {text}")
    print(f"Draft acceptance: {from_draft_count}/{len(tokens)} ({accept_rate:.0f}%)")
    print()
    return len(tokens), elapsed


if __name__ == "__main__":
    print("Speculative Decoding Smoke Test")
    print("Target:", TARGET)
    print("Draft:", DRAFT)
    print()

    n1, t1 = test_without_draft()
    # Clear model from memory
    import gc
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()

    n2, t2 = test_with_draft()

    print("=" * 60)
    print("RESULTS:")
    print(f"  Without draft: {n1} tokens in {t1:.2f}s ({n1/t1:.1f} tok/s)")
    print(f"  With draft:    {n2} tokens in {t2:.2f}s ({n2/t2:.1f} tok/s)")
    if t1 > 0 and t2 > 0:
        speedup = (n1 / t1) / (n2 / t2) if n2 / t2 > 0 else 0
        print(f"  Speedup: {1/speedup:.2f}x" if speedup > 0 else "  N/A")
