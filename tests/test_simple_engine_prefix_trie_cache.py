# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleEngine's optional mlx-lm prompt trie cache."""

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from vllm_mlx.engine.simple import SimpleEngine

pytestmark = pytest.mark.anyio


class FakeTokenizer:
    bos_token = None
    eos_token_ids = []

    def apply_chat_template(self, messages, **_kwargs):
        rendered = ""
        for message in messages:
            rendered += f"<|{message['role']}|>{message.get('content', '')}\n"
        return rendered + "<|assistant|>"

    def encode(self, text, add_special_tokens=True):
        tokens = [ord(ch) for ch in text]
        return ([1] if add_special_tokens else []) + tokens


class FakeCache:
    def __init__(self):
        self.state = (
            mx.array([[1]], dtype=mx.float32),
            mx.array([[2]], dtype=mx.float32),
        )
        self.nbytes = 8

    def is_trimmable(self):
        return False


class FakeModel:
    def __call__(self, *_args, **_kwargs):
        return mx.zeros((1, 1, 4), dtype=mx.float32)


class NoFetchTrie:
    nbytes = 0

    def __len__(self):
        return 0

    def fetch_nearest_cache(self, *_args, **_kwargs):
        pytest.fail("prefix trie lookup should not run on exact snapshot hit")

    def insert_cache(self, *_args, **_kwargs):
        return None


def _engine(**kwargs):
    engine = SimpleEngine("test-model", **kwargs)
    engine._loaded = True
    engine._supports_system_kv_cache = True
    engine._model = SimpleNamespace(model=FakeModel(), tokenizer=FakeTokenizer())
    return engine


def _responses(tokens):
    def fake_stream_generate(*_args, **kwargs):
        seen_prompts = fake_stream_generate.seen_prompts
        seen_prompts.append(kwargs["prompt"].tolist())
        for token in tokens:
            yield SimpleNamespace(text=chr(token), token=token, finish_reason="stop")

    fake_stream_generate.seen_prompts = []
    return fake_stream_generate


async def _collect(engine, messages):
    return [
        chunk
        async for chunk in engine.stream_chat(
            messages,
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
        )
    ]


async def test_prefix_trie_cache_reuses_growing_conversation_prefix():
    engine = _engine(prefix_trie_cache=True, prefix_trie_cache_size=8)
    fake_stream_generate = _responses([ord("X")])

    with (
        patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
        patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
    ):
        await _collect(
            engine,
            [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "first"},
            ],
        )
        await _collect(
            engine,
            [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "X"},
                {"role": "user", "content": "second"},
            ],
        )

    stats = engine.get_stats()["prefix_trie_cache"]
    assert stats["hits"] == 1
    assert stats["tokens_saved"] > 0
    assert stats["inserts"] == 2
    assert len(fake_stream_generate.seen_prompts[1]) < len(
        FakeTokenizer().encode(
            FakeTokenizer().apply_chat_template(
                [
                    {"role": "system", "content": "Rules"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "X"},
                    {"role": "user", "content": "second"},
                ]
            )
        )
    )


async def test_existing_exact_snapshot_hit_wins_before_prefix_trie_lookup():
    tokenizer = FakeTokenizer()
    engine = _engine(prefix_trie_cache=True)
    messages = [
        {"role": "system", "content": "Rules"},
        {"role": "user", "content": "first"},
    ]
    rendered_a = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Alpha"},
        ]
    )
    rendered_b = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Bravo"},
        ]
    )
    boundary = next(i for i, (a, b) in enumerate(zip(rendered_a, rendered_b)) if a != b)
    prefix = rendered_a[:boundary]
    engine._system_kv_hash = hashlib.sha256(prefix.encode()).hexdigest()[:16]
    engine._system_kv_token_count = len(
        tokenizer.encode(prefix, add_special_tokens=True)
    )
    engine._system_kv_snapshot = [FakeCache().state]
    engine._prefix_trie_cache = NoFetchTrie()

    with (
        patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
        patch("mlx_lm.stream_generate", side_effect=_responses([ord("Y")])),
    ):
        await _collect(engine, messages)

    assert engine.get_stats()["prefix_trie_cache"]["lookups"] == 0


async def test_prefix_trie_cache_is_disabled_by_default():
    engine = _engine()

    with (
        patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
        patch("mlx_lm.stream_generate", side_effect=_responses([ord("Z")])),
    ):
        await _collect(
            engine,
            [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "first"},
            ],
        )

    assert "prefix_trie_cache" not in engine.get_stats()


async def test_prefix_trie_cache_honors_entry_bound():
    engine = _engine(prefix_trie_cache=True, prefix_trie_cache_size=1)

    with (
        patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
        patch("mlx_lm.stream_generate", side_effect=_responses([ord("A"), ord("B")])),
    ):
        await _collect(
            engine,
            [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "one"},
            ],
        )
        await _collect(
            engine,
            [
                {"role": "system", "content": "Other rules"},
                {"role": "user", "content": "two"},
            ],
        )

    assert engine.get_stats()["prefix_trie_cache"]["entries"] == 1
