# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleEngine's optional mlx-lm prompt trie cache."""

import hashlib
import sys
import threading
import types
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


class ThreadRecordingTrie:
    nbytes = 0

    def __init__(self):
        self.fetch_threads = []
        self.insert_threads = []

    def __len__(self):
        return 0

    def fetch_nearest_cache(self, _model, tokens):
        self.fetch_threads.append(threading.get_ident())
        return None, tokens

    def insert_cache(self, *_args, **_kwargs):
        self.insert_threads.append(threading.get_ident())


class NonWinningTrie:
    """Expose a shorter match without materializing its prompt cache."""

    nbytes = 0

    def __init__(self, tokens_saved):
        self.tokens_saved = tokens_saved
        self.fetches = 0
        self._trie = self

    def __len__(self):
        return 1

    def search(self, model, tokens):
        return SimpleNamespace(
            model=model,
            exact=None,
            shorter=tokens[: self.tokens_saved],
            longer=None,
            common_prefix=self.tokens_saved,
        )

    def fetch_nearest_cache(self, _model, tokens):
        self.fetches += 1
        return [object()], tokens[self.tokens_saved :]


class ExactUntrimmableTrie:
    """Expose an exact match whose cache cannot drop the final token."""

    nbytes = 0

    def __init__(self):
        self.fetches = 0
        self._trie = self
        self.prompt_cache = [SimpleNamespace(is_trimmable=lambda: False)]

    def __len__(self):
        return 1

    def search(self, model, tokens):
        return SimpleNamespace(
            model=model,
            exact=tokens,
            shorter=None,
            longer=None,
            common_prefix=0,
        )

    def get(self, _model, _tokens):
        return SimpleNamespace(prompt_cache=self.prompt_cache)

    def fetch_nearest_cache(self, _model, _tokens):
        self.fetches += 1
        return self.prompt_cache, []


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


async def test_prefix_trie_cache_reuses_prefix_without_system_message():
    engine = _engine(prefix_trie_cache=True, prefix_trie_cache_size=8)
    fake_stream_generate = _responses([ord("X")])

    with (
        patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
        patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
    ):
        await _collect(engine, [{"role": "user", "content": "first"}])
        await _collect(
            engine,
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "X"},
                {"role": "user", "content": "second"},
            ],
        )

    stats = engine.get_stats()["prefix_trie_cache"]
    assert stats["hits"] == 1
    assert stats["inserts"] == 2
    assert len(fake_stream_generate.seen_prompts[1]) < len(
        FakeTokenizer().encode(
            FakeTokenizer().apply_chat_template(
                [
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
    system_hash = hashlib.sha256(prefix.encode()).hexdigest()[:16]
    system_token_count = len(tokenizer.encode(prefix, add_special_tokens=True))
    engine._system_kv_cache[system_hash] = (
        [FakeCache().state],
        system_token_count,
    )
    engine._prefix_trie_cache = NoFetchTrie()

    with (
        patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
        patch("mlx_lm.stream_generate", side_effect=_responses([ord("Y")])),
    ):
        await _collect(engine, messages)

    assert engine.get_stats()["prefix_trie_cache"]["lookups"] == 0


async def test_system_snapshot_rejects_shorter_trie_before_materializing_cache():
    engine = _engine(prefix_trie_cache=True)
    trie = NonWinningTrie(tokens_saved=4)
    engine._prefix_trie_cache = trie

    cache, rest, tokens_saved = engine._fetch_prefix_trie_cache(
        engine._model.model,
        list(range(8)),
        minimum_tokens_saved=4,
    )

    assert (cache, rest, tokens_saved) == (None, None, 0)
    assert trie.fetches == 0


async def test_system_snapshot_rejects_untrimmable_exact_trie_before_fetch(
    monkeypatch,
):
    cache_module = types.ModuleType("mlx_lm.models.cache")
    cache_module.can_trim_prompt_cache = lambda cache: all(
        entry.is_trimmable() for entry in cache
    )
    cache_module.trim_prompt_cache = lambda cache, count: count
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_module)

    engine = _engine(prefix_trie_cache=True)
    trie = ExactUntrimmableTrie()
    engine._prefix_trie_cache = trie

    cache, rest, tokens_saved = engine._fetch_prefix_trie_cache(
        engine._model.model,
        list(range(8)),
        minimum_tokens_saved=4,
    )

    assert (cache, rest, tokens_saved) == (None, None, 0)
    assert trie.fetches == 0


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


async def test_prefix_trie_cache_stays_on_generation_owner_thread():
    engine = _engine(prefix_trie_cache=True)
    trie = ThreadRecordingTrie()
    engine._prefix_trie_cache = trie
    event_loop_thread = threading.get_ident()

    with (
        patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
        patch("mlx_lm.stream_generate", side_effect=_responses([ord("X")])),
    ):
        await _collect(
            engine,
            [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "first"},
            ],
        )

    assert trie.fetch_threads
    assert trie.insert_threads
    owner_threads = set(trie.fetch_threads + trie.insert_threads)
    assert len(owner_threads) == 1
    assert event_loop_thread not in owner_threads


async def test_prefix_trie_cache_is_cleared_on_stop():
    engine = _engine(prefix_trie_cache=True)
    engine._prefix_trie_cache = ThreadRecordingTrie()
    engine._prefix_trie_cache_stats["hits"] = 2

    await engine.stop()

    stats = engine.get_stats()["prefix_trie_cache"]
    assert stats["entries"] == 0
    assert stats["hits"] == 0
