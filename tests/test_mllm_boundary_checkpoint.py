# SPDX-License-Identifier: Apache-2.0
"""Conversation-boundary checkpoints for hybrid (non-rewindable) prefix caches.

Hybrid models (Qwen3.5 family) keep recurrent state that cannot be rewound, so
a stored prompt state is reusable only by prompts it is an exact token prefix
of. A thinking-enabled generation prompt ends with the template's think suffix
(``<think>\\n``), which the next turn renders differently once that turn is
history. These tests pin that prefill checkpoints at the boundary before the
suffix, that the checkpoint replaces (not adds to) the full-prompt entry, that
a growing conversation keeps one entry, and that prompts without the suffix
keep the previous full-prompt behaviour.

Runs without MLX: ``tests._mlx_stub`` is installed when mlx is unavailable,
and the fakes below use numpy state so prefix-cache byte accounting is exact.
Keep this file in its own CI invocation (see _mlx_stub.py's docstring).
"""

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests import _mlx_stub

_mlx_stub.install_if_unavailable()

import vllm_mlx.mllm_batch_generator as module  # noqa: E402
from vllm_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
)
from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator  # noqa: E402

SUFFIX = (98, 99)  # stands in for the tokenized "<think>\n"
OFF_SUFFIX = (98, 77, 97, 77)  # "<think>\n\n</think>\n\n" (thinking disabled)
BYTES_PER_TOKEN = 4  # float32 state element per token, see _RecurrentLeaf


class _RecurrentLeaf:
    """Non-trimmable leaf whose state grows by one float32 per token."""

    def __init__(self, position=0):
        self.position = position

    @property
    def state(self):
        return [np.zeros(self.position, dtype=np.float32)]

    @state.setter
    def state(self, value):
        self.position = int(value[0].shape[0])

    @property
    def meta_state(self):
        return []

    @classmethod
    def from_state(cls, state, _meta_state):
        return cls(int(state[0].shape[0]))

    def is_trimmable(self):
        return False


@pytest.fixture
def generator(monkeypatch):
    if isinstance(sys.modules["mlx.core"], MagicMock):
        # Under the stub, memory_cache's ``isinstance(x, mx.array)`` would
        # test against a MagicMock and raise. A real class makes numpy state
        # pass through detach unchanged, as with no MLX installed at all.
        monkeypatch.setattr(
            sys.modules["mlx.core"], "array", type("_NoMLXArray", (), {})
        )
    monkeypatch.setattr(
        MLLMBatchGenerator, "_cache_class_families", staticmethod(lambda: ((), (), ()))
    )
    monkeypatch.setattr(module, "_eval_prompt_cache", lambda _cache: None)
    monkeypatch.setattr(module.mx, "eval", lambda *_args: None)

    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.prefix_cache = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(max_memory_mb=1, min_prefix_tokens=1),
    )
    gen.prefill_step_size = 64
    gen._think_suffix_len = len(SUFFIX)
    gen._think_suffixes = (SUFFIX, OFF_SUFFIX)
    gen._aborted_request_ids = set()
    gen._prefill_progress = {}
    gen._prefix_checkpoint_lock = threading.Lock()
    gen._request_prefix_checkpoints = {}
    gen.calls = []

    def language_model(tokens, cache):
        gen.calls.append(np.asarray(tokens).reshape(-1).tolist())
        cache[0].position += tokens.shape[1]
        return np.zeros((1, tokens.shape[1], 4), dtype=np.float32)

    gen.language_model = language_model
    return gen


def _request(tokens, request_id="req"):
    return SimpleNamespace(
        request_id=request_id,
        input_ids=np.array([tokens]),
        vision_encoded=False,
        is_text_only=True,
        images=None,
        videos=None,
        audio=None,
        pixel_values=None,
        attention_mask=None,
        image_grid_thw=None,
        extra_kwargs={},
    )


def _prefill(gen, tokens, request_id="req"):
    """Run prefill the way _process_prompts does: restore any prefix hit,
    resume prefill there, then publish the post-prefill prompt state."""
    lookup = list(tokens[: -len(SUFFIX)])
    stored, remaining = gen.prefix_cache.fetch(lookup)
    if stored is not None:
        cache = gen._clone_prefix_for_replay(stored)
        start = len(lookup) - len(remaining)
    else:
        cache = [_RecurrentLeaf()]
        start = 0
    req = _request(tokens, request_id)
    logits = gen._run_chunked_text_prefill(req, cache, start=start)
    gen._publish_prompt_state(req, cache, logits[:, -1, :], extends_cached=start > 0)
    return start, cache


def _keys(gen):
    return sorted(gen.prefix_cache._entries)


def test_boundary_checkpoint_replaces_full_prompt_entry(generator):
    turn1 = [10, 11, 12, 13, *SUFFIX]

    start, cache = _prefill(generator, turn1)

    assert start == 0
    # Prefill splits exactly at the boundary, then runs the suffix.
    assert generator.calls == [[10, 11, 12, 13], [*SUFFIX]]
    assert cache[0].position == len(turn1)
    # One entry, keyed at the boundary, with no full-prompt duplicate.
    assert _keys(generator) == [(10, 11, 12, 13)]
    assert generator.prefix_cache.fetch_exact_auxiliary(turn1) is None


def test_next_turn_hits_boundary_across_think_suffix(generator):
    turn1 = [10, 11, 12, 13, *SUFFIX]
    _prefill(generator, turn1)
    generator.calls.clear()

    # History re-renders turn 1 without the generation suffix (50, 51 stand
    # for its reasoning and answer), then the new user turn and the suffix.
    turn2 = [10, 11, 12, 13, 50, 51, 20, 21, *SUFFIX]
    start, cache = _prefill(generator, turn2)

    assert start == 4
    assert generator.calls == [[50, 51, 20, 21], [*SUFFIX]]
    assert cache[0].position == len(turn2)
    # Turn 2's boundary supersedes turn 1's, which it was restored from.
    assert _keys(generator) == [(10, 11, 12, 13, 50, 51, 20, 21)]

    generator.calls.clear()
    turn3 = [*turn2[:-2], 52, 53, 30, *SUFFIX]
    start, _ = _prefill(generator, turn3)
    assert start == len(turn2) - len(SUFFIX)
    assert generator.calls == [[52, 53, 30], [*SUFFIX]]


def test_exact_repeat_replays_only_the_suffix(generator):
    turn1 = [10, 11, 12, 13, *SUFFIX]
    _prefill(generator, turn1)
    generator.calls.clear()

    start, cache = _prefill(generator, turn1, request_id="repeat")

    assert start == 4
    assert generator.calls == [[*SUFFIX]]
    assert cache[0].position == len(turn1)
    assert _keys(generator) == [(10, 11, 12, 13)]


def test_next_turn_hits_boundary_across_disabled_think_block(generator):
    # Thinking disabled on a template that drops the empty think block from
    # history (Qwen3.5): the boundary sits before the 4-token disabled block.
    turn1 = [10, 11, 12, 13, *OFF_SUFFIX]
    _prefill(generator, turn1)
    assert _keys(generator) == [(10, 11, 12, 13)]
    generator.calls.clear()

    turn2 = [10, 11, 12, 13, 50, 20, 21, *OFF_SUFFIX]
    start, _ = _prefill(generator, turn2)

    assert start == 4
    assert generator.calls == [[50, 20, 21], [*OFF_SUFFIX]]
    assert _keys(generator) == [(10, 11, 12, 13, 50, 20, 21)]


def test_no_checkpoint_without_think_suffix(generator):
    # No think block at the end (e.g. a template without thinking support).
    prompt = [10, 11, 12, 13, 14]

    _prefill(generator, prompt)

    assert generator.calls == [[10, 11, 12, 13, 14]]
    assert _keys(generator) == [tuple(prompt)]
    assert generator.prefix_cache.fetch_exact_auxiliary(prompt) is not None


def test_no_checkpoint_when_model_has_no_think_suffix(generator):
    generator._think_suffix_len = 0
    generator._think_suffixes = ()
    prompt = [10, 11, 12, 13, *SUFFIX]
    req = _request(prompt)

    assert generator._prefill_checkpoint_plan(req.input_ids, [_RecurrentLeaf()]) == (
        None,
        None,
    )


def test_rewindable_cache_never_checkpoints(generator):
    class _KVLeaf(_RecurrentLeaf):
        def is_trimmable(self):
            return True

    prompt = [10, 11, 12, 13, *SUFFIX]
    req = _request(prompt)

    assert generator._prefill_checkpoint_plan(req.input_ids, [_KVLeaf()]) == (
        None,
        None,
    )


def test_memory_accounting_one_entry_per_conversation(generator):
    turns = [
        [10, 11, 12, 13, *SUFFIX],
        [10, 11, 12, 13, 50, 51, 20, 21, *SUFFIX],
        [10, 11, 12, 13, 50, 51, 20, 21, 52, 53, 30, *SUFFIX],
    ]
    for i, tokens in enumerate(turns):
        _prefill(generator, tokens, request_id=f"turn{i}")
        boundary = len(tokens) - len(SUFFIX)
        stats = generator.prefix_cache.get_stats()
        assert stats["entry_count"] == 1
        # Exactly the boundary state: no logits, no superseded parents.
        assert generator.prefix_cache._current_memory == boundary * BYTES_PER_TOKEN


def test_memory_without_suffix_matches_previous_full_prompt_entry(generator):
    prompt = [10, 11, 12, 13, 14]

    _prefill(generator, prompt)

    entry = generator.prefix_cache._entries[tuple(prompt)]
    logits_bytes = 4 * 4  # (1, 4) float32 final logits auxiliary
    assert entry.memory_bytes == len(prompt) * BYTES_PER_TOKEN + logits_bytes
    assert generator.prefix_cache._current_memory == entry.memory_bytes


@pytest.mark.parametrize(
    "tokens, expected",
    [
        ([10, 11, *SUFFIX], 2),
        ([10, 11, 12], None),
        ([*SUFFIX], None),
        ([10, 99, 98], None),
        ([], None),
    ],
)
def test_conversation_boundary(generator, tokens, expected):
    assert generator._conversation_boundary(tokens) == expected


def test_think_suffixes_recorded_for_both_modes_without_bos():
    class Tokenizer:
        def apply_chat_template(self, _messages, **kwargs):
            if kwargs["enable_thinking"]:
                return "<|assistant|>\n<think>\n"
            return "<|assistant|>\n<think>\n\n</think>\n\n"

        def encode(self, text):
            ids = [1]  # BOS on every encode
            ids += {
                "<think>\n": list(SUFFIX),
                "<think>\n\n</think>\n\n": list(OFF_SUFFIX),
            }.get(text, [])
            return ids

    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.processor = SimpleNamespace(tokenizer=Tokenizer())

    assert gen._compute_think_suffix_len() == 2
    assert gen._think_suffixes == (SUFFIX, OFF_SUFFIX)


def test_no_disabled_block_when_template_appends_none():
    class Tokenizer:
        def apply_chat_template(self, _messages, **kwargs):
            if kwargs["enable_thinking"]:
                return "<|assistant|>\n<think>\n"
            return "<|assistant|>\n"

        def encode(self, text):
            return list(SUFFIX) if text == "<think>\n" else []

    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.processor = SimpleNamespace(tokenizer=Tokenizer())

    assert gen._compute_think_suffix_len() == 2
    assert gen._think_suffixes == (SUFFIX,)


def test_think_suffixes_empty_without_think_template():
    class Tokenizer:
        def apply_chat_template(self, _messages, **_kwargs):
            return "<|assistant|>\n"

        def encode(self, _text):
            return []

    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.processor = SimpleNamespace(tokenizer=Tokenizer())

    assert gen._compute_think_suffix_len() == 0
    assert gen._think_suffixes == ()


def test_disabled_block_found_when_thinking_changes_the_preamble():
    # Some templates (Qwen3.8) add a thinking preamble to the system prompt,
    # so only the assistant header is shared between the two prompts.
    class Tokenizer:
        def apply_chat_template(self, _messages, **kwargs):
            if kwargs["enable_thinking"]:
                return "<|system|>\nThink first.\n<|assistant|>\n<think>\n"
            return "<|assistant|>\n<think>\n\n</think>\n\n"

        def encode(self, text):
            return {
                "<think>\n": list(SUFFIX),
                "<think>\n\n</think>\n\n": list(OFF_SUFFIX),
            }.get(text, [])

    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.processor = SimpleNamespace(tokenizer=Tokenizer())

    assert gen._compute_think_suffix_len() == 2
    assert gen._think_suffixes == (SUFFIX, OFF_SUFFIX)
