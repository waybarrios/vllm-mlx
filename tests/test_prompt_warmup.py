# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_mlx.prompt_warmup.

Covers:
- load_warmup_file: validation of shape, content, error paths
- warm_prefix_cache: happy path + error handling with a stub engine
- Agent/code-assistant scenarios that match real code-agent workloads
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from vllm_mlx.prompt_warmup import (
    _build_strict_prefix_string,
    _ensure_user_terminator,
    load_warmup_file,
    warm_prefix_cache,
)

# ---------------------------------------------------------------------------
# Realistic agent system prompts (code-agent-style, ~1-2k tokens each)
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT_CODING_BASE = """You are a code assistant running inside a developer's terminal.

Rules of engagement:
- Read files before editing. Never guess at code behaviour.
- Prefer small, focused edits over sweeping refactors.
- Match the project's existing conventions.
- Add tests that would catch the regression you are fixing.
- Do not add comments that restate what the code obviously does.
- Do not introduce new dependencies without flagging.

Safety:
- Never generate secrets or credentials.
- Refuse destructive actions that affect shared state without confirmation.
- Sanitize input before interpolating into shell, SQL, HTML, regex.

Tools: Read, Write, Edit, Bash, Grep, Glob, Agent.
Keep final responses concise. Use file_path:line_number for code references."""

AGENT_SYSTEM_PROMPT_CODING = AGENT_SYSTEM_PROMPT_CODING_BASE * 3  # ~1.5k tokens


AGENT_SYSTEM_PROMPT_REVIEWER_BASE = """You are a senior code reviewer.

Evaluate diffs for:
- Correctness: off-by-one, race conditions, null handling, error paths.
- Security: input validation, authn/authz, injection, secrets leakage.
- Performance: N+1 queries, unbounded loops, memory leaks.
- Maintainability: naming, layering, testability, complexity.
- Conventions: does this match the rest of the codebase?

For each finding, cite file_path:line_number and explain WHY it matters.
Prioritize by severity: blocker, major, minor, nit.
Do not nitpick formatting that a linter would catch."""

AGENT_SYSTEM_PROMPT_REVIEWER = AGENT_SYSTEM_PROMPT_REVIEWER_BASE * 4  # ~2k tokens


AGENT_CODING_MESSAGES_1 = [{"role": "system", "content": AGENT_SYSTEM_PROMPT_CODING}]

AGENT_REVIEWER_MESSAGES = [{"role": "system", "content": AGENT_SYSTEM_PROMPT_REVIEWER}]

AGENT_CONVERSATION_HISTORY = [
    {"role": "system", "content": AGENT_SYSTEM_PROMPT_CODING},
    {
        "role": "user",
        "content": (
            "I'm getting a TypeError in my LRU cache:\n\n"
            "```python\n"
            "class LRUCache:\n"
            "    def __init__(self, max_size):\n"
            "        self.max_size = max_size\n"
            "        self.data = {}\n"
            "\n"
            "    def get(self, key):\n"
            "        with self._lock:\n"
            "            return self.data.get(key)\n"
            "```\n\n"
            "Error: `AttributeError: 'LRUCache' object has no attribute '_lock'`. "
            "What's wrong?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "`_lock` is referenced in `get()` but never created in `__init__`.\n\n"
            "Fix: add `self._lock = threading.Lock()` in `__init__` and `import threading` "
            "at the top. Also apply the lock on any other mutating methods "
            "(`put`, `evict`) so the cache is thread-safe.\n\n"
            "Want me to patch the file?"
        ),
    },
]


# ---------------------------------------------------------------------------
# Stub engine (mimics BaseEngine.stream_chat interface)
# ---------------------------------------------------------------------------


class _FakeOutput:
    def __init__(self, text: str, finished: bool, prompt_tokens: int):
        self.text = text
        self.new_text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = 1
        self.finished = finished
        self.finish_reason = "stop" if finished else None


class _StubEngine:
    """Mimics the minimal interface warm_prefix_cache expects."""

    def __init__(
        self,
        *,
        raise_on: int | None = None,
        prompt_tokens_per_call: int = 100,
    ) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.raise_on = raise_on
        self.prompt_tokens_per_call = prompt_tokens_per_call

    async def stream_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **_: Any,
    ):
        self.calls.append(messages)
        if self.raise_on is not None and len(self.calls) - 1 == self.raise_on:
            raise RuntimeError("simulated engine failure")
        yield _FakeOutput(
            "ok", finished=True, prompt_tokens=self.prompt_tokens_per_call
        )


# ---------------------------------------------------------------------------
# load_warmup_file
# ---------------------------------------------------------------------------


def test_load_warmup_file_missing(tmp_path: Path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        load_warmup_file(str(missing))


def test_load_warmup_file_not_a_list(tmp_path: Path):
    p = tmp_path / "w.json"
    p.write_text('{"role": "system", "content": "x"}')
    with pytest.raises(ValueError, match="top-level JSON list"):
        load_warmup_file(str(p))


def test_load_warmup_file_empty_list(tmp_path: Path):
    p = tmp_path / "w.json"
    p.write_text("[]")
    with pytest.raises(ValueError, match="empty"):
        load_warmup_file(str(p))


def test_load_warmup_file_entry_not_list(tmp_path: Path):
    p = tmp_path / "w.json"
    p.write_text('[{"role": "system", "content": "x"}]')
    with pytest.raises(ValueError, match="non-empty list of message dicts"):
        load_warmup_file(str(p))


def test_load_warmup_file_entry_empty_list(tmp_path: Path):
    p = tmp_path / "w.json"
    p.write_text("[[]]")
    with pytest.raises(ValueError, match="non-empty list"):
        load_warmup_file(str(p))


def test_load_warmup_file_message_missing_keys(tmp_path: Path):
    p = tmp_path / "w.json"
    p.write_text('[[{"role": "system"}]]')
    with pytest.raises(ValueError, match="missing 'role' or 'content'"):
        load_warmup_file(str(p))


def test_load_warmup_file_valid_single(tmp_path: Path):
    p = tmp_path / "w.json"
    data = [AGENT_CODING_MESSAGES_1]
    p.write_text(json.dumps(data))
    loaded = load_warmup_file(str(p))
    assert loaded == data


def test_load_warmup_file_valid_multi_agent(tmp_path: Path):
    """Real code-agent scenarios: coding agent, reviewer, and conversation history."""
    p = tmp_path / "w.json"
    data = [
        AGENT_CODING_MESSAGES_1,
        AGENT_REVIEWER_MESSAGES,
        AGENT_CONVERSATION_HISTORY,
    ]
    p.write_text(json.dumps(data))
    loaded = load_warmup_file(str(p))
    assert len(loaded) == 3
    assert loaded[0][0]["role"] == "system"
    assert loaded[2][-1]["role"] == "assistant"


def test_load_warmup_file_expands_user_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """~/ in the path should resolve to $HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    p = tmp_path / "w.json"
    p.write_text(json.dumps([AGENT_CODING_MESSAGES_1]))
    # Use a relative-to-$HOME path
    loaded = load_warmup_file("~/w.json")
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# warm_prefix_cache — agent workload scenarios
# ---------------------------------------------------------------------------


def test_warm_prefix_cache_single_coding_agent():
    """Warm up a single code-agent-style system prompt."""
    engine = _StubEngine(prompt_tokens_per_call=400)
    result = asyncio.run(warm_prefix_cache(engine, [AGENT_CODING_MESSAGES_1]))
    assert result["count"] == 1
    assert result["skipped"] == 0
    assert result["total_prompt_tokens"] == 400
    # System-only prompt gets a user-terminator appended (required by some templates).
    assert engine.calls[0][0] == AGENT_CODING_MESSAGES_1[0]
    assert engine.calls[0][-1]["role"] == "user"


def test_warm_prefix_cache_multi_agent_personas():
    """Realistic multi-agent deployment: coding + reviewer + conversation history."""
    engine = _StubEngine(prompt_tokens_per_call=500)
    prompts = [
        AGENT_CODING_MESSAGES_1,
        AGENT_REVIEWER_MESSAGES,
        AGENT_CONVERSATION_HISTORY,
    ]
    result = asyncio.run(warm_prefix_cache(engine, prompts))
    assert result["count"] == 3
    assert result["skipped"] == 0
    assert result["total_prompt_tokens"] == 1500
    # Two system-only prompts get user-terminators; the conversation history
    # (which already ends with assistant) passes through untouched.
    # Calls arrive in gather order — index by system-content match.
    for prompt in prompts:
        match = [c for c in engine.calls if c and c[0] == prompt[0]]
        assert match, f"no call matched system content of {prompt[0]}"
    # History prompt ends in assistant → untouched
    hist_match = [c for c in engine.calls if len(c) == len(AGENT_CONVERSATION_HISTORY)]
    assert hist_match and hist_match[0] == AGENT_CONVERSATION_HISTORY


def test_warm_prefix_cache_handles_individual_failure():
    """One prompt failing must not abort the rest of the warm-up."""
    engine = _StubEngine(raise_on=1, prompt_tokens_per_call=200)
    prompts = [
        AGENT_CODING_MESSAGES_1,
        AGENT_REVIEWER_MESSAGES,
        AGENT_CONVERSATION_HISTORY,
    ]
    result = asyncio.run(warm_prefix_cache(engine, prompts))
    assert result["count"] == 2
    assert result["skipped"] == 1
    # total_prompt_tokens reflects only successful calls
    assert result["total_prompt_tokens"] == 400


def test_warm_prefix_cache_empty_list():
    """Empty prompt list returns zeros and does not touch the engine."""
    engine = _StubEngine()
    result = asyncio.run(warm_prefix_cache(engine, []))
    assert result["count"] == 0
    assert result["skipped"] == 0
    assert result["total_prompt_tokens"] == 0
    assert engine.calls == []


def test_warm_prefix_cache_reports_elapsed_time():
    """elapsed_ms should be >= 0 and not negative."""
    engine = _StubEngine(prompt_tokens_per_call=100)
    result = asyncio.run(warm_prefix_cache(engine, [AGENT_CODING_MESSAGES_1]))
    assert result["elapsed_ms"] >= 0


def test_warm_prefix_cache_uses_max_tokens_1_by_default():
    """The warmer must not waste GPU time generating output — max_tokens defaults to 1."""
    captured_max_tokens: list[int] = []

    class _Recording(_StubEngine):
        async def stream_chat(self, *, messages, max_tokens, temperature, **_):
            captured_max_tokens.append(max_tokens)
            self.calls.append(messages)
            yield _FakeOutput("x", finished=True, prompt_tokens=100)

    engine = _Recording()
    asyncio.run(warm_prefix_cache(engine, [AGENT_CODING_MESSAGES_1]))
    assert captured_max_tokens == [1]


def test_warm_prefix_cache_all_fail_does_not_raise():
    """If every prompt errors, the warmer reports skipped=N but does not crash."""
    engine = _StubEngine(raise_on=0, prompt_tokens_per_call=100)
    # single prompt that errors
    result = asyncio.run(warm_prefix_cache(engine, [AGENT_CODING_MESSAGES_1]))
    assert result["count"] == 0
    assert result["skipped"] == 1
    assert result["total_prompt_tokens"] == 0


# ---------------------------------------------------------------------------
# load → warm end-to-end (file → stub engine)
# ---------------------------------------------------------------------------


def test_end_to_end_file_then_warm(tmp_path: Path):
    """Loading a warm-up file and running it through the warmer produces the
    exact messages the engine would receive from a real HTTP request."""
    p = tmp_path / "agents.json"
    prompts = [
        AGENT_CODING_MESSAGES_1,
        AGENT_REVIEWER_MESSAGES,
    ]
    p.write_text(json.dumps(prompts))

    loaded = load_warmup_file(str(p))
    engine = _StubEngine(prompt_tokens_per_call=750)

    result = asyncio.run(warm_prefix_cache(engine, loaded))
    assert result["count"] == 2
    assert result["total_prompt_tokens"] == 1500
    # The warmer appends a minimal user message for system-only prompts so
    # templates that require a user (Qwen3.6, DeepSeek-VL) don't error.
    # Verify the system content survives the transformation.
    assert engine.calls[0][0] == prompts[0][0]
    assert engine.calls[1][0] == prompts[1][0]


# ---------------------------------------------------------------------------
# Optimization 1 — user-terminator auto-append (Qwen3.6 / DeepSeek-VL fix)
# ---------------------------------------------------------------------------


def test_ensure_user_terminator_appends_for_system_only():
    msgs = [{"role": "system", "content": "SYS"}]
    out = _ensure_user_terminator(msgs)
    assert out == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": " "},
    ]


def test_ensure_user_terminator_preserves_trailing_user():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]
    out = _ensure_user_terminator(msgs)
    assert out == msgs  # untouched


def test_ensure_user_terminator_preserves_trailing_assistant():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = _ensure_user_terminator(msgs)
    assert out == msgs  # untouched — conversation history is fine


def test_warm_prefix_cache_appends_user_for_system_only():
    """Real behaviour: system-only prompt gets a user appended before going
    to the engine — otherwise Qwen3.6-style templates raise TemplateError."""
    engine = _StubEngine(prompt_tokens_per_call=500)
    asyncio.run(warm_prefix_cache(engine, [AGENT_CODING_MESSAGES_1]))
    # Engine received [system, user-placeholder]
    assert len(engine.calls[0]) == 2
    assert engine.calls[0][0]["role"] == "system"
    assert engine.calls[0][1]["role"] == "user"


# ---------------------------------------------------------------------------
# Optimization 2 — parallel warm-up (asyncio.gather)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Edge cases — load_warmup_file
# ---------------------------------------------------------------------------


def test_load_warmup_file_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_warmup_file(str(p))


def test_load_warmup_file_scalar_top_level(tmp_path: Path):
    p = tmp_path / "str.json"
    p.write_text('"just a string"')
    with pytest.raises(ValueError, match="top-level JSON list"):
        load_warmup_file(str(p))


def test_load_warmup_file_number_top_level(tmp_path: Path):
    p = tmp_path / "num.json"
    p.write_text("42")
    with pytest.raises(ValueError, match="top-level JSON list"):
        load_warmup_file(str(p))


def test_load_warmup_file_non_dict_message(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text('[[["role", "system"]]]')  # list of [list of list]
    with pytest.raises(ValueError, match="expected dict"):
        load_warmup_file(str(p))


def test_load_warmup_file_unicode_content(tmp_path: Path):
    """Unicode in content survives round-trip."""
    p = tmp_path / "u.json"
    content = "Héllo 世界 🌍 ñoño"
    payload = [[{"role": "system", "content": content}]]
    p.write_text(json.dumps(payload, ensure_ascii=False))
    loaded = load_warmup_file(str(p))
    assert loaded[0][0]["content"] == content


def test_load_warmup_file_many_prompts(tmp_path: Path):
    """100 prompts load cleanly."""
    p = tmp_path / "many.json"
    payload = [AGENT_CODING_MESSAGES_1 for _ in range(100)]
    p.write_text(json.dumps(payload))
    loaded = load_warmup_file(str(p))
    assert len(loaded) == 100


# ---------------------------------------------------------------------------
# Edge cases — _ensure_user_terminator
# ---------------------------------------------------------------------------


def test_ensure_user_terminator_empty_list():
    """Empty list → append user (harmless; won't be sent to engine in practice)."""
    out = _ensure_user_terminator([])
    assert out == [{"role": "user", "content": " "}]


def test_ensure_user_terminator_tool_role():
    """Trailing tool message → we still append user (tool is not user/assistant)."""
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "call the tool"},
        {"role": "tool", "name": "calc", "content": "42"},
    ]
    out = _ensure_user_terminator(msgs)
    assert out[-1] == {"role": "user", "content": " "}
    assert len(out) == len(msgs) + 1


def test_ensure_user_terminator_only_user():
    """List with only user (no system) — preserved untouched."""
    msgs = [{"role": "user", "content": "just a user message"}]
    out = _ensure_user_terminator(msgs)
    assert out == msgs


# ---------------------------------------------------------------------------
# Edge cases — warm_prefix_cache
# ---------------------------------------------------------------------------


def test_warm_prefix_cache_engine_never_finishes():
    """Engine yields outputs but never sets finished=True → counted as skipped."""

    class _NeverFinish(_StubEngine):
        async def stream_chat(self, *, messages, max_tokens, temperature, **_):
            self.calls.append(messages)
            yield _FakeOutput("a", finished=False, prompt_tokens=100)
            yield _FakeOutput("b", finished=False, prompt_tokens=100)

    engine = _NeverFinish()
    result = asyncio.run(warm_prefix_cache(engine, [AGENT_CODING_MESSAGES_1]))
    assert result["count"] == 0
    assert result["skipped"] == 1
    assert result["total_prompt_tokens"] == 0


def test_warm_prefix_cache_all_prompts_fail_same_way():
    """All N prompts error identically → skipped=N, count=0, no crash."""
    engine = _StubEngine()

    # Force every call to raise by setting raise_on=0 and iterating manually
    class _AlwaysFail(_StubEngine):
        async def stream_chat(self, *, messages, max_tokens, temperature, **_):
            self.calls.append(messages)
            raise RuntimeError("boom")
            yield  # pragma: no cover — makes it an async generator

    engine = _AlwaysFail()
    prompts = [AGENT_CODING_MESSAGES_1] * 3
    result = asyncio.run(warm_prefix_cache(engine, prompts))
    assert result["count"] == 0
    assert result["skipped"] == 3


def test_warm_prefix_cache_unicode_in_prompts():
    """Unicode content passes through unmodified."""
    engine = _StubEngine(prompt_tokens_per_call=100)
    prompts = [[{"role": "system", "content": "Héllo 世界 🌍"}]]
    result = asyncio.run(warm_prefix_cache(engine, prompts))
    assert result["count"] == 1
    assert engine.calls[0][0]["content"] == "Héllo 世界 🌍"


def test_warm_prefix_cache_missing_prompt_tokens():
    """Engine returns None for prompt_tokens → counted as 0, no crash."""

    class _NoTokenCount(_StubEngine):
        async def stream_chat(self, *, messages, max_tokens, temperature, **_):
            self.calls.append(messages)
            yield _FakeOutput("x", finished=True, prompt_tokens=None)

    engine = _NoTokenCount()
    result = asyncio.run(warm_prefix_cache(engine, [AGENT_CODING_MESSAGES_1]))
    assert result["count"] == 1
    assert result["total_prompt_tokens"] == 0


# ---------------------------------------------------------------------------
# _build_strict_prefix_string — the strict-prefix warmup builder
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer that mimics apply_chat_template returning a string."""

    def __init__(self, render):
        self._render = render

    def apply_chat_template(self, messages, **kwargs):
        return self._render(messages, kwargs)


def test_build_strict_prefix_basic():
    """Qwen-style template: two user probes diverge at user content."""

    def render(msgs, kwargs):
        sys_content = msgs[0]["content"]
        user_content = msgs[-1]["content"]
        return (
            f"<|im_start|>system\n{sys_content}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    tok = _FakeTokenizer(render)
    msgs = [{"role": "system", "content": "You are helpful."}]
    prefix = _build_strict_prefix_string(tok, msgs)
    assert prefix is not None
    # Must end right before the user content insertion point.
    assert prefix.endswith("<|im_start|>user\n")
    # System content preserved.
    assert "You are helpful." in prefix


def test_build_strict_prefix_none_if_tokenizer_lacks_method():
    """Tokenizer without apply_chat_template → None."""

    class _Bare:
        pass

    assert (
        _build_strict_prefix_string(_Bare(), [{"role": "system", "content": "x"}])
        is None
    )


def test_build_strict_prefix_none_on_template_error():
    """apply_chat_template raising returns None (and doesn't propagate)."""

    def render(msgs, kwargs):
        raise RuntimeError("template broken")

    tok = _FakeTokenizer(render)
    assert (
        _build_strict_prefix_string(tok, [{"role": "system", "content": "x"}]) is None
    )


def test_build_strict_prefix_none_if_identical_probes():
    """If the template ignores user content (both probes identical), there's
    no divergence → can't produce a strict prefix → return None."""

    def render(msgs, kwargs):
        return "<|im_start|>system\nSYS<|im_end|><|im_start|>assistant\n"

    tok = _FakeTokenizer(render)
    assert (
        _build_strict_prefix_string(tok, [{"role": "system", "content": "SYS"}]) is None
    )


def test_build_strict_prefix_retries_without_enable_thinking():
    """If the template rejects enable_thinking (non-Qwen models), retry without."""
    calls = []

    def render(msgs, kwargs):
        calls.append(kwargs.copy())
        if "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        sys = msgs[0]["content"]
        user = msgs[-1]["content"]
        return f"<|begin_of_text|>system\n{sys}\n<|end|>user\n{user}\n<|end|>"

    tok = _FakeTokenizer(render)
    prefix = _build_strict_prefix_string(tok, [{"role": "system", "content": "SYS"}])
    assert prefix is not None
    # Called at least twice: first with enable_thinking (failed), then without
    assert any("enable_thinking" in c for c in calls)
    assert any("enable_thinking" not in c for c in calls)


def test_warm_prefix_cache_runs_prompts_concurrently():
    """With N prompts, elapsed_ms should be closer to time-of-one than N * time-of-one.

    We simulate engine latency by sleeping inside the stub. If the warmer
    runs sequentially, total time is N * sleep. If concurrent, ~1 * sleep.
    """
    import asyncio as _asyncio

    SLEEP_PER_CALL = 0.05  # 50 ms

    class _SlowEngine(_StubEngine):
        async def stream_chat(self, *, messages, max_tokens, temperature, **_):
            self.calls.append(messages)
            await _asyncio.sleep(SLEEP_PER_CALL)
            yield _FakeOutput("x", finished=True, prompt_tokens=100)

    N = 5
    engine = _SlowEngine(prompt_tokens_per_call=100)
    prompts = [AGENT_CODING_MESSAGES_1] * N

    result = asyncio.run(warm_prefix_cache(engine, prompts))
    assert result["count"] == N

    # If serial, total >= N * SLEEP_PER_CALL * 1000 = 250 ms.
    # If parallel via asyncio.gather, total is ~SLEEP_PER_CALL * 1000 = 50-100 ms.
    # Allow generous slack; we just need to prove it's NOT serial.
    assert result["elapsed_ms"] < (N * SLEEP_PER_CALL * 1000 * 0.6), (
        f"warm_prefix_cache appears serial: {result['elapsed_ms']:.1f} ms for "
        f"{N} * {SLEEP_PER_CALL*1000:.0f} ms sleep"
    )
