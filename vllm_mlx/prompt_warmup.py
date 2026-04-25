# SPDX-License-Identifier: Apache-2.0
"""Prompt warm-up for vllm-mlx.

At server startup, pre-populates the prefix cache by running one short
generation per warm-up prompt. The first user request that shares a prefix
with a warmed prompt sees cache-hit TTFT instead of cold prefill latency.

File format (JSON):
  [
    [{"role": "system", "content": "You are ..."}],
    [{"role": "system", "content": "..."}, {"role": "user", "content": "hi"}]
  ]

Each entry is a list of chat messages — same shape as a ``/v1/chat/completions``
``messages`` field. The warmer runs a ``max_tokens=1`` chat completion for each,
which flows through the exact same path as a real request and writes the KV
state to the prefix cache.

Paths resolve from the current working directory. A single-message system
prompt is sufficient if that is the shared prefix.

Sizing note: prompts are warmed concurrently via ``asyncio.gather``, so N
entries fire N concurrent prefills at startup. Each prefill allocates KV
cache for its prompt length. For typical agent deployments 1–3 entries
(one per active persona) cover the hot paths; a very large warm-prompts
file on a memory-tight model can exhaust headroom at boot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_warmup_file(path: str) -> list[list[dict[str, Any]]]:
    """Load and validate a warm-up prompts JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file shape is invalid.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Warm-up prompts file not found: {p}")

    data = json.loads(p.read_text())
    if not isinstance(data, list):
        raise ValueError(
            f"Warm-up file must contain a top-level JSON list, got {type(data).__name__}"
        )

    if not data:
        raise ValueError(f"Warm-up file is empty: {p}")

    for i, entry in enumerate(data):
        if not isinstance(entry, list) or not entry:
            raise ValueError(
                f"Warm-up entry {i}: expected non-empty list of message dicts"
            )
        for j, msg in enumerate(entry):
            if not isinstance(msg, dict):
                raise ValueError(
                    f"Warm-up entry {i} message {j}: expected dict, got {type(msg).__name__}"
                )
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"Warm-up entry {i} message {j}: missing 'role' or 'content'"
                )

    return data


def _ensure_user_terminator(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure the message list ends with a user message.

    Some chat templates (Qwen3.6, DeepSeek-VL, a handful of others) require
    at least one user message or raise ``TemplateError: No user query found``.
    We prefer to cache just the system prefix, but when the template won't
    render without a user, append a minimal placeholder. The common prefix
    up to the start of user content still matches real requests, so the
    system tokens still get cached.
    """
    if messages and messages[-1].get("role") in ("user", "assistant"):
        return messages
    return [*messages, {"role": "user", "content": " "}]


def _build_strict_prefix_string(
    tokenizer: Any, messages: list[dict[str, Any]], enable_thinking: bool = True
) -> str | None:
    """Build a STRING prefix that is a prefix of any real request's rendered
    chat template for the same system and empty chat history.

    Strategy: render the chat template twice with two DIFFERENT user contents
    and ``tokenize=False`` (matching what the server does). Truncate the
    first output at the position where the two strings diverge — that's
    where user content gets inserted.

    We return a STRING (not tokens) because the engine's request path also
    applies the template with ``tokenize=False`` and then lets the tokenizer
    encode the result. Going through the same pipeline guarantees the warm
    entry's tokens are a strict prefix of a real request's tokens.

    This enables warm-prompts on hybrid SSM+attention models where LCP
    matching is disabled (SSM state can't be trimmed) — they rely purely
    on strict PREFIX match.

    Returns None if rendering fails or the two probes don't diverge past a
    reasonable prefix length (unusual template).
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return None

    def _with_user(user_content: str) -> list[dict[str, Any]]:
        msgs = [dict(m) for m in messages]
        if msgs and msgs[-1].get("role") == "user":
            msgs[-1] = {**msgs[-1], "content": user_content}
        else:
            msgs = [*msgs, {"role": "user", "content": user_content}]
        return msgs

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }

    # Multi-character probes that differ at position 0. Multi-char is safer
    # than "A"/"B" against hypothetical templates that treat single-character
    # content specially (e.g. stripping or wrapping). Differ-at-position-0 is
    # required: probes that share a prefix (e.g. "__PROBE_A__"/"__PROBE_B__")
    # would make the divergence loop overshoot into the shared-prefix region
    # and cache bytes that are not in real requests.
    probe_a, probe_b = "Alpha", "Bravo"
    try:
        a = apply(_with_user(probe_a), **kwargs)
        b = apply(_with_user(probe_b), **kwargs)
    except Exception:
        # Template may reject enable_thinking on non-Qwen models — retry without
        try:
            kwargs.pop("enable_thinking", None)
            a = apply(_with_user(probe_a), **kwargs)
            b = apply(_with_user(probe_b), **kwargs)
        except Exception:
            return None

    if not isinstance(a, str) or not isinstance(b, str):
        return None

    # Character-level divergence — the boundary is where user content starts.
    # Track whether we actually diverged; identical probes mean the template
    # ignored user content (unusual — bail out rather than cache the full
    # rendering which would include whatever template fallback was used).
    boundary = 0
    diverged = False
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            diverged = True
            break
        boundary = i + 1

    if not diverged:
        return None

    # Require a reasonable prefix. Too-short means the template is unusual.
    if boundary < 16:
        return None

    return a[:boundary]


async def warm_prefix_cache(
    engine: Any,
    prompts: list[list[dict[str, Any]]],
    *,
    max_tokens: int = 1,
) -> dict[str, Any]:
    """Run each prompt through the engine to populate the prefix cache.

    Prefers the strict-prefix path when the engine exposes a tokenizer:
    manually tokenize with ``add_generation_prompt=False`` and feed the
    raw token IDs to the engine's ``stream_generate`` (which accepts
    ``prompt: str | list[int]``). Real requests — which always use
    ``add_generation_prompt=True`` — will then find the warm entry as an
    exact strict prefix, independent of the engine's LCP matcher.

    This is the difference between warm-prompts helping dense models only
    and helping hybrid SSM+attention models too.

    Falls back to ``engine.stream_chat`` with a placeholder user message
    appended if no tokenizer is exposed — strict-prefix match won't apply
    there, so the feature is effectively LCP-only for that engine.

    Runs all prompts concurrently (``asyncio.gather``).

    Args:
        engine: The vllm-mlx engine (exposes ``stream_chat`` and optionally
            ``tokenizer`` + ``stream_generate``).
        prompts: List of message arrays.
        max_tokens: Tokens to generate per warm-up. 1 is enough.

    Returns:
        Dict with ``count``, ``skipped``, ``elapsed_ms``,
        ``total_prompt_tokens``, and ``mode`` (``"strict-prefix"`` or
        ``"chat-fallback"``) describing which path was used.
    """
    tokenizer = getattr(engine, "tokenizer", None)
    use_strict_prefix = tokenizer is not None and hasattr(engine, "stream_generate")

    async def _one_strict(
        idx: int, messages: list[dict[str, Any]]
    ) -> tuple[int, int, str | None]:
        prefix_str = _build_strict_prefix_string(tokenizer, messages)
        if prefix_str is None:
            return await _one_chat(idx, messages)
        try:
            async for output in engine.stream_generate(
                prompt=prefix_str,
                max_tokens=max_tokens,
                temperature=0.0,
            ):
                if output.finished:
                    return 1, int(output.prompt_tokens or 0), None
            return 0, 0, "no finished output"
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            logger.warning("[warmup] prompt %d (strict) failed: %s", idx, err)
            return await _one_chat(idx, messages)

    async def _one_chat(
        idx: int, messages: list[dict[str, Any]]
    ) -> tuple[int, int, str | None]:
        patched = _ensure_user_terminator(messages)
        try:
            async for output in engine.stream_chat(
                messages=patched,
                max_tokens=max_tokens,
                temperature=0.0,
            ):
                if output.finished:
                    return 1, int(output.prompt_tokens or 0), None
            return 0, 0, "no finished output"
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            logger.warning("[warmup] prompt %d (chat) failed: %s", idx, err)
            return 0, 0, err

    runner = _one_strict if use_strict_prefix else _one_chat
    mode = "strict-prefix" if use_strict_prefix else "chat-fallback"

    t0 = time.perf_counter()
    results = await asyncio.gather(
        *(runner(i, msgs) for i, msgs in enumerate(prompts)),
        return_exceptions=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    completed = sum(r[0] for r in results)
    total_prompt_tokens = sum(r[1] for r in results)
    skipped = sum(1 for r in results if r[2] is not None)

    return {
        "count": completed,
        "skipped": skipped,
        "elapsed_ms": elapsed_ms,
        "total_prompt_tokens": total_prompt_tokens,
        "mode": mode,
    }
