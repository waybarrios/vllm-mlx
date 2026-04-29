"""Tests for _ThinkingAwareLogitsProcessor preamble bypass fix.

The processor must wait for the first JSON-start character ({ or [) before
activating the inner JSONSchemaLogitsProcessor.  Without this, preamble text
like "Here's a " crashes the enforcer with KeyError: ''.
"""

# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Maps token IDs → strings via a simple lookup table."""

    def __init__(self, vocab: dict[int, str]):
        self._vocab = vocab

    def decode(self, ids: list[int]) -> str:
        return "".join(self._vocab.get(i, "") for i in ids)


class _FakeInnerProcessor:
    """Stands in for JSONSchemaLogitsProcessor.

    Records every call so tests can assert when/how activation happened.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._prompt_len: int | None = None
        self._disabled = False
        self.calls: list[tuple[list[int], str]] = []  # (tokens, logits)

    def __call__(self, tokens, logits):
        self.calls.append((list(tokens), logits))
        return "CONSTRAINED"

    @property
    def schema(self):
        return {}


# ---------------------------------------------------------------------------
# Import the class under test
# ---------------------------------------------------------------------------

from vllm_mlx.server import _ThinkingAwareLogitsProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_processor(vocab: dict[int, str]):
    tok = _FakeTokenizer(vocab)
    inner = _FakeInnerProcessor(tok)
    proc = _ThinkingAwareLogitsProcessor(inner)
    return proc, inner


def _feed(proc, prompt_ids: list[int], gen_ids: list[int]):
    """Feed tokens one-by-one and return list of results."""
    results = []
    for i in range(1, len(gen_ids) + 1):
        tokens = prompt_ids + gen_ids[:i]
        result = proc(tokens, "LOGITS")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreambleSkipped:
    """No <think> emitted, model generates preamble before JSON."""

    def test_preamble_skipped_before_json(self):
        # Tokens: "Here" "'s" " a" " {" '"key"' '}'
        vocab = {10: "Here", 11: "'s", 12: " a", 13: " {", 14: '"key"', 15: "}"}
        proc, inner = _make_processor(vocab)
        prompt = [1, 2, 3]  # 3 prompt tokens

        results = _feed(proc, prompt, [10, 11, 12, 13, 14, 15])

        # First 3 tokens (detection window): no <think> detected → enters waiting
        # Token 10,11,12 = "Here's a" — no { or [ → returns LOGITS
        assert results[0] == "LOGITS"
        assert results[1] == "LOGITS"
        assert results[2] == "LOGITS"

        # Token 13 = " {" — JSON start detected → enforcer activated
        assert results[3] == "CONSTRAINED"
        assert proc._active is True
        # prompt_len should point to the { token (index 3 in gen = base+3)
        assert inner._prompt_len == len(prompt) + 3  # skip "Here's a "

        # Subsequent tokens go through inner directly
        assert results[4] == "CONSTRAINED"
        assert results[5] == "CONSTRAINED"

    def test_direct_json_no_delay(self):
        # Model emits { as very first token (no preamble)
        vocab = {10: "{", 11: '"k"', 12: "}"}
        proc, inner = _make_processor(vocab)
        prompt = [1, 2, 3]

        results = _feed(proc, prompt, [10, 11, 12])

        # Tokens 10,11: detection window (< 3 tokens) → wait
        assert results[0] == "LOGITS"
        assert results[1] == "LOGITS"
        # Token 12: 3 tokens, no <think> → enters waiting → immediate scan
        # finds { at gen index 0 → activate
        assert results[2] == "CONSTRAINED"
        assert proc._active is True
        assert inner._prompt_len == len(prompt) + 0  # { is at gen position 0


class TestThinkThenPreamble:
    """Model emits <think>...</think> then preamble before JSON."""

    def test_think_then_preamble_then_json(self):
        vocab = {
            20: "<think>",
            21: "I need",
            22: "</think>",
            23: "Here",
            24: " {",
            25: "}",
        }
        proc, inner = _make_processor(vocab)
        prompt = [1, 2]

        results = _feed(proc, prompt, [20, 21, 22, 23, 24, 25])

        # Token 20: <think> detected → in_thinking=True
        assert results[0] == "LOGITS"

        # Token 21: still thinking
        assert results[1] == "LOGITS"

        # Token 22: </think> detected → enters waiting → immediate scan
        # gen_tokens = [20,21,22], none contain { → still waiting
        assert results[2] == "LOGITS"
        # (state is waiting_for_json after this point)

        # Token 23: "Here" — no { → still waiting
        assert results[3] == "LOGITS"

        # Token 24: " {" — JSON start → activate
        assert results[4] == "CONSTRAINED"
        assert proc._active is True

        # Token 25: direct pass-through
        assert results[5] == "CONSTRAINED"

    def test_think_then_direct_json(self):
        """</think> immediately followed by { — no preamble."""
        vocab = {20: "<think>", 21: "ok", 22: "</think>", 23: "{", 24: "}"}
        proc, inner = _make_processor(vocab)
        prompt = [1]

        results = _feed(proc, prompt, [20, 21, 22, 23, 24])

        # After </think>, next token is { → immediate activation
        assert results[2] == "LOGITS"  # </think> → enters waiting
        assert results[3] == "CONSTRAINED"  # { → activated
        assert inner._prompt_len == len(prompt) + 3  # { at gen index 3


class TestSafetyLimit:
    """50+ preamble tokens without JSON start → forced activation."""

    def test_safety_limit_activates_after_50_tokens(self):
        # 55 preamble tokens without { or [
        vocab = {i: f"word{i}" for i in range(100, 160)}
        proc, inner = _make_processor(vocab)
        prompt = [1, 2, 3]
        gen = list(range(100, 156))  # 56 tokens

        results = _feed(proc, prompt, gen)

        # First 3 tokens: detection → waiting → immediate scan (3 tokens, <50)
        # Tokens 4-51: no { → waiting
        # When gen_tokens > 50: safety limit → forced activation
        assert proc._active is True
        # The last result should be CONSTRAINED
        assert results[-1] == "CONSTRAINED"
        # Inner was activated with prompt_len = n (total tokens at activation)
        assert inner._prompt_len is not None
        assert inner._prompt_len > len(prompt)  # activated past prompt


class TestArrayJsonStart:
    """JSON arrays start with [ not {."""

    def test_array_start_detected(self):
        vocab = {10: "The", 11: " result", 12: " [", 13: "1", 14: "]"}
        proc, inner = _make_processor(vocab)
        prompt = [1, 2, 3]

        results = _feed(proc, prompt, [10, 11, 12, 13, 14])

        # Tokens 10,11: detection window (< 3) → wait
        assert results[0] == "LOGITS"
        assert results[1] == "LOGITS"
        # Token 12: 3 tokens, no <think> → enters waiting → immediate scan
        # Scans gen_tokens [10,11,12]: "The"=no, " result"=no, " ["=yes!
        # → activates at gen index 2
        assert results[2] == "CONSTRAINED"
        assert proc._active is True
        assert inner._prompt_len == len(prompt) + 2


class TestEnforcerNeverCalledDuringWaiting:
    """Inner processor must never be called during detection/thinking/waiting
    phases (except when activating)."""

    def test_no_inner_calls_during_phases(self):
        vocab = {20: "<think>", 21: "reasoning", 22: "</think>", 23: "preamble"}
        proc, inner = _make_processor(vocab)
        prompt = [1, 2]

        _feed(proc, prompt, [20, 21, 22, 23])

        # Inner should have zero calls — still in waiting phase, no { found
        assert len(inner.calls) == 0
        assert proc._waiting_for_json is True
        assert proc._active is False


class TestBPETokenWithJsonChar:
    """BPE tokens like '\\n{' contain { embedded in whitespace."""

    def test_newline_brace_detected(self):
        vocab = {10: "Ok", 11: "!", 12: " ", 13: "\n{", 14: "}"}
        proc, inner = _make_processor(vocab)
        prompt = [1, 2, 3]

        results = _feed(proc, prompt, [10, 11, 12, 13, 14])

        # 3 tokens → waiting, then scan finds \n{ at gen index 3
        assert results[3] == "CONSTRAINED"
        assert inner._prompt_len == len(prompt) + 3


class TestThinkingSpanWithJson:
    """JSON-like text inside <think>...</think> must NOT trigger the enforcer.

    Regression test for review feedback: the scan must start AFTER the
    thinking span, not from _base_prompt_len.
    """

    def test_json_inside_thinking_ignored(self):
        # <think>consider {"bad": true}</think>Here is {"ok": true}
        vocab = {
            30: "<think>",
            31: "consider",
            32: ' {"bad"',  # { inside thinking — must be ignored
            33: ": true}",
            34: "</think>",
            35: "Here is",
            36: ' {"ok"',  # { in answer — must trigger enforcer
            37: ": true}",
        }
        proc, inner = _make_processor(vocab)
        prompt = [1, 2]
        gen = [30, 31, 32, 33, 34, 35, 36, 37]

        # Step through token-by-token to check state at each point
        def step(i):
            return proc(prompt + gen[: i + 1], "LOGITS")

        # Tokens 30-33: thinking phase (contains { at token 32)
        assert step(0) == "LOGITS"  # <think>
        assert step(1) == "LOGITS"  # consider
        assert step(2) == "LOGITS"  # {"bad" — inside thinking, NOT activated
        assert step(3) == "LOGITS"  # : true}

        # Token 34: </think> → enters waiting, scan starts from HERE
        assert step(4) == "LOGITS"
        assert proc._waiting_for_json is True
        assert proc._active is False  # { inside thinking was NOT matched

        # Token 35: "Here is" — no { → still waiting
        assert step(5) == "LOGITS"

        # Token 36: ' {"ok"' — { found → enforcer activates
        assert step(6) == "CONSTRAINED"
        assert proc._active is True
        # prompt_len must point to the answer {, NOT the thinking {
        # scan_offset=7 (n at </think>), scan_tokens=[35,36], 36 at index 1
        assert inner._prompt_len == len(prompt) + 6  # absolute pos of {"ok"

        # Token 37: direct pass-through
        assert step(7) == "CONSTRAINED"

    def test_json_array_inside_thinking_ignored(self):
        # <think>output [1,2,3]</think>[{"result": 1}]
        vocab = {
            40: "<think>",
            41: "output",
            42: " [1",  # [ inside thinking — must be ignored
            43: ",2,3]",
            44: "</think>",
            45: "[",  # [ in answer — must trigger enforcer
            46: "]",
        }
        proc, inner = _make_processor(vocab)
        prompt = [1]

        results = _feed(proc, prompt, [40, 41, 42, 43, 44, 45, 46])

        # Thinking tokens with [ should not activate
        assert all(r == "LOGITS" for r in results[:4])

        # </think> → waiting, no { or [ yet in post-think scan
        assert results[4] == "LOGITS"

        # [ in answer → activate
        assert results[5] == "CONSTRAINED"
        assert proc._active is True

    def test_safety_limit_counts_from_scan_offset(self):
        """50-token safety limit must count from the scan offset, not from
        _base_prompt_len.  Long thinking should not exhaust the limit."""
        # 10 thinking tokens (some with {), then 5 post-think tokens (no {)
        vocab = {}
        think_tokens = []
        # <think>
        vocab[50] = "<think>"
        think_tokens.append(50)
        # 8 reasoning tokens, some with {
        for i in range(51, 59):
            vocab[i] = '{"x"}' if i % 3 == 0 else f"reason{i}"
            think_tokens.append(i)
        # </think>
        vocab[59] = "</think>"
        think_tokens.append(59)

        # 5 post-think preamble tokens (no { or [)
        post_tokens = []
        for i in range(60, 65):
            vocab[i] = f"word{i}"
            post_tokens.append(i)

        proc, inner = _make_processor(vocab)
        prompt = [1, 2]
        gen = think_tokens + post_tokens

        results = _feed(proc, prompt, gen)

        # Should still be waiting — only 5 tokens past scan offset, not 50
        assert proc._waiting_for_json is True
        assert proc._active is False
        assert all(r == "LOGITS" for r in results)
