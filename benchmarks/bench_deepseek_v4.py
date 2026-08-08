"""Benchmark: DeepSeek-V4 prompt encoding and streaming parser overhead.

Covers both serving paths. Single-stream is one sequence decoded token by
token, where the parser sits directly in the latency path. Batch is N
concurrent sequences interleaved, the shape continuous batching produces: each
request carries its own parser state, so per-token cost is paid N times per
decode step and any per-token work that scales with output length compounds.

The thing to watch for is quadratic scaling. A parser that rescans the whole
accumulated text on every delta is O(N²) over a generation, which is invisible
on short replies and dominant on long ones — so the ms/tok column is the one
that matters, not the totals. It is currently flat for the DSML tool parser and
grows for the reasoning path; see the notes the script prints at the end.

Usage:
    python benchmarks/bench_deepseek_v4.py
"""

import time

from vllm_mlx.reasoning.deepseek_v4_parser import DeepSeekV4ReasoningParser
from vllm_mlx.tool_parsers.deepseek_v4_tool_parser import DeepSeekV4ToolParser
from vllm_mlx.utils.deepseek_v4_encoding import apply_chat_template

D = "｜DSML｜"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    }
]


def reasoning_tokens(n: int) -> list[str]:
    """A plain thinking turn: N reasoning tokens, then a short answer."""
    return (
        [f"step{i} " for i in range(n)]
        + ["</think>"]
        + [f"word{i} " for i in range(10)]
    )


def tool_call_tokens(n: int) -> list[str]:
    """A tool-calling turn: N reasoning tokens, then DSML markup.

    The markup is split the way the tokenizer splits it — ｜DSML｜ has an id of
    its own, the surrounding punctuation does not — so the parser sees the same
    fragmentation it sees in production.
    """
    tokens = [f"step{i} " for i in range(n)] + ["</think>", "\n\n"]
    tokens += ["<", D, "tool_calls", ">", "\n"]
    tokens += ["<", D, "invoke", ' name="get_weather"', ">", "\n"]
    tokens += [
        "<",
        D,
        "parameter",
        ' name="city"',
        ' string="true"',
        ">",
        "Prague",
        "</",
        D,
        "parameter",
        ">",
        "\n",
    ]
    tokens += ["</", D, "invoke", ">", "\n"]
    tokens += ["</", D, "tool_calls", ">"]
    return tokens


def bench_stream(make_tokens, n_tokens: int, streams: int) -> tuple[float, int]:
    """Interleave `streams` concurrent sequences, one delta each per step.

    Returns (total ms, total deltas). With streams=1 this is the single-stream
    latency path; above that it is the batched decode step.
    """
    token_lists = [make_tokens(n_tokens) for _ in range(streams)]
    parsers = []
    for _ in range(streams):
        reasoner = DeepSeekV4ReasoningParser()
        reasoner.reset_state()
        tools = DeepSeekV4ToolParser()
        tools.reset()
        parsers.append((reasoner, tools, {"acc": "", "tool_acc": ""}))

    steps = max(len(t) for t in token_lists)
    deltas = 0
    start = time.perf_counter()
    for step in range(steps):
        for stream in range(streams):
            tokens = token_lists[stream]
            if step >= len(tokens):
                continue
            reasoner, tools, state = parsers[stream]
            delta = tokens[step]
            previous, state["acc"] = state["acc"], state["acc"] + delta
            deltas += 1

            message = reasoner.extract_reasoning_streaming(
                previous, state["acc"], delta
            )
            if message is None or not message.content:
                continue
            prev_tool = state["tool_acc"]
            state["tool_acc"] = prev_tool + message.content
            tools.extract_tool_calls_streaming(
                prev_tool, state["tool_acc"], message.content
            )
    return (time.perf_counter() - start) * 1000, deltas


def bench_tool_parser_only(make_tokens, n_tokens: int) -> tuple[float, int]:
    """The DSML parser without the reasoning parser in front of it.

    Isolates how much of the per-token cost is the tool parser's own work.
    """
    tokens = make_tokens(n_tokens)
    parser = DeepSeekV4ToolParser()
    parser.reset()

    accumulated = ""
    start = time.perf_counter()
    for delta in tokens:
        previous, accumulated = accumulated, accumulated + delta
        parser.extract_tool_calls_streaming(previous, accumulated, delta)
    return (time.perf_counter() - start) * 1000, len(tokens)


def bench_encoder(turns: int, repeats: int = 200) -> float:
    """Prompt build cost for a conversation of `turns` user/assistant pairs."""
    conversation = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(turns):
        conversation.append({"role": "user", "content": f"Question number {i}?"})
        conversation.append(
            {
                "role": "assistant",
                "content": f"Answer number {i}.",
                "reasoning_content": f"Thinking about question {i} at some length.",
            }
        )
    conversation.append({"role": "user", "content": "And finally?"})

    start = time.perf_counter()
    for _ in range(repeats):
        apply_chat_template(conversation, tools=TOOLS)
    return (time.perf_counter() - start) * 1000 / repeats


def main():
    print("DeepSeek-V4 encoder and parser benchmark")
    print("=" * 68)

    print("\nPrompt encoding (per call, tools attached)")
    for turns in (1, 4, 16, 64):
        ms = bench_encoder(turns)
        print(f"  {turns * 2 + 2:>4} messages -> {ms:>8.3f} ms")

    for label, make_tokens in (
        ("plain thinking turn", reasoning_tokens),
        ("tool-calling turn", tool_call_tokens),
    ):
        print(f"\nSingle stream, {label}")
        for n in (100, 500, 1000, 2000, 5000):
            ms, deltas = bench_stream(make_tokens, n, streams=1)
            print(
                f"  {n:>5} reasoning tokens -> {ms:>8.2f} ms total, "
                f"{ms / deltas:>7.4f} ms/tok"
            )

    print("\nDSML tool parser alone, tool-calling turn")
    for n in (100, 500, 1000, 2000, 5000):
        ms, deltas = bench_tool_parser_only(tool_call_tokens, n)
        print(
            f"  {n:>5} reasoning tokens -> {ms:>8.2f} ms total, "
            f"{ms / deltas:>7.4f} ms/tok"
        )

    print("\nBatched decode, tool-calling turn, 1000 reasoning tokens each")
    for streams in (1, 2, 4, 8, 16):
        ms, deltas = bench_stream(tool_call_tokens, 1000, streams=streams)
        print(
            f"  {streams:>3} concurrent -> {ms:>8.2f} ms total, "
            f"{ms / deltas:>7.4f} ms/tok"
        )

    print("\nReading the numbers:")
    print("  At 50 tok/s the per-token budget is 20 ms, so anything under")
    print("  0.1 ms/tok is noise. Batching adds no per-token cost — each")
    print("  request carries independent parser state and the totals scale")
    print("  linearly with the number of streams.")
    print()
    print("  The single-stream ms/tok does grow with output length. That comes")
    print("  from BaseThinkingReasoningParser, which searches the accumulated")
    print("  text for its start and end tags on every delta while the reasoning")
    print("  block is open; the DSML tool parser on its own stays flat. It is")
    print("  0.2% of the decode budget even at 5000 tokens, but it is quadratic,")
    print("  so it is worth fixing in the base class rather than per model.")


if __name__ == "__main__":
    main()
