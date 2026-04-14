"""Benchmark: reasoning parser streaming performance.

Measures per-token overhead of extract_reasoning_streaming() at various
output lengths. Demonstrates the difference between O(N²) accumulated
text scanning and O(1) state-machine tracking.

Usage:
    python benchmarks/bench_reasoning_parser.py
"""

import time

from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser


def bench_streaming(parser, n_tokens: int, label: str) -> float:
    """Simulate n_tokens of streaming through the parser. Returns total ms."""
    parser.reset_state()

    # Simulate: <think> + N reasoning tokens + </think> + 10 content tokens
    tokens = ["<think>"]
    tokens += [f"word{i} " for i in range(n_tokens)]
    tokens += ["</think>"]
    tokens += [f"answer{i} " for i in range(10)]

    accumulated = ""
    start = time.perf_counter()
    for tok in tokens:
        prev = accumulated
        accumulated += tok
        parser.extract_reasoning_streaming(prev, accumulated, tok)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"  {label}: {n_tokens:>6} tokens -> {elapsed:>8.2f}ms "
          f"({elapsed / (n_tokens + 11):.3f}ms/tok)")
    return elapsed


def main():
    parser = Qwen3ReasoningParser()

    print("Reasoning parser streaming benchmark")
    print("=" * 60)
    print()

    for n in [50, 100, 200, 500, 1000, 2000, 5000]:
        bench_streaming(parser, n, f"{n} tokens")

    print()
    print("At 50 tok/s, per-token budget is 20ms.")
    print("Parser overhead should be <0.1ms/tok to be negligible.")


if __name__ == "__main__":
    main()
