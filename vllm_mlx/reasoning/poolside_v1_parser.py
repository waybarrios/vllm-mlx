# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Poolside Laguna models."""

from .qwen3_parser import Qwen3ReasoningParser


class PoolsideV1ReasoningParser(Qwen3ReasoningParser):
    """Parse Laguna's template-injected ``<think>`` reasoning boundary.

    vllm-mlx reasoning parsers receive only the generated assistant text, not
    prompt-history token IDs. Historical ``</think>`` markers therefore cannot
    affect this output-scoped parser as they can in token-aware vLLM serving.
    """
