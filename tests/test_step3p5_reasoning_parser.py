# SPDX-License-Identifier: Apache-2.0
"""Tests for Step3p5 reasoning parser registration and trimming."""

from vllm_mlx.reasoning import get_parser, list_parsers


def test_step3p5_reasoning_parser_is_registered():
    assert "step3p5" in list_parsers()


def test_step3p5_prompt_seeded_reasoning_splits_final_content():
    parser = get_parser("step3p5")()

    reasoning, content = parser.extract_reasoning(
        "The user asked for a direct acknowledgement.\n</think>\nOK"
    )

    assert reasoning == "The user asked for a direct acknowledgement."
    assert content == "OK"


def test_step3p5_reasoning_trims_newline_around_end_token():
    parser = get_parser("step3p5")()

    reasoning, content = parser.extract_reasoning(
        "<think>\nWork through the request.\n</think>\nFinal answer"
    )

    assert reasoning == "Work through the request."
    assert content == "Final answer"


def test_step3p5_no_tag_output_remains_visible_content():
    parser = get_parser("step3p5")()

    reasoning, content = parser.extract_reasoning("OK")

    assert reasoning is None
    assert content == "OK"


def test_step3p5_streaming_drops_newline_after_end_token():
    parser = get_parser("step3p5")()

    first = parser.extract_reasoning_streaming(
        previous_text="hidden reasoning",
        current_text="hidden reasoning</think>",
        delta_text="</think>",
    )
    second = parser.extract_reasoning_streaming(
        previous_text="hidden reasoning</think>",
        current_text="hidden reasoning</think>\n",
        delta_text="\n",
    )
    third = parser.extract_reasoning_streaming(
        previous_text="hidden reasoning</think>\n",
        current_text="hidden reasoning</think>\nFinal",
        delta_text="Final",
    )

    assert first is None or first.content is None
    assert second is None
    assert third is not None
    assert third.content == "Final"
