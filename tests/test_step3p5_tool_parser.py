# SPDX-License-Identifier: Apache-2.0
"""Tests for Step3p5 XML tool call parsing."""

import json

from vllm_mlx.tool_parsers import ToolParserManager


def test_step3p5_tool_parser_is_registered_and_native():
    parser_cls = ToolParserManager.get_tool_parser("step3p5")

    assert parser_cls.__name__ == "Step3p5ToolParser"
    assert parser_cls.supports_native_format() is True
    assert ToolParserManager.get_tool_parser("step") is parser_cls


def test_step3p5_tool_parser_extracts_template_xml_call():
    parser = ToolParserManager.get_tool_parser("step3p5")()

    result = parser.extract_tool_calls(
        "I should call the tool.\n"
        "</think>\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>\n"
        "Austin\n"
        "</parameter>\n"
        "<parameter=units>\n"
        "metric\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )

    assert result.tools_called is True
    assert result.content is None
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert json.loads(tool_call["arguments"]) == {
        "city": "Austin",
        "units": "metric",
    }


def test_step3p5_tool_parser_preserves_non_tool_content():
    parser = ToolParserManager.get_tool_parser("step3p5")()

    result = parser.extract_tool_calls("No tool is required here.")

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == "No tool is required here."


def test_step3p5_tool_parser_streaming_buffers_until_tool_close():
    parser = ToolParserManager.get_tool_parser("step3p5")()

    first = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="<tool_call>\n<function=get_weather>\n",
        delta_text="<tool_call>\n<function=get_weather>\n",
    )
    second = parser.extract_tool_calls_streaming(
        previous_text="<tool_call>\n<function=get_weather>\n",
        current_text=(
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Austin</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
        delta_text="<parameter=city>Austin</parameter>\n</function>\n</tool_call>",
    )

    assert first is None
    assert second is not None
    assert "tool_calls" in second
    assert second["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(second["tool_calls"][0]["function"]["arguments"]) == {
        "city": "Austin"
    }


def test_step3p5_tool_parser_collapses_consecutive_duplicate_calls():
    parser = ToolParserManager.get_tool_parser("step3p5")()
    repeated_call = (
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>Austin</parameter>\n"
        "<parameter=units>metric</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )

    result = parser.extract_tool_calls(repeated_call * 4)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert json.loads(tool_call["arguments"]) == {
        "city": "Austin",
        "units": "metric",
    }


def test_step3p5_tool_parser_preserves_distinct_consecutive_calls():
    parser = ToolParserManager.get_tool_parser("step3p5")()
    output = (
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>Austin</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>Dallas</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )

    result = parser.extract_tool_calls(output)

    assert len(result.tool_calls) == 2
    assert [json.loads(call["arguments"])["city"] for call in result.tool_calls] == [
        "Austin",
        "Dallas",
    ]
