# SPDX-License-Identifier: Apache-2.0
"""Tests for the Gemma 4 tool call parser."""

import json

from vllm_mlx.tool_parsers.gemma4_tool_parser import (
    Gemma4ToolParser,
    _gemma_args_to_json,
)

# The Gemma 4 escape token for quotes
Q = '<|"|>'


def test_args_to_json_simple_strings():
    raw = f"location:{Q}San Francisco{Q},unit:{Q}celsius{Q}"
    result = "{" + _gemma_args_to_json(raw) + "}"
    parsed = json.loads(result)
    assert parsed == {"location": "San Francisco", "unit": "celsius"}


def test_args_to_json_numbers():
    raw = "temperature:15,humidity:0.75"
    result = "{" + _gemma_args_to_json(raw) + "}"
    parsed = json.loads(result)
    assert parsed == {"temperature": 15, "humidity": 0.75}


def test_args_to_json_booleans():
    raw = "enabled:true,verbose:false"
    result = "{" + _gemma_args_to_json(raw) + "}"
    parsed = json.loads(result)
    assert parsed == {"enabled": True, "verbose": False}


def test_args_to_json_mixed():
    raw = f"name:{Q}test{Q},count:42,active:true"
    result = "{" + _gemma_args_to_json(raw) + "}"
    parsed = json.loads(result)
    assert parsed == {"name": "test", "count": 42, "active": True}


def test_args_to_json_nested_object():
    raw = f"query:{Q}weather{Q},options:{{format:{Q}json{Q},verbose:true}}"
    result = "{" + _gemma_args_to_json(raw) + "}"
    parsed = json.loads(result)
    assert parsed == {"query": "weather", "options": {"format": "json", "verbose": True}}


def test_args_to_json_array():
    raw = f"tags:[{Q}a{Q},{Q}b{Q},{Q}c{Q}]"
    result = "{" + _gemma_args_to_json(raw) + "}"
    parsed = json.loads(result)
    assert parsed == {"tags": ["a", "b", "c"]}


def test_args_to_json_string_with_special_chars():
    raw = f'message:{Q}Hello, how are you? I\'m fine!{Q}'
    result = "{" + _gemma_args_to_json(raw) + "}"
    parsed = json.loads(result)
    assert parsed == {"message": "Hello, how are you? I'm fine!"}


def test_args_to_json_string_with_embedded_quotes():
    raw = f'message:{Q}She said "hello"{Q}'
    result = "{" + _gemma_args_to_json(raw) + "}"
    parsed = json.loads(result)
    assert parsed == {"message": 'She said "hello"'}


def test_args_to_json_null():
    raw = f"name:{Q}test{Q},value:null"
    result = "{" + _gemma_args_to_json(raw) + "}"
    parsed = json.loads(result)
    assert parsed == {"name": "test", "value": None}


def test_extract_single_tool_call():
    parser = Gemma4ToolParser()
    output = f"<|tool_call>call:get_weather{{location:{Q}San Francisco{Q},unit:{Q}celsius{Q}}}<tool_call|>"
    result = parser.extract_tool_calls(output)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc["name"] == "get_weather"
    args = json.loads(tc["arguments"])
    assert args == {"location": "San Francisco", "unit": "celsius"}
    assert tc["id"].startswith("call_")


def test_extract_multiple_tool_calls():
    parser = Gemma4ToolParser()
    output = (
        f"<|tool_call>call:get_weather{{location:{Q}SF{Q}}}<tool_call|>"
        f"<|tool_call>call:get_time{{timezone:{Q}PST{Q}}}<tool_call|>"
    )
    result = parser.extract_tool_calls(output)

    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0]["name"] == "get_weather"
    assert result.tool_calls[1]["name"] == "get_time"


def test_extract_tool_call_with_surrounding_text():
    parser = Gemma4ToolParser()
    output = f"Let me check the weather. <|tool_call>call:get_weather{{location:{Q}SF{Q}}}<tool_call|> Done."
    result = parser.extract_tool_calls(output)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.content == "Let me check the weather.  Done."


def test_no_tool_calls():
    parser = Gemma4ToolParser()
    output = "Just a regular response with no tool calls."
    result = parser.extract_tool_calls(output)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == output


def test_extract_with_think_tags():
    parser = Gemma4ToolParser()
    output = f"<think>I should check the weather</think><|tool_call>call:get_weather{{location:{Q}SF{Q}}}<tool_call|>"
    result = parser.extract_tool_calls(output)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "get_weather"


def test_extract_numeric_args():
    parser = Gemma4ToolParser()
    output = f"<|tool_call>call:set_temp{{value:72,unit:{Q}fahrenheit{Q}}}<tool_call|>"
    result = parser.extract_tool_calls(output)

    assert result.tools_called is True
    args = json.loads(result.tool_calls[0]["arguments"])
    assert args == {"value": 72, "unit": "fahrenheit"}


def test_streaming_no_tool_call():
    parser = Gemma4ToolParser()
    result = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="Hello",
        delta_text="Hello",
    )
    assert result == {"content": "Hello"}


def test_streaming_tool_call_buffering():
    parser = Gemma4ToolParser()
    # Mid tool call — should buffer
    result = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=f"<|tool_call>call:get_weather{{location:{Q}SF",
        delta_text=f"{Q}SF",
    )
    assert result is None  # buffering


def test_streaming_tool_call_complete():
    parser = Gemma4ToolParser()
    full = f"<|tool_call>call:get_weather{{location:{Q}SF{Q}}}<tool_call|>"
    result = parser.extract_tool_calls_streaming(
        previous_text=f"<|tool_call>call:get_weather{{location:{Q}SF{Q}}}",
        current_text=full,
        delta_text="<tool_call|>",
    )
    assert result is not None
    assert "tool_calls" in result
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"


def test_extract_hyphenated_tool_name():
    parser = Gemma4ToolParser()
    output = f"<|tool_call>call:web-search{{query:{Q}hello{Q}}}<tool_call|>"
    result = parser.extract_tool_calls(output)

    assert result.tools_called is True
    assert result.tool_calls[0]["name"] == "web-search"
    args = json.loads(result.tool_calls[0]["arguments"])
    assert args == {"query": "hello"}


def test_auto_parser_detects_gemma4():
    from vllm_mlx.tool_parsers.auto_tool_parser import AutoToolParser

    parser = AutoToolParser()
    output = f"<|tool_call>call:search{{query:{Q}hello world{Q}}}<tool_call|>"
    result = parser.extract_tool_calls(output)

    assert result.tools_called is True
    assert result.tool_calls[0]["name"] == "search"


def test_auto_parser_streaming_gemma4():
    from vllm_mlx.tool_parsers.auto_tool_parser import AutoToolParser

    parser = AutoToolParser()
    full = f"<|tool_call>call:get_weather{{location:{Q}SF{Q}}}<tool_call|>"

    # No marker yet — pass through
    r1 = parser.extract_tool_calls_streaming("", "Hello", "Hello")
    assert r1 == {"content": "Hello"}

    # Mid tool call — buffer
    r2 = parser.extract_tool_calls_streaming(
        "", f"<|tool_call>call:get_weather{{location:{Q}SF", f"{Q}SF"
    )
    assert r2 is None

    # End marker arrives — parse
    r3 = parser.extract_tool_calls_streaming(
        f"<|tool_call>call:get_weather{{location:{Q}SF{Q}}}",
        full,
        "<tool_call|>",
    )
    assert r3 is not None
    assert "tool_calls" in r3
    assert r3["tool_calls"][0]["function"]["name"] == "get_weather"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
