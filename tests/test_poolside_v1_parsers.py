# SPDX-License-Identifier: Apache-2.0

import json

from vllm_mlx.reasoning import get_parser as get_reasoning_parser
from vllm_mlx.tool_parsers import ToolParserManager


def _request():
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "line": {"type": "integer"},
                        },
                    },
                },
            }
        ]
    }


def test_poolside_reasoning_parser_handles_explicit_and_injected_thinking():
    parser = get_reasoning_parser("poolside_v1")()

    assert parser.extract_reasoning("<think>inspect</think>done") == (
        "inspect",
        "done",
    )
    assert parser.extract_reasoning("inspect</think>done") == ("inspect", "done")
    assert parser.extract_reasoning("plain final") == (None, "plain final")


def test_poolside_tool_parser_preserves_string_content_and_coerces_numbers():
    parser = ToolParserManager.get_tool_parser("poolside_v1")()
    output = (
        "I will update it."
        "<tool_call>write_file"
        "<arg_key>path</arg_key><arg_value>/tmp/a.py</arg_value>"
        "<arg_key>content</arg_key><arg_value>  x = 1\n</arg_value>"
        "<arg_key>line</arg_key><arg_value>7</arg_value>"
        "</tool_call>"
    )

    result = parser.extract_tool_calls(output, _request())

    assert result.tools_called is True
    assert result.content == "I will update it."
    assert result.tool_calls[0]["name"] == "write_file"
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "path": "/tmp/a.py",
        "content": "  x = 1\n",
        "line": 7,
    }


def test_poolside_tool_parser_handles_zero_args_and_multiple_calls():
    parser = ToolParserManager.get_tool_parser("poolside_v1")()
    output = (
        "<tool_call>status</tool_call>"
        "<tool_call>write_file"
        "<arg_key>path</arg_key><arg_value>a.py</arg_value>"
        "</tool_call>"
    )

    result = parser.extract_tool_calls(output)

    assert [call["name"] for call in result.tool_calls] == ["status", "write_file"]
    assert json.loads(result.tool_calls[0]["arguments"]) == {}


def test_poolside_tool_parser_suppresses_unclosed_or_unknown_markup():
    parser = ToolParserManager.get_tool_parser("poolside_v1")()

    truncated = parser.extract_tool_calls(
        "visible<tool_call>write_file<arg_key>path</arg_key>", _request()
    )
    unknown = parser.extract_tool_calls(
        "<tool_call>delete_everything</tool_call>", _request()
    )

    assert truncated.tools_called is False
    assert truncated.content == "visible"
    assert unknown.tools_called is False
    assert unknown.content is None


def test_poolside_tool_parser_streams_plain_text_and_closed_tool_call():
    parser = ToolParserManager.get_tool_parser("poolside_v1")()
    chunks = [
        "<tool_",
        "call>write_file",
        "<arg_key>path</arg_key>",
        "<arg_value>a.py</arg_value></tool_call>",
    ]
    accumulated = ""
    emitted = []
    for chunk in chunks:
        previous = accumulated
        accumulated += chunk
        delta = parser.extract_tool_calls_streaming(
            previous, accumulated, chunk, request=_request()
        )
        if delta:
            emitted.append(delta)

    calls = [delta["tool_calls"][0] for delta in emitted]
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads("".join(call["function"]["arguments"] for call in calls)) == {
        "path": "a.py"
    }


def test_poolside_tool_parser_streaming_uses_declared_tool_schema():
    parser = ToolParserManager.get_tool_parser("poolside_v1")()
    output = (
        "<tool_call>write_file"
        "<arg_key>content</arg_key><arg_value>  x = 1\n</arg_value>"
        "<arg_key>line</arg_key><arg_value>7</arg_value>"
        "</tool_call>"
    )

    result = parser.extract_tool_calls_streaming("", output, output, request=_request())

    assert result is not None
    arguments = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"content": "  x = 1\n", "line": 7}


def test_poolside_tool_parser_streaming_rejects_unknown_tool():
    parser = ToolParserManager.get_tool_parser("poolside_v1")()
    output = "<tool_call>delete_everything</tool_call>"

    assert (
        parser.extract_tool_calls_streaming("", output, output, request=_request())
        is None
    )


def test_poolside_tool_parser_streaming_emits_each_call_once():
    parser = ToolParserManager.get_tool_parser("poolside_v1")()
    first = (
        "<tool_call>write_file"
        "<arg_key>path</arg_key><arg_value>a.py</arg_value>"
        "</tool_call>"
    )
    second = (
        "<tool_call>write_file"
        "<arg_key>path</arg_key><arg_value>b.py</arg_value>"
        "</tool_call>"
    )

    first_delta = parser.extract_tool_calls_streaming(
        "", first, first, request=_request()
    )
    second_delta = parser.extract_tool_calls_streaming(
        first, first + second, second, request=_request()
    )

    assert first_delta is not None and second_delta is not None
    assert len(first_delta["tool_calls"]) == 1
    assert len(second_delta["tool_calls"]) == 1
    assert first_delta["tool_calls"][0]["index"] == 0
    assert second_delta["tool_calls"][0]["index"] == 1
    assert json.loads(first_delta["tool_calls"][0]["function"]["arguments"]) == {
        "path": "a.py"
    }
    assert json.loads(second_delta["tool_calls"][0]["function"]["arguments"]) == {
        "path": "b.py"
    }


def test_poolside_tool_parser_recovers_after_unknown_streamed_call():
    parser = ToolParserManager.get_tool_parser("poolside_v1")()
    unknown = "<tool_call>delete_everything</tool_call>"

    assert (
        parser.extract_tool_calls_streaming("", unknown, unknown, request=_request())
        is None
    )
    assert parser.extract_tool_calls_streaming(
        unknown, unknown + "after", "after", request=_request()
    ) == {"content": "after"}


def test_poolside_tool_parser_unclosed_markup_cannot_contaminate_next_request():
    parser_class = ToolParserManager.get_tool_parser("poolside_v1")
    malformed = parser_class()
    clean_request = parser_class()
    output = "<tool_call>unknown<arg_key>path</arg_key>"

    assert (
        malformed.extract_tool_calls_streaming("", output, output, request=_request())
        is None
    )
    assert clean_request.extract_tool_calls_streaming(
        "", "visible", "visible", request=_request()
    ) == {"content": "visible"}


def test_poolside_tool_parser_streams_long_string_argument_incrementally():
    parser = ToolParserManager.get_tool_parser("poolside_v1")()
    chunks = [
        "<tool_call>write_file<arg_key>content</arg_key><arg_value>abc",
        "def",
        "</arg_value></tool_call>",
    ]
    accumulated = ""
    fragments = []
    for chunk in chunks:
        previous = accumulated
        accumulated += chunk
        delta = parser.extract_tool_calls_streaming(
            previous, accumulated, chunk, request=_request()
        )
        if delta and delta.get("tool_calls"):
            fragments.extend(
                call["function"]["arguments"] for call in delta["tool_calls"]
            )

    assert json.loads("".join(fragments)) == {"content": "abcdef"}


def test_poolside_parser_fifty_fifty_valid_malformed_coercion():
    parser = ToolParserManager.get_tool_parser("poolside_v1")()
    accepted = 0
    rejected = 0

    for index in range(50):
        output = (
            "<tool_call>write_file"
            f"<arg_key>path</arg_key><arg_value>file-{index}.py</arg_value>"
            f"<arg_key>line</arg_key><arg_value>{index}</arg_value>"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, _request())
        accepted += int(result.tools_called)
        assert json.loads(result.tool_calls[0]["arguments"])["line"] == index

    for index in range(50):
        output = (
            f"prefix-{index}<tool_call>unknown"
            "<arg_key>path</arg_key><arg_value>unsafe</arg_value>"
        )
        result = parser.extract_tool_calls(output, _request())
        rejected += int(not result.tools_called and result.content == f"prefix-{index}")

    assert accepted == 50
    assert rejected == 50
