# SPDX-License-Identifier: Apache-2.0
"""Server integration tests for Step3p5 tool parsing."""

import json

from vllm_mlx.api.models import ChatCompletionRequest


def test_server_step3p5_parser_collapses_repeated_xml_tool_call(monkeypatch):
    from vllm_mlx import server

    monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(server, "_tool_call_parser", "step3p5")
    monkeypatch.setattr(server, "_tool_parser_instance", None)

    repeated_call = (
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>Austin</parameter>\n"
        "<parameter=units>metric</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "weather?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "units": {"type": "string"},
                        },
                    },
                },
            }
        ],
        tool_choice="auto",
    )

    cleaned, tool_calls = server._parse_tool_calls_with_parser(
        repeated_call * 4,
        request,
    )

    assert cleaned == ""
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {
        "city": "Austin",
        "units": "metric",
    }
