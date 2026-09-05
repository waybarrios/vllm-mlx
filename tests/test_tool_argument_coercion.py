import json

import pytest

from vllm_mlx.server import (
    _coerce_tool_arguments,
    _finalize_streaming_tool_calls,
    _merge_streaming_tool_call_fragments,
)


def _tool_schema(properties):
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "parameters": {"type": "object", "properties": properties},
            },
        }
    ]


def test_coerces_json_encoded_array_and_integer_to_declared_types():
    arguments = json.dumps(
        {
            "argv": '["/usr/bin/printf", "transport-ok\\n"]',
            "timeout_seconds": "1",
        }
    )
    tools = _tool_schema(
        {
            "argv": {"type": "array", "items": {"type": "string"}},
            "timeout_seconds": {"type": "integer"},
        }
    )

    assert json.loads(_coerce_tool_arguments(arguments, "terminal", tools)) == {
        "argv": ["/usr/bin/printf", "transport-ok\n"],
        "timeout_seconds": 1,
    }


def test_preserves_invalid_value_when_conversion_is_not_lossless():
    arguments = json.dumps({"argv": "not-json", "timeout_seconds": "1.5"})
    tools = _tool_schema(
        {
            "argv": {"type": "array"},
            "timeout_seconds": {"type": "integer"},
        }
    )

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_preserves_value_that_already_matches_any_union_type():
    arguments = json.dumps({"timeout_seconds": "1"})
    tools = _tool_schema({"timeout_seconds": {"type": ["integer", "string"]}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {"answer": 42, "labels": ["café", "ok"]},
            '{"answer":42,"labels":["café","ok"]}',
        ),
        ([{"answer": 42}, "café"], '[{"answer":42},"café"]'),
    ],
)
def test_object_and_array_to_string_normalization_uses_compact_json(value, expected):
    arguments = json.dumps({"content": value})
    tools = _tool_schema({"content": {"type": "string"}})

    normalized = json.loads(_coerce_tool_arguments(arguments, "terminal", tools))
    assert normalized["content"] == expected


def test_preserves_existing_string_bytes():
    content = '{\n  "answer": 42,\n  "label": "café"\n}'
    arguments = json.dumps({"content": content})
    tools = _tool_schema({"content": {"type": "string"}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_coerces_json_boolean_without_treating_it_as_integer():
    arguments = json.dumps({"enabled": "true", "count": True})
    tools = _tool_schema({"enabled": {"type": "boolean"}, "count": {"type": "integer"}})

    assert json.loads(_coerce_tool_arguments(arguments, "terminal", tools)) == {
        "enabled": True,
        "count": True,
    }


@pytest.mark.parametrize("encoded", ["1e999", "-1e999", "NaN", "Infinity"])
def test_preserves_nonfinite_numeric_strings_byte_for_byte(encoded):
    arguments = json.dumps({"value": encoded})
    tools = _tool_schema({"value": {"type": "number"}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


@pytest.mark.parametrize(
    ("encoded", "declared_type", "expected"),
    [
        (
            '[[1, 2], {"nested": [3, 4]}]',
            "array",
            [[1, 2], {"nested": [3, 4]}],
        ),
        (
            '{"nested": {"values": [1, 2]}, "ok": true}',
            "object",
            {"nested": {"values": [1, 2]}, "ok": True},
        ),
    ],
)
def test_recovers_finite_nested_arrays_and_objects_without_bare_nonfinite(
    encoded, declared_type, expected
):
    arguments = json.dumps({"value": encoded}, ensure_ascii=False)
    tools = _tool_schema({"value": {"type": declared_type}})

    normalized = _coerce_tool_arguments(arguments, "terminal", tools)

    def reject_nonfinite(token):
        raise AssertionError(f"unexpected nonfinite JSON constant: {token}")

    assert json.loads(normalized, parse_constant=reject_nonfinite) == {
        "value": expected
    }


@pytest.mark.parametrize(
    "encoded",
    ['[1, {"nested": 1e999}]', '{"nested": [Infinity]}'],
)
def test_preserves_nested_nonfinite_json_strings_byte_for_byte(encoded):
    arguments = json.dumps({"value": encoded})
    declared_type = "object" if encoded.startswith("{") else "array"
    tools = _tool_schema({"value": {"type": declared_type}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_preserves_invalid_object_instead_of_stringifying_nonfinite_values():
    arguments = '{ "value": {"nested": [NaN]} }'
    tools = _tool_schema({"value": {"type": "string"}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_preserves_invalid_outer_payload_when_another_field_is_recoverable():
    arguments = '{ "count": "1", "unrelated": Infinity }'
    tools = _tool_schema({"count": {"type": "integer"}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_changed_serialization_never_emits_bare_nonfinite():
    arguments = json.dumps(
        {"argv": '["printf"]', "value": "1e999", "options": '{"ok": true}'},
        ensure_ascii=False,
    )
    tools = _tool_schema(
        {
            "argv": {"type": "array"},
            "value": {"type": "number"},
            "options": {"type": "object"},
        }
    )

    normalized = _coerce_tool_arguments(arguments, "terminal", tools)

    def reject_nonfinite(token):
        raise AssertionError(f"unexpected nonfinite JSON constant: {token}")

    assert json.loads(normalized, parse_constant=reject_nonfinite) == {
        "argv": ["printf"],
        "value": "1e999",
        "options": {"ok": True},
    }


def test_buffers_typed_stream_fragments_until_complete_json_is_available():
    tools = _tool_schema(
        {"argv": {"type": "array"}, "timeout_seconds": {"type": "integer"}}
    )
    calls = {}
    fragments = [
        {"index": 0, "id": "call_1", "function": {"name": "terminal"}},
        {"index": 0, "function": {"arguments": '{"argv": "[\\"printf\\"]", '}},
        {"index": 0, "function": {"arguments": '"timeout_seconds": "1"}'}},
    ]

    identities = []
    for fragment in fragments:
        identities.extend(_merge_streaming_tool_call_fragments(calls, [fragment]))
    assert identities == [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "terminal"},
        }
    ]

    finalized = _finalize_streaming_tool_calls(calls, tools)
    assert finalized == [
        {
            "index": 0,
            "function": {
                "arguments": json.dumps(
                    {"argv": ["printf"], "timeout_seconds": 1},
                    ensure_ascii=False,
                ),
            },
        }
    ]


def test_recovery_defaults_to_none():
    from vllm_mlx.api.models import ChatCompletionRequest

    request = ChatCompletionRequest(model="test", messages=[])
    assert request.tool_argument_recovery == "none"


@pytest.mark.parametrize(
    "value", [None, True, False, 0, 1, "true", "BUFFERED", "unknown", {}, []]
)
def test_invalid_recovery_selector_is_rejected(value):
    from pydantic import ValidationError
    from vllm_mlx.api.models import ChatCompletionRequest

    with pytest.raises(ValidationError) as exc:
        ChatCompletionRequest(model="test", messages=[], tool_argument_recovery=value)
    assert exc.value.errors()[0]["loc"] == ("tool_argument_recovery",)


def test_buffered_identity_is_not_repeated_for_interleaved_calls():
    calls = {}
    first = {"index": 0, "id": "a", "function": {"name": "terminal", "arguments": "{"}}
    second = {
        "index": 1,
        "id": "b",
        "function": {"name": "terminal", "arguments": "{}"},
    }
    assert len(_merge_streaming_tool_call_fragments(calls, [first, second])) == 2
    assert (
        _merge_streaming_tool_call_fragments(
            calls, [{**first, "function": {"name": "terminal", "arguments": "}"}}]
        )
        == []
    )
    assert _finalize_streaming_tool_calls(calls, None) == [
        {"index": 0, "function": {"arguments": "{}"}},
        {"index": 1, "function": {"arguments": "{}"}},
    ]


@pytest.mark.anyio
async def test_stream_buffers_fragments_before_schema_coercion(monkeypatch):
    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.server import stream_chat_completion
    import vllm_mlx.server as server

    fragments = [
        {"index": 0, "id": "call_1", "function": {"name": "terminal"}},
        {"index": 0, "function": {"arguments": '{"argv": '}},
        {"index": 0, "function": {"arguments": '"[\\"printf\\"]"'}},
        {"index": 0, "function": {"arguments": ', "timeout_seconds": "1"}'}},
    ]

    class FakeEngine:
        model_name = "fake-engine"

        async def stream_chat(self, messages, **kwargs):
            for index in range(len(fragments) + 1):
                finished = index == len(fragments)
                yield GenerationOutput(
                    text="",
                    new_text=f"fragment-{index}",
                    finished=finished,
                    finish_reason="stop" if finished else None,
                    prompt_tokens=3 if finished else 0,
                    completion_tokens=4 if finished else 0,
                )

    class FakeParser:
        def __init__(self):
            self.index = 0

        def extract_tool_calls_streaming(
            self, previous_text, current_text, delta_text, request=None
        ):
            if self.index == len(fragments):
                return None
            fragment = fragments[self.index]
            self.index += 1
            return {"tool_calls": [fragment]}

    monkeypatch.setattr(server, "_model_name", "served-model")
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_get_streaming_tool_parser", lambda *_: FakeParser())
    monkeypatch.setattr(
        server, "_streaming_tool_markup_possible_after_delta", lambda *_: True
    )

    request = ChatCompletionRequest(
        model="served-model",
        messages=[Message(role="user", content="run it")],
        tools=_tool_schema(
            {"argv": {"type": "array"}, "timeout_seconds": {"type": "integer"}}
        ),
        stream=True,
        tool_argument_recovery="buffered",
    )
    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            FakeEngine(), request.messages, request
        )
    ]
    payloads = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk != "data: [DONE]\n\n"
    ]
    tool_payloads = [
        payload
        for payload in payloads
        if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
    ]

    assert len(tool_payloads) == 2
    assert tool_payloads[0]["choices"][0]["delta"]["tool_calls"] == [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "terminal"},
        }
    ]
    call = tool_payloads[-1]["choices"][0]["delta"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {
        "argv": ["printf"],
        "timeout_seconds": 1,
    }
    assert tool_payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.anyio
@pytest.mark.parametrize("use_reasoning_parser", [False, True])
@pytest.mark.parametrize("recovery", [None, "none", "buffered"])
@pytest.mark.parametrize(
    "arguments,properties,normalized",
    [
        (
            '{"argv": "[\\"printf\\"]"}',
            {"argv": {"type": "array"}},
            {"argv": ["printf"]},
        ),
        (
            '{"value": "[1, {\\"nested\\": 1e999}]"}',
            {"value": {"type": "array"}},
            {"value": '[1, {"nested": 1e999}]'},
        ),
        ('{ "content": "café" }', {"content": {"type": "string"}}, {"content": "café"}),
        (
            '{"content": {"b": "café", "a": 1}}',
            {"content": {"type": "string"}},
            {"content": '{"b":"café","a":1}'},
        ),
    ],
)
async def test_stream_emits_sibling_content_and_reasoning_before_terminal_tool_call(
    monkeypatch, use_reasoning_parser, recovery, arguments, properties, normalized
):
    from types import SimpleNamespace

    from vllm_mlx.api.models import ChatCompletionRequest, Message
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.server import stream_chat_completion
    import vllm_mlx.server as server

    fragments = [
        {"index": 0, "id": "call_1", "function": {"name": "terminal"}},
        {"index": 0, "function": {"arguments": arguments[:5]}},
        {"index": 0, "function": {"arguments": arguments[5:]}},
    ]
    parser_results = [
        {"tool_calls": [fragments[0]], "content": "sibling content"},
        {"tool_calls": [fragments[1]]},
        {"tool_calls": [fragments[2]]},
        None,
    ]

    class FakeEngine:
        model_name = "fake-engine"

        async def stream_chat(self, messages, **kwargs):
            for index in range(len(parser_results)):
                yield GenerationOutput(
                    text="",
                    new_text="<tool_call>" if index == 0 else "delta",
                    finished=index == len(parser_results) - 1,
                    finish_reason="stop" if index == len(parser_results) - 1 else None,
                    prompt_tokens=3 if index == len(parser_results) - 1 else 0,
                    completion_tokens=4 if index == len(parser_results) - 1 else 0,
                )

    class FakeParser:
        def __init__(self):
            self.index = 0

        def extract_tool_calls_streaming(
            self, previous_text, current_text, delta_text, request=None
        ):
            result = parser_results[self.index]
            self.index += 1
            return result

    class FakeReasoningParser:
        def __init__(self, tokenizer=None):
            self.index = 0

        def reset_state(self, implicit_mode: bool = False):
            self.index = 0
            self._implicit_mode = implicit_mode

        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            if self.index == 0:
                self.index += 1
                return SimpleNamespace(
                    reasoning="sibling reasoning", content=delta_text
                )
            self.index += 1
            return SimpleNamespace(reasoning=None, content=delta_text)

    # Bind the current server's keyword call even on older no-argument routes.
    reset_probe = FakeReasoningParser()
    reset_probe.index = 7
    reset_probe.reset_state(implicit_mode=True)
    assert reset_probe.index == 0 and reset_probe._implicit_mode is True
    reset_probe.index = 7
    reset_probe.reset_state()
    assert reset_probe.index == 0 and reset_probe._implicit_mode is False

    monkeypatch.setattr(server, "_model_name", "served-model")
    monkeypatch.setattr(
        server,
        "_reasoning_parser",
        FakeReasoningParser() if use_reasoning_parser else None,
    )
    monkeypatch.setattr(server, "_reasoning_parser_name", None)
    monkeypatch.setattr(server, "_get_streaming_tool_parser", lambda *_: FakeParser())
    monkeypatch.setattr(
        server, "_streaming_tool_markup_possible_after_delta", lambda *_: True
    )

    request = ChatCompletionRequest(
        model="served-model",
        messages=[Message(role="user", content="run it")],
        tools=_tool_schema(properties),
        stream=True,
        **({"tool_argument_recovery": recovery} if recovery is not None else {}),
    )
    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            FakeEngine(), request.messages, request
        )
    ]
    payloads = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk != "data: [DONE]\n\n"
    ]
    choices = [payload["choices"][0] for payload in payloads if payload["choices"]]
    content_deltas = [
        choice["delta"].get("content")
        for choice in choices
        if choice["delta"].get("content")
    ]
    reasoning_deltas = [
        choice["delta"].get("reasoning_content")
        for choice in choices
        if choice["delta"].get("reasoning_content")
    ]
    tool_payloads = [
        payload
        for payload in payloads
        if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
    ]

    assert content_deltas == ["sibling content"]
    assert reasoning_deltas == (["sibling reasoning"] if use_reasoning_parser else [])
    assert len(tool_payloads) == (2 if recovery == "buffered" else 3)
    all_calls = [
        call
        for payload in tool_payloads
        for call in payload["choices"][0]["delta"]["tool_calls"]
    ]
    assert [call["id"] for call in all_calls if "id" in call] == ["call_1"]
    assert [
        call["function"]["name"]
        for call in all_calls
        if "name" in call.get("function", {})
    ] == ["terminal"]
    argument_payloads = [
        payload
        for payload in tool_payloads
        if any(
            "arguments" in call.get("function", {})
            for call in payload["choices"][0]["delta"]["tool_calls"]
        )
    ]
    tool_index = payloads.index(argument_payloads[0])
    sibling_index = next(
        index
        for index, payload in enumerate(payloads)
        if payload["choices"]
        and payload["choices"][0]["delta"].get("content") == "sibling content"
    )
    assert sibling_index < tool_index
    assert payloads.index(tool_payloads[0]) == sibling_index
    emitted_arguments = "".join(
        call.get("function", {}).get("arguments", "") for call in all_calls
    )
    if recovery == "buffered":
        assert len(argument_payloads) == 1
        assert json.loads(emitted_arguments) == normalized
        assert argument_payloads[0]["choices"][0]["finish_reason"] == "tool_calls"
        if json.loads(arguments) == normalized:
            assert emitted_arguments == arguments
    else:
        assert emitted_arguments == arguments
        assert all(
            payload["choices"][0]["finish_reason"] is None
            for payload in argument_payloads
        )
