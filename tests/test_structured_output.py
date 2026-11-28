# SPDX-License-Identifier: Apache-2.0
"""
Tests for structured output (JSON Schema) functionality.

Tests the JSON parsing, validation, and response_format handling.
"""

import json

import pytest
from vllm_mlx.api.models import ResponseFormat, ResponseFormatJsonSchema
from vllm_mlx.api.tool_calling import (
    InvalidResponseFormatOutput,
    apply_response_format_or_error,
    build_json_system_prompt,
    extract_json_from_text,
    parse_json_output,
    validate_json_schema,
)


class TestValidateJsonSchema:
    """Tests for validate_json_schema function."""

    def test_valid_object(self):
        """Test validation of a valid object against schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        data = {"name": "Alice", "age": 30}
        is_valid, error = validate_json_schema(data, schema)
        assert is_valid is True
        assert error is None

    def test_invalid_type(self):
        """Test validation fails for wrong type."""
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        data = {"age": "not an integer"}
        is_valid, error = validate_json_schema(data, schema)
        assert is_valid is False
        assert error is not None
        assert "integer" in error.lower() or "type" in error.lower()

    def test_missing_required(self):
        """Test validation fails for missing required field."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {}
        is_valid, error = validate_json_schema(data, schema)
        assert is_valid is False
        assert error is not None

    def test_array_validation(self):
        """Test validation of array types."""
        schema = {
            "type": "object",
            "properties": {"colors": {"type": "array", "items": {"type": "string"}}},
        }
        # Valid
        is_valid, _ = validate_json_schema({"colors": ["red", "blue"]}, schema)
        assert is_valid is True

        # Invalid - number in array
        is_valid, _ = validate_json_schema({"colors": ["red", 123]}, schema)
        assert is_valid is False


class TestExtractJsonFromText:
    """Tests for extract_json_from_text function."""

    def test_pure_json(self):
        """Test extraction from pure JSON string."""
        text = '{"name": "test", "value": 42}'
        result = extract_json_from_text(text)
        assert result == {"name": "test", "value": 42}

    def test_json_in_markdown(self):
        """Test extraction from markdown code block."""
        text = """Here is the result:
```json
{"name": "test", "value": 42}
```
"""
        result = extract_json_from_text(text)
        assert result == {"name": "test", "value": 42}

    def test_json_in_plain_code_block(self):
        """Test extraction from plain code block without json marker."""
        text = """Result:
```
{"items": [1, 2, 3]}
```
"""
        result = extract_json_from_text(text)
        assert result == {"items": [1, 2, 3]}

    def test_json_embedded_in_text(self):
        """Test extraction from JSON embedded in text."""
        text = 'The answer is: {"result": true} and that concludes the analysis.'
        result = extract_json_from_text(text)
        assert result == {"result": True}

    def test_array_extraction(self):
        """Test extraction of JSON arrays."""
        text = 'Colors: ["red", "green", "blue"]'
        result = extract_json_from_text(text)
        assert result == ["red", "green", "blue"]

    def test_no_json(self):
        """Test returns None when no JSON found."""
        text = "This is just plain text with no JSON."
        result = extract_json_from_text(text)
        assert result is None

    def test_invalid_json(self):
        """Test returns None for invalid JSON."""
        text = '{"broken": json, not valid}'
        result = extract_json_from_text(text)
        assert result is None

    def test_nested_json(self):
        """Test extraction of nested JSON."""
        text = '{"outer": {"inner": {"deep": "value"}}}'
        result = extract_json_from_text(text)
        assert result == {"outer": {"inner": {"deep": "value"}}}

    def test_chatty_preamble_with_json(self):
        """
        Test the classic Minimax / chain-of-thought failure mode:
        the model explains itself before emitting JSON.
        """
        text = (
            "Let me format this as JSON:\n" '```json\n{"name": "John", "age": 25}\n```'
        )
        result = extract_json_from_text(text)
        assert result == {"name": "John", "age": 25}

    def test_unterminated_markdown_fence(self):
        """
        Test truncation: model started ```json fence but ran out of
        tokens before closing it. Previously returned None.
        """
        text = 'Let me format this as JSON:\n```json\n{"name": "John", "age": 25}'
        result = extract_json_from_text(text)
        assert result == {"name": "John", "age": 25}

    def test_truncated_json_unclosed_object(self):
        """Test repair of JSON truncated mid-object (max_tokens hit)."""
        text = '{"name": "John", "age": 25, "city": "Prague"'
        result = extract_json_from_text(text)
        assert result == {"name": "John", "age": 25, "city": "Prague"}

    def test_truncated_json_unclosed_string(self):
        """Test repair of JSON truncated mid-string."""
        text = '{"name": "John", "city": "Pra'
        result = extract_json_from_text(text)
        assert result == {"name": "John", "city": "Pra"}

    def test_truncated_json_unclosed_nested(self):
        """Test repair of JSON truncated deep inside nested structures."""
        text = '{"user": {"name": "John", "items": [1, 2, 3'
        result = extract_json_from_text(text)
        assert result == {"user": {"name": "John", "items": [1, 2, 3]}}

    def test_truncated_json_dangling_key(self):
        """Test repair when truncation leaves a dangling key."""
        text = '{"name": "John", "age":'
        result = extract_json_from_text(text)
        # Dangling "age": should get repaired to null or dropped.
        assert result is not None
        assert result.get("name") == "John"

    def test_balanced_scan_prefers_valid_json(self):
        """
        Test that balanced-brace scanning picks the first balanced JSON,
        not a greedy region between first { and last }.
        """
        text = 'First: {"a": 1}. Garbage: {broken}'
        result = extract_json_from_text(text)
        assert result == {"a": 1}

    def test_json_with_escaped_braces_in_string(self):
        """Test balanced scanner respects strings containing braces."""
        text = '{"template": "Hello {user}!", "count": 5}'
        result = extract_json_from_text(text)
        assert result == {"template": "Hello {user}!", "count": 5}

    def test_json_with_escaped_quotes(self):
        """Test balanced scanner respects escaped quotes inside strings."""
        text = '{"text": "He said \\"hi\\"", "ok": true}'
        result = extract_json_from_text(text)
        assert result == {"text": 'He said "hi"', "ok": True}


class TestRawJsonToolCallHijackPrevention:
    """
    Regression tests for the MiniMax-M2 bug where ``response_format`` with a
    schema that contained ``"name"`` (e.g. person name) was hijacked into a
    fake ``function.name`` tool call by the raw-JSON tool-call fallback.
    """

    def test_response_format_not_hijacked_no_tools(self):
        """JSON data with 'name' field must not become a tool call."""
        from vllm_mlx.api.tool_calling import parse_tool_calls

        # Simulate Minimax emitting user-schema JSON after response_format.
        text = '{"name": "John", "age": 25}'
        # Request has NO tools — any parse_tool_calls output is a hijack.
        cleaned, tool_calls = parse_tool_calls(text, request={"tools": None})
        assert tool_calls is None, f"Hijacked JSON into tool_calls: {tool_calls}"

    def test_response_format_with_response_format_no_tools(self):
        """
        Even when the request has ``response_format`` set and no tools,
        a JSON object with 'name' must not become a tool call.
        """
        from vllm_mlx.api.tool_calling import parse_tool_calls

        text = '{"name": "Alice", "age": 30, "city": "Prague"}'
        cleaned, tool_calls = parse_tool_calls(
            text,
            request={
                "tools": None,
                "response_format": {"type": "json_object"},
            },
        )
        assert tool_calls is None

    def test_genuine_tool_call_still_detected(self):
        """
        Regression guard: a genuine tool call (``name`` + ``arguments``)
        MUST still be detected when the caller passed ``tools``.
        """
        from vllm_mlx.api.tool_calling import parse_tool_calls

        text = '{"name": "get_weather", "arguments": {"city": "Prague"}}'
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        cleaned, tool_calls = parse_tool_calls(text, request={"tools": tools})
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"

    def test_no_name_no_arguments_not_tool_call(self):
        """Plain JSON without 'name' stays as content."""
        from vllm_mlx.api.tool_calling import parse_tool_calls

        text = '{"result": 42, "status": "ok"}'
        cleaned, tool_calls = parse_tool_calls(
            text, request={"tools": [{"type": "function"}]}
        )
        assert tool_calls is None

    def test_name_only_not_tool_call(self):
        """
        ``{"name": "x"}`` without ``"arguments"`` must NOT be a tool call.
        This is the specific shape that hijacked MiniMax-M2 output.
        """
        from vllm_mlx.api.tool_calling import _parse_raw_json_tool_calls

        assert _parse_raw_json_tool_calls('{"name": "John"}') is None
        assert _parse_raw_json_tool_calls('{"name": "John", "age": 25}') is None

    def test_looks_like_tool_call_helper(self):
        """Direct coverage for the _looks_like_tool_call heuristic."""
        from vllm_mlx.api.tool_calling import _looks_like_tool_call

        # Genuine tool calls.
        assert _looks_like_tool_call({"name": "f", "arguments": {}})
        assert _looks_like_tool_call({"name": "f", "arguments": '{"a":1}'})
        assert _looks_like_tool_call(
            {"name": "get_weather", "arguments": {"city": "Prague"}}
        )
        # Not tool calls.
        assert not _looks_like_tool_call({"name": "John", "age": 25})
        assert not _looks_like_tool_call({"name": "", "arguments": {}})
        assert not _looks_like_tool_call({"arguments": {}})
        assert not _looks_like_tool_call({"name": 123, "arguments": {}})
        assert not _looks_like_tool_call({"name": "f", "arguments": 42})
        assert not _looks_like_tool_call("not a dict")
        assert not _looks_like_tool_call(None)


class TestParseJsonOutput:
    """Tests for parse_json_output function."""

    def test_no_response_format(self):
        """Test with no response_format returns original text."""
        text = "Hello, world!"
        cleaned, parsed, is_valid, error = parse_json_output(text, None)
        assert cleaned == text
        assert parsed is None
        assert is_valid is True
        assert error is None

    def test_text_format(self):
        """Test with type='text' returns original text."""
        text = "Hello, world!"
        response_format = {"type": "text"}
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert cleaned == text
        assert parsed is None
        assert is_valid is True

    def test_json_object_valid(self):
        """Test json_object mode extracts valid JSON."""
        text = '{"name": "test"}'
        response_format = {"type": "json_object"}
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"name": "test"}
        assert is_valid is True
        assert error is None

    def test_json_object_invalid(self):
        """Test json_object mode fails for non-JSON."""
        text = "This is not JSON"
        response_format = {"type": "json_object"}
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed is None
        assert is_valid is False
        assert "Failed to extract" in error

    def test_json_object_rejects_array_root(self):
        """OpenAI json_object requires an object at the root, not any JSON value."""
        text = '[{"ok": true}]'
        response_format = {"type": "json_object"}

        _cleaned, parsed, is_valid, error = parse_json_output(text, response_format)

        assert parsed == [{"ok": True}]
        assert is_valid is False
        assert "root" in error.lower()

    def test_json_schema_valid(self):
        """Test json_schema mode validates against schema."""
        text = '{"name": "Alice", "age": 30}'
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
        }
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"name": "Alice", "age": 30}
        assert is_valid is True
        assert error is None

    def test_json_schema_invalid(self):
        """Test json_schema mode fails validation for wrong data."""
        text = '{"name": 123}'  # name should be string
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"name": 123}
        assert is_valid is False
        assert "validation failed" in error.lower()

    def test_jobs_compact_schema_rejects_non_integer_scores(self):
        """Regression for Jobs work item 1167's exact compact score contract."""
        score = {"type": "integer", "minimum": 1, "maximum": 5}
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["p", "s", "sc", "i"],
            "properties": {
                "p": {"type": "boolean"},
                "s": {"type": "string", "maxLength": 240},
                "sc": {
                    "type": "array",
                    "minItems": 9,
                    "maxItems": 9,
                    "prefixItems": [score] * 9,
                },
                "i": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["a", "v", "c", "m", "e", "r"],
                        "properties": {
                            "a": {"type": "string", "enum": ["r", "c", "p"]},
                            "v": {"type": "string", "enum": ["b", "w", "i"]},
                            "c": {"type": "string", "maxLength": 48},
                            "m": {"type": "string", "maxLength": 280},
                            "e": {
                                "type": "array",
                                "maxItems": 4,
                                "items": {"type": "string", "maxLength": 80},
                            },
                            "r": {
                                "type": "array",
                                "maxItems": 4,
                                "items": {"type": "string", "maxLength": 80},
                            },
                        },
                    },
                },
            },
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "qa", "schema": schema, "strict": True},
        }
        raw = json.dumps(
            {
                "p": False,
                "s": "review",
                "sc": [4, "p", "s", "sc", "i", "p", "s", "sc", "i"],
                "i": [],
            }
        )

        _cleaned, parsed, is_valid, error = parse_json_output(raw, response_format)

        assert parsed is not None
        assert is_valid is False
        assert "not of type 'integer'" in error

    def test_response_format_model(self):
        """Test with ResponseFormat Pydantic model."""
        text = '{"result": true}'
        response_format = ResponseFormat(type="json_object")
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"result": True}
        assert is_valid is True

    def test_response_format_with_json_schema_model(self):
        """Test with ResponseFormat and ResponseFormatJsonSchema models."""
        text = '{"colors": ["red", "blue"]}'
        json_schema = ResponseFormatJsonSchema(
            name="colors",
            schema={
                "type": "object",
                "properties": {
                    "colors": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["colors"],
            },
        )
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema)
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"colors": ["red", "blue"]}
        assert is_valid is True


class TestServerResponseFormatBoundary:
    """Tests for the server boundary after response_format parsing."""

    def test_invalid_response_format_output_fails_before_success_response(self):
        with pytest.raises(InvalidResponseFormatOutput) as excinfo:
            apply_response_format_or_error(
                "I will reason first and then maybe return JSON.",
                {"type": "json_object"},
            )

        assert "Failed to extract valid JSON" in excinfo.value.message

    def test_valid_response_format_output_returns_canonical_json(self):
        content = apply_response_format_or_error(
            'prefix {"ok": true, "count": 2} suffix',
            {"type": "json_object"},
        )

        assert json.loads(content) == {"ok": True, "count": 2}


class TestBuildJsonSystemPrompt:
    """Tests for build_json_system_prompt function."""

    def test_no_format(self):
        """Test returns None for no format."""
        result = build_json_system_prompt(None)
        assert result is None

    def test_text_format(self):
        """Test returns None for text format."""
        result = build_json_system_prompt({"type": "text"})
        assert result is None

    def test_json_object(self):
        """Test prompt for json_object mode."""
        result = build_json_system_prompt({"type": "json_object"})
        assert result is not None
        assert "valid JSON" in result
        assert "only" in result.lower()

    def test_json_object_has_strict_rules(self):
        """
        Test that json_object prompt contains explicit no-markdown /
        no-preamble rules (added to prevent chatty model failures).
        """
        result = build_json_system_prompt({"type": "json_object"})
        assert result is not None
        assert "STRICT" in result
        assert "markdown" in result.lower()
        assert "preamble" in result.lower() or "Preamble" in result

    def test_json_schema(self):
        """Test prompt for json_schema mode."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "description": "A person object",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        result = build_json_system_prompt(response_format)
        assert result is not None
        assert "person" in result
        assert "A person object" in result
        assert "JSON Schema" in result

    def test_json_schema_has_strict_rules(self):
        """Test json_schema prompt also carries the strict output rules."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {"type": "object"},
            },
        }
        result = build_json_system_prompt(response_format)
        assert result is not None
        assert "STRICT" in result
        assert "markdown" in result.lower()

    def test_json_schema_model(self):
        """Test prompt with ResponseFormat model."""
        json_schema = ResponseFormatJsonSchema(
            name="output", description="Output format", schema={"type": "object"}
        )
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema)
        result = build_json_system_prompt(response_format)
        assert result is not None
        assert "output" in result


class TestInjectJsonInstruction:
    """Tests for _inject_json_instruction function in server."""

    def test_inject_new_system_message(self):
        """Test injecting instruction when no system message exists."""
        from vllm_mlx.server import _inject_json_instruction

        messages = [{"role": "user", "content": "Hello"}]
        result = _inject_json_instruction(messages, "Return JSON only")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Return JSON only" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_append_to_existing_system(self):
        """Test appending to existing system message."""
        from vllm_mlx.server import _inject_json_instruction

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = _inject_json_instruction(messages, "Return JSON only")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "Return JSON only" in result[0]["content"]

    def test_does_not_modify_original(self):
        """Test that original messages are not modified."""
        from vllm_mlx.server import _inject_json_instruction

        original = [{"role": "user", "content": "Hello"}]
        original_content = original[0]["content"]
        _inject_json_instruction(original, "Return JSON only")

        # Original should be unchanged
        assert len(original) == 1
        assert original[0]["content"] == original_content


# Integration test - run only if model available
@pytest.mark.skip(reason="Requires model loaded")
class TestStructuredOutputIntegration:
    """Integration tests for structured output with real model."""

    @pytest.fixture
    def client(self):
        """Create OpenAI client pointing to local server."""
        from openai import OpenAI

        return OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

    def test_json_object_mode(self, client):
        """Test json_object mode returns valid JSON."""
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "List 3 colors"}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        # Should be valid JSON
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_json_schema_mode(self, client):
        """Test json_schema mode returns valid structured data."""
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "List 3 colors"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "colors",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "colors": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["colors"],
                    },
                },
            },
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        assert "colors" in data
        assert isinstance(data["colors"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestResponseFormatNormalizationContract:
    """Guard the normalized local type and its existing runtime semantics."""

    @pytest.mark.parametrize("function", [parse_json_output, build_json_system_prompt])
    def test_normalized_local_reuses_declared_dictionary_alternative(self, function):
        # Dropping or narrowing this local contract recreates nested-schema
        # inference errors; runtime characterization alone cannot detect that.
        import ast
        import inspect

        node = ast.parse(inspect.getsource(function)).body[0]
        argument = next(arg for arg in node.args.args if arg.arg == "response_format")
        boundary = next(
            item
            for item in ast.walk(argument.annotation)
            if isinstance(item, ast.Subscript)
            and isinstance(item.value, ast.Name)
            and item.value.id in ("Dict", "dict")
        )
        initializers = [
            item
            for item in ast.walk(node)
            if isinstance(item, (ast.Assign, ast.AnnAssign))
            and isinstance(item.value, ast.Dict)
            and {key.value for key in item.value.keys if isinstance(key, ast.Constant)}
            == {"type", "json_schema"}
        ]
        assert len(initializers) == 1
        initializer = initializers[0]
        assert isinstance(
            initializer, ast.AnnAssign
        ), "Normalized local must explicitly reuse the declared dictionary alternative"
        assert ast.dump(initializer.annotation) == ast.dump(
            boundary
        ), "Normalized local must not narrow the declared dictionary alternative"
        assert isinstance(initializer.target, ast.Name)
        assert any(
            isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == initializer.target.id
                for target in item.targets
            )
            and isinstance(item.value, ast.Name)
            and item.value.id == argument.arg
            for item in ast.walk(node)
        ), "The same normalized local must retain the raw dictionary input"

    @pytest.mark.parametrize("format_type", [None, "text", "unknown", "json_object"])
    def test_model_and_raw_formats_preserve_unicode_and_prompt_modes(self, format_type):
        text = 'prefix {"name":"Zoë"} suffix'
        formats = (
            [None]
            if format_type is None
            else [ResponseFormat(type=format_type), {"type": format_type}]
        )
        for response_format in formats:
            if format_type == "json_object":
                assert parse_json_output(text, response_format) == (
                    text,
                    {"name": "Zoë"},
                    True,
                    None,
                )
                for ensure_ascii, expected in [
                    (False, '{"name": "Zoë"}'),
                    (True, '{"name": "Zo\\u00eb"}'),
                ]:
                    assert (
                        apply_response_format_or_error(
                            text, response_format, ensure_ascii=ensure_ascii
                        )
                        == expected
                    )
            else:
                assert parse_json_output(text, response_format) == (
                    text,
                    None,
                    True,
                    None,
                )
                assert apply_response_format_or_error(text, response_format) == text

            for thinking in (False, True):
                prompt = build_json_system_prompt(
                    response_format, thinking_model=thinking
                )
                if format_type == "json_object":
                    assert prompt.startswith(
                        "You must respond with a single valid JSON value only.\n\n"
                    )
                    assert ("You may reason first" in prompt) is thinking
                    assert ("Your first character MUST" in prompt) is not thinking
                    assert "no ``` fences" in prompt
                else:
                    assert prompt is None

    @pytest.mark.parametrize("strict", [None, False])
    @pytest.mark.parametrize("schema_key", ["schema", "schema_"])
    def test_schema_alias_references_and_raw_input_are_preserved(
        self, strict, schema_key, monkeypatch
    ):
        import copy

        from vllm_mlx.api import tool_calling

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        nested = ResponseFormatJsonSchema(
            name="person", description=None, strict=strict, **{schema_key: schema}
        )
        model = ResponseFormat(type="json_schema", json_schema=nested)
        accesses = []

        class RawFormat(dict):
            def get(self, key, default=None):
                accesses.append(key)
                return super().get(key, default)

        raw = RawFormat(
            type="json_schema",
            json_schema={
                "name": "person",
                "description": None,
                "strict": strict,
                "schema": schema,
            },
        )
        before = copy.deepcopy(raw)
        validated = []
        rendered = []
        validate = tool_calling.validate_json_schema
        dumps = json.dumps

        def validate_recording_reference(data, supplied_schema):
            validated.append(supplied_schema)
            return validate(data, supplied_schema)

        def dump_recording_reference(value, **kwargs):
            if kwargs.get("indent") == 2:
                rendered.append(value)
            return dumps(value, **kwargs)

        monkeypatch.setattr(
            tool_calling, "validate_json_schema", validate_recording_reference
        )
        monkeypatch.setattr(tool_calling.json, "dumps", dump_recording_reference)
        for response_format, original_schema in [
            (model, nested.schema_),
            (raw, schema),
        ]:
            assert parse_json_output('{"name":"Zoë"}', response_format) == (
                '{"name":"Zoë"}',
                {"name": "Zoë"},
                True,
                None,
            )
            assert validated[-1] is original_schema
            assert (
                apply_response_format_or_error('{"name":"Zoë"}', response_format)
                == '{"name": "Zoë"}'
            )
            assert validated[-1] is original_schema
            for thinking in (False, True):
                prompt = build_json_system_prompt(
                    response_format, thinking_model=thinking
                )
                assert "matching the 'person' schema." in prompt
                assert '"required": [\n    "name"\n  ]' in prompt
                assert ("You may reason first" in prompt) is thinking
                assert ("Your first character MUST" in prompt) is not thinking
                assert rendered[-1] is original_schema
        assert accesses == ["type", "json_schema"] * 4
        assert raw == before
        assert raw["json_schema"]["schema"] is schema
        assert nested.schema_ == schema == before["json_schema"]["schema"]
        assert nested.description is None
        assert nested.strict is strict

    @pytest.mark.parametrize("as_model", [False, True])
    @pytest.mark.parametrize(
        "format_type,text,error_fragment",
        [
            ("json_object", "[1,2]", "JSON object root must be an object"),
            ("json_schema", '{"user":{}}', "'name' is a required property"),
        ],
    )
    def test_model_and_raw_formats_keep_validation_errors(
        self, as_model, format_type, text, error_fragment
    ):
        response_format = {"type": format_type}
        if format_type == "json_schema":
            response_format["json_schema"] = {
                "name": "nested",
                "schema": {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        }
                    },
                    "required": ["user"],
                },
            }
        if as_model:
            response_format = ResponseFormat(**response_format)
        cleaned, parsed, valid, error = parse_json_output(text, response_format)
        assert cleaned == text
        assert parsed == json.loads(text)
        assert valid is False
        assert error_fragment in error
        with pytest.raises(InvalidResponseFormatOutput) as excinfo:
            apply_response_format_or_error(text, response_format)
        assert excinfo.value.message == error


class TestResponseFormatBoundaryContract:
    """Typed boundaries, with selected real server bodies and explicit CPU stubs.

    Engine/media/default resolution and native processor construction are stubbed;
    these tests do not execute endpoints or certify constrained decoding.
    """

    @staticmethod
    def _server_namespace(source=None):
        import ast
        import dataclasses
        import logging
        from pathlib import Path
        from typing import Any, Callable, TypedDict, overload

        from fastapi import HTTPException

        from vllm_mlx.api import tool_calling
        from vllm_mlx.api.models import ChatCompletionRequest

        if source is None:
            source = (Path(__file__).parents[1] / "vllm_mlx/server.py").read_text()
        names = {
            "ChatGenerationKwargs",
            "PreparedChatInvocation",
            "_prepare_json_logits_processor",
            "_prepare_chat_completion_invocation",
            "_prepare_anthropic_invocation",
            "_apply_response_format_or_raise",
            "_inject_json_instruction",
            "_strip_backslash_before_unicode",
            "_is_outlines_backend_selector",
            "_response_format_uses_outlines_backend",
            "_raise_if_outlines_response_format_requested",
        }
        nodes = [
            node
            for node in ast.parse(source).body
            if getattr(node, "name", None) in names
        ]
        assert {node.name for node in nodes} == names
        namespace = dict(vars(tool_calling)) | {
            "dataclass": dataclasses.dataclass,
            "Any": Any,
            "Callable": Callable,
            "TypedDict": TypedDict,
            "overload": overload,
            "BaseEngine": object,
            "ChatCompletionRequest": ChatCompletionRequest,
            "HTTPException": HTTPException,
            "logger": logging.getLogger(__name__),
            "_OUTLINES_BACKEND_UNAVAILABLE_DETAIL": "response_format backend 'outlines' is unavailable",
            "_reasoning_parser": None,
            "_enable_auto_tool_choice": False,
            "_tool_call_parser": None,
            "_default_thinking_token_budget": None,
            "_get_engine_tokenizer": lambda engine: engine,
            "_prepare_chat_messages": lambda engine, messages: (
                [message.model_dump(exclude_none=True) for message in messages],
                [],
                [],
                [],
                False,
            ),
            "_attach_logit_bias_processor": lambda *args: None,
            "_resolve_chat_template_kwargs": lambda *args, **kwargs: {},
            "get_parser_stop_tokens": lambda *args: None,
            "_attach_response_format_logits_processor": lambda kwargs, processor: processor,
        }
        for name in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
        ):
            namespace["_resolve_" + name] = lambda value: value
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "selected-server", "exec"),
            namespace,
        )
        return namespace

    @pytest.mark.parametrize(
        "site",
        [
            "apply_response_format_or_error",
            "PreparedChatInvocation",
            "_prepare_json_logits_processor",
            "_apply_response_format_or_raise",
        ],
    )
    def test_boundary_preserves_semantic_model_dictionary_nullable_domain(self, site):
        # Erasing any one site to object recreates lost-domain diagnostics.
        import ast
        import inspect
        from pathlib import Path
        from typing import Any, get_args, get_origin

        if site == "apply_response_format_or_error":
            node = ast.parse(inspect.getsource(apply_response_format_or_error)).body[0]
        else:
            tree = ast.parse(
                (Path(__file__).parents[1] / "vllm_mlx/server.py").read_text()
            )
            node = next(
                node for node in tree.body if getattr(node, "name", None) == site
            )
        if isinstance(node, ast.ClassDef):
            annotation = next(
                item.annotation
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and item.target.id == "response_format"
            )
        else:
            annotation = next(
                arg.annotation for arg in node.args.args if arg.arg == "response_format"
            )
        from vllm_mlx.api import tool_calling

        domain = eval(
            compile(ast.Expression(annotation), "boundary-annotation", "eval"),
            vars(tool_calling),
        )
        alternatives = {
            (get_origin(value), get_args(value)) if get_origin(value) else value
            for value in get_args(domain)
        }
        assert alternatives == {ResponseFormat, (dict, (str, Any)), type(None)}, site

    def test_required_nullable_constructor_and_intended_reflection(self):
        import dataclasses
        import inspect
        from typing import Any, get_type_hints

        namespace = self._server_namespace()
        constructor = namespace["PreparedChatInvocation"]
        expected = ResponseFormat | dict[str, Any] | None
        assert get_type_hints(constructor)["response_format"] == expected
        assert constructor.__init__.__annotations__["response_format"] == expected
        fields = dataclasses.fields(constructor)
        assert [field.name for field in fields] == [
            "messages",
            "chat_kwargs",
            "response_format",
            "json_logits_processor",
            "thinking_processor",
        ]
        for field in fields:
            assert field.default_factory is dataclasses.MISSING
            assert field.init and field.repr and field.compare
            assert field.hash is None and not field.kw_only
            assert field.default is (
                None if field.name == "thinking_processor" else dataclasses.MISSING
            )
        assert fields[2].type == expected
        signature = inspect.signature(constructor)
        assert list(signature.parameters) == [field.name for field in fields]
        assert all(
            parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in signature.parameters.values()
        )
        assert (
            signature.parameters["response_format"].default is inspect.Parameter.empty
        )
        with pytest.raises(TypeError):
            constructor(messages=[], chat_kwargs={}, json_logits_processor=None)
        assert constructor([], {}, None, None).response_format is None
        for function in (
            apply_response_format_or_error,
            namespace["_prepare_json_logits_processor"],
            namespace["_apply_response_format_or_raise"],
        ):
            assert get_type_hints(function)["response_format"] == expected
            assert (
                inspect.signature(function).parameters["response_format"].default
                is inspect.Parameter.empty
            )

    @pytest.mark.parametrize("adapted", [False, True])
    @pytest.mark.parametrize("with_format", [False, True])
    @pytest.mark.parametrize(
        "producer",
        ["_prepare_chat_completion_invocation", "_prepare_anthropic_invocation"],
    )
    def test_actual_requests_and_extracted_producers_keep_format_identity(
        self, adapted, with_format, producer
    ):
        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
        from vllm_mlx.api.anthropic_models import AnthropicRequest
        from vllm_mlx.api.models import ChatCompletionRequest

        response_format = ResponseFormat(type="json_object") if with_format else None
        values = {"model": "offline", "messages": [{"role": "user", "content": "json"}]}
        if adapted:
            request = anthropic_to_openai(
                AnthropicRequest(
                    **values,
                    max_tokens=32,
                    response_format=(
                        response_format.model_dump(by_alias=True)
                        if response_format
                        else None
                    ),
                )
            )
        else:
            request = ChatCompletionRequest(**values, response_format=response_format)
            assert request.response_format is response_format
        snapshot = request.model_dump(by_alias=True)
        namespace = self._server_namespace()
        supplied = []
        processor = object()

        def build(format_value, tokenizer):
            supplied.append(format_value)
            return processor

        namespace["build_json_logits_processor"] = build
        prepared = namespace[producer](object(), request, 32)
        assert prepared.response_format is request.response_format
        assert prepared.chat_kwargs["max_tokens"] == 32
        assert prepared.json_logits_processor is (processor if with_format else None)
        assert supplied == ([request.response_format] if with_format else [])
        if supplied:
            assert supplied[0] is request.response_format
        assert request.model_dump(by_alias=True) == snapshot

    def test_raw_schema_reference_and_http_output_are_preserved(self):
        import copy
        from fastapi import HTTPException

        namespace = self._server_namespace()
        schema = {"type": "object", "required": ["name"]}
        raw = {
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": schema},
        }
        snapshot = copy.deepcopy(raw)
        supplied = []
        namespace["build_json_logits_processor"] = (
            lambda format_value, tokenizer: supplied.append(format_value)
        )
        namespace["_prepare_json_logits_processor"](
            object(), [], raw, tools=None, tool_choice=None
        )
        assert supplied[0] is raw
        assert (
            namespace["PreparedChatInvocation"]([], {}, raw, None).response_format
            is raw
        )
        apply = namespace["_apply_response_format_or_raise"]
        assert apply('{"name":"Zoë"}', raw) == '{"name": "Zoë"}'
        assert (
            apply('{"name":"Zoë"}', raw, ensure_ascii=True) == '{"name": "Zo\\u00eb"}'
        )
        with pytest.raises(HTTPException) as excinfo:
            apply("not JSON", raw)
        assert excinfo.value.status_code == 422
        assert excinfo.value.detail == {
            "error": "invalid_response_format_output",
            "message": "Failed to extract valid JSON from output",
        }
        assert raw == snapshot and raw["json_schema"]["schema"] is schema
        assert apply("plain text", None) == "plain text"

    @pytest.mark.parametrize(
        "case,status,detail",
        [
            ("backend", 422, "response_format backend 'outlines' is unavailable"),
            ("tools", 400, "tools and response_format cannot be combined"),
            (
                "tokenizer",
                422,
                "response_format is unavailable: serving tokenizer not found",
            ),
            ("builder", 422, "response_format is unavailable: bounded failure"),
        ],
    )
    def test_preparation_retains_admission_and_construction_errors(
        self, case, status, detail
    ):
        from fastapi import HTTPException

        namespace = self._server_namespace()

        def fail_builder(*args):
            raise ValueError("bounded failure")

        namespace["build_json_logits_processor"] = fail_builder
        with pytest.raises(HTTPException) as excinfo:
            namespace["_prepare_json_logits_processor"](
                None if case == "tokenizer" else object(),
                [],
                {"type": "outlines" if case == "backend" else "json_object"},
                tools=[1] if case == "tools" else None,
                tool_choice=None,
            )
        assert (excinfo.value.status_code, excinfo.value.detail) == (status, detail)

    def test_backend_precedes_prompt_and_prompt_precedes_tools(self):
        from fastapi import HTTPException

        namespace = self._server_namespace()

        def fail_prompt(*args, **kwargs):
            raise ValueError("prompt first")

        namespace["build_json_system_prompt"] = fail_prompt
        prepare = namespace["_prepare_json_logits_processor"]
        with pytest.raises(HTTPException) as excinfo:
            prepare(object(), [], {"type": "outlines"}, tools=[1], tool_choice=None)
        assert excinfo.value.status_code == 422
        with pytest.raises(ValueError, match="prompt first"):
            prepare(object(), [], {"type": "json_object"}, tools=[1], tool_choice=None)


class TestJsonDecoderValueContract:
    """Recursive decoder typing and existing extraction/consumer behavior."""

    @pytest.mark.parametrize(
        "name", ["_repair_truncated_json", "extract_json_from_text"]
    )
    def test_return_contract_includes_recursive_json_values(self, name):
        from typing import Any, ForwardRef, get_args, get_origin, get_type_hints

        from vllm_mlx.api import tool_calling

        result = get_type_hints(getattr(tool_calling, name))["return"]
        members = get_args(result)
        assert {get_origin(member) or member for member in members} == {
            dict,
            list,
            str,
            int,
            float,
            bool,
            type(None),
        }
        assert (
            getattr(tool_calling, name).__annotations__["return"]
            is tool_calling._JSONValue
        )

        def check(member):
            assert member is not Any and member is not object
            if isinstance(member, ForwardRef):
                assert member.__forward_arg__ == "_JSONValue"
                return
            for arg in get_args(member):
                check(arg)

        check(result)
        containers = {
            get_origin(member): member for member in get_args(tool_calling._JSONValue)
        }
        assert get_args(containers[dict])[0] is str
        assert get_args(containers[dict])[1] == get_args(containers[list])[0]
        assert isinstance(get_args(containers[list])[0], ForwardRef)

    @pytest.mark.parametrize(
        "name,site",
        [("_repair_truncated_json", 0)]
        + [("extract_json_from_text", site) for site in range(4)],
    )
    def test_each_decoder_producer_has_typed_immediate_result(self, name, site):
        import ast
        import inspect
        import textwrap

        from vllm_mlx.api import tool_calling

        function = getattr(tool_calling, name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        calls = sorted(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
                and node.func.attr == "loads"
            ),
            key=lambda node: node.lineno,
        )
        assert len(calls) == (1 if name == "_repair_truncated_json" else 4)
        call = calls[site]
        assert len(call.args) == 1 and not call.keywords
        parent = next(
            node for node in ast.walk(tree) if call in list(ast.iter_child_nodes(node))
        )
        assert isinstance(parent, ast.AnnAssign), "Untyped stdlib decoder result"
        assert isinstance(parent.target, ast.Name)
        assert (
            eval(ast.unparse(parent.annotation), vars(tool_calling))
            == function.__annotations__["return"]
        )
        containing_try = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try) and parent in node.body
        )
        following = containing_try.body[containing_try.body.index(parent) + 1]
        assert isinstance(following, ast.Return)
        assert isinstance(following.value, ast.Name)
        assert following.value.id == parent.target.id

    @pytest.mark.parametrize("path", ["direct", "complete_fence", "open_fence"])
    @pytest.mark.parametrize(
        "text,want,kind",
        [
            (
                '{"a":[null,true,1,1.5,"Zoë",{"b":false}]}',
                {"a": [None, True, 1, 1.5, "Zoë", {"b": False}]},
                dict,
            ),
            ('[1,{"x":null}]', [1, {"x": None}], list),
            ('"Zoë"', "Zoë", str),
            ("42", 42, int),
            ("-2.5", -2.5, float),
            ("true", True, bool),
            ("false", False, bool),
            ("null", None, type(None)),
            ("NaN", "nan", float),
            ("Infinity", "inf", float),
            ("-Infinity", "-inf", float),
            ("1e400", "inf", float),
        ],
    )
    def test_real_decoder_root_domain(self, path, text, want, kind):
        import math

        if path != "direct":
            text = "before\n```json\n" + text
            if path == "complete_fence":
                text += "\n```"
        result = extract_json_from_text(text)
        assert type(result) is kind
        if want == "nan":
            assert math.isnan(result)
        elif want in ("inf", "-inf"):
            assert math.isinf(result)
            assert (result > 0) == (want == "inf")
        else:
            assert result == want
        if isinstance(result, dict):
            assert type(result["a"][1]) is bool
            assert type(result["a"][2]) is int

    @pytest.mark.parametrize(
        "fragment,want,attempts",
        [
            ('{"a":1', {"a": 1}, ['{"a":1}']),
            ("[1,", [1], ["[1,]", "[1]"]),
            (
                '{"a":1,"dangling":',
                {"a": 1},
                ['{"a":1,"dangling":}', '{"a":1,"dangling"}', '{"a":1}'],
            ),
            ("[1, tr", [1], ["[1, tr]", "[1]"]),
            ('"abc', "abc", ['"abc"']),
            ("[1, nope", None, ["[1, nope]"]),
        ],
    )
    def test_existing_repair_candidates_and_retry_order(
        self, monkeypatch, fragment, want, attempts
    ):
        from vllm_mlx.api.tool_calling import _repair_truncated_json

        loads = json.loads
        seen = []

        def record(text, *args, **kwargs):
            seen.append(text)
            assert not args and not kwargs
            return loads(text, *args, **kwargs)

        monkeypatch.setattr(json, "loads", record)
        result = _repair_truncated_json(fragment)
        assert result == want and type(result) is type(want)
        assert seen == attempts

    @pytest.mark.parametrize(
        "name,text,attempts",
        [
            ("_repair_truncated_json", '{"x":1', ['{"x":1}']),
            ("extract_json_from_text", '{"x":1}', ['{"x":1}']),
            (
                "extract_json_from_text",
                'before ```json\n{"x":1}\n```',
                ['before ```json\n{"x":1}\n```', '{"x":1}'],
            ),
            (
                "extract_json_from_text",
                'before ```json\n{"x":1}',
                ['before ```json\n{"x":1}', '{"x":1}'],
            ),
            (
                "extract_json_from_text",
                'before {"x":1} after',
                ['before {"x":1} after', '{"x":1}'],
            ),
        ],
    )
    def test_all_five_paths_return_original_decoder_result(
        self, monkeypatch, name, text, attempts
    ):
        from vllm_mlx.api import tool_calling

        loads = json.loads
        made = []
        seen = []

        def record(value, *args, **kwargs):
            seen.append(value)
            assert not args and not kwargs
            result = loads(value, *args, **kwargs)
            made.append(result)
            return result

        monkeypatch.setattr(json, "loads", record)
        result = getattr(tool_calling, name)(text)
        assert result == {"x": 1} and result is made[-1]
        assert seen == attempts

    @pytest.mark.parametrize("exception", [ValueError, TypeError, RecursionError])
    @pytest.mark.parametrize(
        "name,text",
        [
            ("_repair_truncated_json", '{"x":1'),
            ("extract_json_from_text", '{"x":1}'),
            ("extract_json_from_text", 'before ```json\n{"x":1}\n```'),
            ("extract_json_from_text", 'before ```json\n{"x":1}'),
            ("extract_json_from_text", 'before {"x":1} after'),
        ],
    )
    def test_non_decode_errors_propagate_at_each_producer(
        self, monkeypatch, exception, name, text
    ):
        from vllm_mlx.api import tool_calling

        loads = json.loads
        error = exception("decoder boundary")

        def decode(value, *args, **kwargs):
            if value == '{"x":1}':
                raise error
            return loads(value, *args, **kwargs)

        monkeypatch.setattr(json, "loads", decode)
        with pytest.raises(exception) as caught:
            getattr(tool_calling, name)(text)
        assert caught.value is error

    @pytest.mark.parametrize(
        "text,want",
        [("before [1,2] after", [1, 2]), ('before {"x":[1]} after', {"x": [1]})],
    )
    def test_balanced_scan_reachable_container_roots(self, text, want):
        result = extract_json_from_text(text)
        assert type(result) is type(want) and result == want

    def test_existing_object_before_array_scan_preference(self):
        assert extract_json_from_text('before [1] then {"later":2}') == {"later": 2}

    def test_existing_null_first_fence_stops_before_later_block(self, monkeypatch):
        loads = json.loads
        seen = []

        def record(text):
            seen.append(text)
            return loads(text)

        monkeypatch.setattr(json, "loads", record)
        text = '```json\nnull\n```\n```json\n{"later":1}\n```'
        assert extract_json_from_text(text) is None
        assert seen == [text, "null"]

    def test_existing_private_string_repair_is_not_bare_string_extraction(self):
        from vllm_mlx.api.tool_calling import _repair_truncated_json

        assert _repair_truncated_json('"abc') == "abc"
        assert extract_json_from_text('"abc') is None

    @pytest.mark.parametrize(
        "text,want", [("[1]", [1]), ('"abc"', "abc"), ("42", 42), ("true", True)]
    )
    def test_object_root_restriction_remains_in_parser(self, text, want):
        extracted = extract_json_from_text(text)
        assert type(extracted) is type(want) and extracted == want
        _, parsed, valid, error = parse_json_output(text, {"type": "json_object"})
        assert parsed == want and not valid
        assert error == "JSON object root must be an object"
        _, parsed, valid, error = parse_json_output(
            text, {"type": "json_schema", "json_schema": {"schema": {}}}
        )
        assert parsed == want and valid and error is None
