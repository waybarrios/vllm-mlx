# SPDX-License-Identifier: Apache-2.0
"""
Tests for structured output (JSON Schema) functionality.

Tests the JSON parsing, validation, and response_format handling.
"""

import json
import pytest
from vllm_mlx.api.tool_calling import (
    validate_json_schema,
    extract_json_from_text,
    parse_json_output,
    build_json_system_prompt,
)
from vllm_mlx.api.models import ResponseFormat, ResponseFormatJsonSchema


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
        result = _inject_json_instruction(original, "Return JSON only")

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
