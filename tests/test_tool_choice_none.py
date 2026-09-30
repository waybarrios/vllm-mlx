"""Tests for tool_choice='none' handling."""


class TestToolChoiceNoneParserSuppression:
    """Verify tool call parsing is suppressed when tool_choice='none'."""

    def test_parse_tool_calls_skipped_when_tool_choice_none(self):
        """_parse_tool_calls_with_parser should return no tools when tool_choice='none'."""
        from vllm_mlx.api.models import ChatCompletionRequest
        from vllm_mlx.server import _parse_tool_calls_with_parser

        # Text that looks like a tool call
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "London"}}</tool_call>'
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
            tool_choice="none",
        )
        cleaned, tool_calls = _parse_tool_calls_with_parser(text, request)
        # With tool_choice="none", parser should be suppressed
        assert tool_calls is None
        assert cleaned == text  # text returned unchanged

    def test_parse_tool_calls_works_when_tool_choice_auto(self):
        """Tool parsing should work normally when tool_choice is not 'none'."""
        from vllm_mlx.api.models import ChatCompletionRequest
        from vllm_mlx.server import _parse_tool_calls_with_parser

        text = "Hello, how can I help?"
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
            tool_choice="auto",
        )
        cleaned, tool_calls = _parse_tool_calls_with_parser(text, request)
        # No tool markup in text, so no tools found — but parser was NOT skipped
        assert tool_calls is None

    def test_parse_tool_calls_works_when_tool_choice_absent(self):
        """Tool parsing should work when tool_choice is not set."""
        from vllm_mlx.api.models import ChatCompletionRequest
        from vllm_mlx.server import _parse_tool_calls_with_parser

        text = "Hello, how can I help?"
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
        )
        cleaned, tool_calls = _parse_tool_calls_with_parser(text, request)
        assert tool_calls is None

    def test_tool_markup_ignored_when_tool_choice_none(self):
        """Even Qwen bracket-style tool calls should be suppressed."""
        from vllm_mlx.api.models import ChatCompletionRequest
        from vllm_mlx.server import _parse_tool_calls_with_parser

        text = '[Calling tool: get_weather({"city": "London"})]'
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "weather?"}],
            tool_choice="none",
        )
        cleaned, tool_calls = _parse_tool_calls_with_parser(text, request)
        assert tool_calls is None
        assert cleaned == text


class TestNoToolsParserSuppression:
    """Configured parsers must not invent calls without declared tools."""

    def test_complete_parser_is_skipped_without_tools(self, monkeypatch):
        from vllm_mlx.api.models import ChatCompletionRequest
        from vllm_mlx.server import _parse_tool_calls_with_parser
        from vllm_mlx.tool_parsers import LlamaToolParser
        import vllm_mlx.server as server

        text = '{"name": "Alice", "parameters": {"age": 42}}'
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "Describe Alice"}],
        )
        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "llama")
        monkeypatch.setattr(server, "_tool_parser_instance", LlamaToolParser())

        cleaned, tool_calls = _parse_tool_calls_with_parser(text, request)

        assert cleaned == text
        assert tool_calls is None

    def test_streaming_parser_is_skipped_without_tools(self, monkeypatch):
        from vllm_mlx.api.models import ChatCompletionRequest
        from vllm_mlx.server import _get_streaming_tool_parser
        from vllm_mlx.tool_parsers import LlamaToolParser
        import vllm_mlx.server as server

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "Return JSON"}],
            stream=True,
        )
        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "llama")
        monkeypatch.setattr(server, "_tool_parser_instance", LlamaToolParser())

        assert _get_streaming_tool_parser(request) is None

    def test_bare_llama_gate_is_not_enabled_for_auto_parser(self):
        from vllm_mlx.server import _streaming_tool_markup_possible_after_delta
        from vllm_mlx.tool_parsers import AutoToolParser, LlamaToolParser

        assert _streaming_tool_markup_possible_after_delta("", "{", LlamaToolParser())
        assert not _streaming_tool_markup_possible_after_delta(
            "", "{", AutoToolParser()
        )

    def test_python_tag_gate_is_not_enabled_for_auto_parser(self):
        from vllm_mlx.server import _streaming_tool_markup_possible_after_delta
        from vllm_mlx.tool_parsers import AutoToolParser, LlamaToolParser

        text = '<|python_tag|>{"name": "read", "parameters": {}}'
        assert _streaming_tool_markup_possible_after_delta("", text, LlamaToolParser())
        assert not _streaming_tool_markup_possible_after_delta(
            "", text, AutoToolParser()
        )
