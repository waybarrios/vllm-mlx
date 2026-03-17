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
