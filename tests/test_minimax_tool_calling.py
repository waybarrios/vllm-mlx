"""Tests for MiniMax tool call parsing."""

import json
import unittest

from vllm_mlx.api.tool_calling import parse_tool_calls


class TestMiniMaxToolCallParsing(unittest.TestCase):
    """Test parsing of MiniMax-style tool calls."""

    def test_single_tool_call(self):
        text = """<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Wanaka</parameter>
<parameter name="units">celsius</parameter>
</invoke>
</minimax:tool_call>"""

        cleaned, tool_calls = parse_tool_calls(text)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "get_weather")
        args = json.loads(tool_calls[0].function.arguments)
        self.assertEqual(args["city"], "Wanaka")
        self.assertEqual(args["units"], "celsius")
        self.assertEqual(cleaned, "")

    def test_tool_call_with_surrounding_text(self):
        text = """Let me check the weather for you.
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Wanaka</parameter>
</invoke>
</minimax:tool_call>"""

        cleaned, tool_calls = parse_tool_calls(text)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertIn("Let me check", cleaned)

    def test_multiple_tool_calls(self):
        text = """<minimax:tool_call>
<invoke name="search">
<parameter name="query">MiniMax M2.5</parameter>
</invoke>
</minimax:tool_call>
<minimax:tool_call>
<invoke name="read_file">
<parameter name="path">/tmp/test.txt</parameter>
</invoke>
</minimax:tool_call>"""

        cleaned, tool_calls = parse_tool_calls(text)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].function.name, "search")
        self.assertEqual(tool_calls[1].function.name, "read_file")

    def test_json_parameter_value(self):
        text = """<minimax:tool_call>
<invoke name="create_event">
<parameter name="title">Meeting</parameter>
<parameter name="attendees">["stuart", "frida"]</parameter>
</invoke>
</minimax:tool_call>"""

        cleaned, tool_calls = parse_tool_calls(text)
        self.assertIsNotNone(tool_calls)
        args = json.loads(tool_calls[0].function.arguments)
        self.assertEqual(args["title"], "Meeting")
        self.assertEqual(args["attendees"], ["stuart", "frida"])

    def test_numeric_parameter(self):
        text = """<minimax:tool_call>
<invoke name="set_temperature">
<parameter name="value">42</parameter>
</invoke>
</minimax:tool_call>"""

        cleaned, tool_calls = parse_tool_calls(text)
        args = json.loads(tool_calls[0].function.arguments)
        self.assertEqual(args["value"], 42)

    def test_no_parameters(self):
        text = """<minimax:tool_call>
<invoke name="get_time">
</invoke>
</minimax:tool_call>"""

        cleaned, tool_calls = parse_tool_calls(text)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0].function.name, "get_time")
        args = json.loads(tool_calls[0].function.arguments)
        self.assertEqual(args, {})

    def test_with_think_tags_preserved(self):
        text = """<think>
I should check the weather first.
</think>
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Wanaka</parameter>
</invoke>
</minimax:tool_call>"""

        cleaned, tool_calls = parse_tool_calls(text)
        self.assertIsNotNone(tool_calls)
        self.assertIn("<think>", cleaned)

    def test_no_minimax_tool_calls(self):
        text = "Just a regular message with no tool calls."
        cleaned, tool_calls = parse_tool_calls(text)
        self.assertIsNone(tool_calls)
        self.assertEqual(cleaned, text)

    def test_tool_call_id_format(self):
        text = """<minimax:tool_call>
<invoke name="test">
<parameter name="x">1</parameter>
</invoke>
</minimax:tool_call>"""

        _, tool_calls = parse_tool_calls(text)
        self.assertTrue(tool_calls[0].id.startswith("call_"))
        self.assertEqual(tool_calls[0].type, "function")


if __name__ == "__main__":
    unittest.main()
