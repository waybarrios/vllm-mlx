# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``vllm_mlx.utils.harmony_render``.

Targets the harmony rendering path enabled when ``--tool-call-parser harmony``
is active (see :issue:`568`). The tests check the wire-level output for the
shapes a multi-turn assistant-tool/tool-result conversation should produce,
matching what GPT-OSS was trained on.

These tests run only when the optional ``openai-harmony`` package is
installed; skipped otherwise.
"""

from __future__ import annotations

import pytest

from vllm_mlx.utils.harmony_render import HAS_HARMONY, render_messages

pytestmark = pytest.mark.skipif(
    not HAS_HARMONY,
    reason="openai-harmony not installed; harmony rendering is optional",
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file in the sandbox",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the sandbox",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
]


class TestHarmonyRender:
    def test_single_turn_user_message(self):
        prompt = render_messages(
            [{"role": "user", "content": "Hello."}],
            tools=None,
        )
        assert "<|start|>system" in prompt
        assert "<|start|>user<|message|>Hello.<|end|>" in prompt
        assert prompt.endswith("<|start|>assistant")

    def test_developer_block_renders_tool_namespace(self):
        prompt = render_messages(
            [{"role": "user", "content": "List files."}],
            tools=TOOLS,
        )
        assert "<|start|>developer<|message|>" in prompt
        assert "namespace functions" in prompt
        # Both tool schemas should appear
        assert "type read_file = (_:" in prompt
        assert "type run_command = (_:" in prompt

    def test_section_order_system_developer_user(self):
        """System must precede developer must precede the first user turn."""
        prompt = render_messages(
            [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Find the bug."},
            ],
            tools=TOOLS,
        )
        sys_pos = prompt.index("<|start|>system")
        dev_pos = prompt.index("<|start|>developer")
        user_pos = prompt.index("<|start|>user")
        assert sys_pos < dev_pos < user_pos

    def test_assistant_tool_call_renders_commentary_channel(self):
        """Assistant ``tool_calls`` must render in the commentary channel
        addressed to ``functions.<name>`` (the format the model was trained
        on). The OLD text-flattening (``[Calling tool: …]``) must NOT appear.
        """
        prompt = render_messages(
            [
                {"role": "user", "content": "Find the bug in foo.py."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": '{"cmd": "cat foo.py"}',
                            },
                        }
                    ],
                },
                {"role": "user", "content": "Continue."},
            ],
            tools=TOOLS,
        )
        assert "to=functions.run_command" in prompt
        assert "<|channel|>commentary" in prompt
        assert "<|call|>" in prompt
        # Bracket-text fallback must NOT appear (that's the bug we're fixing).
        assert "[Calling tool:" not in prompt

    def test_tool_result_renders_with_function_name(self):
        """``role=tool`` messages need to come back addressed from
        ``functions.<name> to=assistant`` — the function name is resolved
        by tracing the most recent assistant ``tool_call_id``.
        """
        prompt = render_messages(
            [
                {"role": "user", "content": "Look at foo.py."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": '{"cmd": "cat foo.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "def foo():\n    return None",
                },
                {"role": "user", "content": "Continue."},
            ],
            tools=TOOLS,
        )
        assert "<|start|>functions.run_command to=assistant" in prompt
        # ``[Tool Result …]`` text fallback must not appear.
        assert "[Tool Result" not in prompt

    def test_assistant_thinking_renders_analysis_channel(self):
        """Prior ``thinking`` text on an assistant turn that has tool_calls
        must render in the analysis channel before the commentary call."""
        prompt = render_messages(
            [
                {"role": "user", "content": "Find the bug."},
                {
                    "role": "assistant",
                    "content": "",
                    "thinking": "I should read the file first.",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "foo.py"}',
                            },
                        }
                    ],
                },
            ],
            tools=TOOLS,
        )
        assert "<|channel|>analysis" in prompt
        assert "I should read the file first." in prompt
        # Analysis channel must precede the commentary tool call.
        analysis_pos = prompt.index("<|channel|>analysis")
        call_pos = prompt.index("to=functions.read_file")
        assert analysis_pos < call_pos

    def test_assistant_arguments_dict_is_serialized(self):
        """Some callers pass ``arguments`` as a dict already — it should be
        JSON-serialized into the commentary channel payload."""
        prompt = render_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "foo.py"},
                            },
                        }
                    ],
                },
            ],
            tools=TOOLS,
        )
        assert '{"path": "foo.py"}' in prompt

    def test_final_assistant_message_renders_final_channel(self):
        """A previous assistant turn with content and no tool_calls should
        appear in the final channel."""
        prompt = render_messages(
            [
                {"role": "user", "content": "Hi."},
                {"role": "assistant", "content": "Hello back!"},
                {"role": "user", "content": "And again."},
            ],
        )
        assert "<|channel|>final<|message|>Hello back!" in prompt

    def test_generation_prompt_is_appended(self):
        """The prompt must end with the bare ``<|start|>assistant`` marker so
        the model knows it's its turn to generate."""
        prompt = render_messages([{"role": "user", "content": "Hi."}])
        assert prompt.rstrip().endswith("<|start|>assistant")
