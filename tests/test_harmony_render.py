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


class TestServerPathPreservesNativeForHarmony:
    """End-to-end regression: messages with ``tool_calls`` must survive the
    server's prep step without being flattened to bracket text, so the
    harmony renderer downstream sees structured calls.

    Without the ``use_harmony_rendering=True`` plumbing in
    ``_prepare_chat_messages``, ``extract_multimodal_content`` runs with
    ``preserve_native_format=False`` (because ``HarmonyToolParser`` keeps
    ``SUPPORTS_NATIVE_TOOL_FORMAT=False`` by default) and converts assistant
    ``tool_calls`` into ``[Calling tool: …]`` strings + tool messages into
    ``[Tool Result …]`` strings before the LLM path's render call. The
    rendered prompt then leaks that bracket text through into the
    final-channel slot, defeating the entire harmony rendering effort.

    These assertions fail loudly if that regression ever returns.
    """

    @pytest.fixture
    def harmony_engine(self):
        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False  # harmony parser keeps this False
            use_harmony_rendering = True  # set by _detect_harmony_rendering()

        return _Engine()

    def test_prep_path_preserves_tool_calls_for_harmony(self, harmony_engine):
        from vllm_mlx.server import _prepare_chat_messages

        request_messages = [
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
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "def foo():\n    return None",
            },
            {"role": "user", "content": "Continue."},
        ]
        messages, _images, _videos, _audios, _has_media = _prepare_chat_messages(
            harmony_engine, request_messages
        )

        # The bracket-text fallback must not have run.
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                assert "[Calling tool:" not in content
                assert "[Tool Result" not in content

        # Assistant tool_calls survived structurally.
        assistant = next(m for m in messages if m.get("role") == "assistant")
        assert assistant.get("tool_calls"), "tool_calls were dropped on prep"
        assert assistant["tool_calls"][0]["function"]["name"] == "run_command"

        # Tool message survived as role=tool (not flattened to role=user).
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        assert tool_msg.get("content") == "def foo():\n    return None"

    def test_rendered_prompt_after_prep_has_no_bracket_text(self, harmony_engine):
        """The whole point: end-to-end, the harmony renderer must produce
        commentary-channel calls — not the bracket-text leftovers from
        ``extract_multimodal_content``'s legacy flatten path."""
        from vllm_mlx.server import _prepare_chat_messages

        request_messages = [
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
        ]
        prepped, *_ = _prepare_chat_messages(harmony_engine, request_messages)
        prompt = render_messages(prepped, tools=TOOLS)

        assert "[Calling tool:" not in prompt
        assert "[Tool Result" not in prompt
        # Structural harmony shape did make it through.
        assert "<|channel|>commentary" in prompt
        assert "to=functions.run_command" in prompt
        assert "<|start|>functions.run_command to=assistant" in prompt

    def test_non_harmony_engine_falls_through_unchanged(self):
        """When ``use_harmony_rendering`` is False (default for all
        non-harmony parsers), the prep path must behave exactly as before
        this patch — i.e., honor ``preserve_native_tool_format`` and
        nothing else. Guards against accidentally flipping native
        preservation for unrelated parsers."""
        from vllm_mlx.server import _prepare_chat_messages

        class _NoHarmonyEngine:
            is_mllm = False
            preserve_native_tool_format = False
            # use_harmony_rendering deliberately absent — exercises the
            # default-False branch via getattr().

        engine = _NoHarmonyEngine()
        request_messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "fn", "arguments": "{}"},
                    }
                ],
            },
        ]
        messages, *_ = _prepare_chat_messages(engine, request_messages)
        # Without harmony rendering AND without preserve_native_tool_format,
        # the legacy bracket-text flatten still runs — that's the existing
        # behaviour this PR doesn't change.
        assistant = next(m for m in messages if m.get("role") == "assistant")
        assert "[Calling tool: fn" in (assistant.get("content") or "")
