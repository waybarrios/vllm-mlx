# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7 tool call parser for vllm-mlx.

Handles GLM-4.7-Flash style tool calling format.
Based on vLLM's glm47_moe_tool_parser.py
"""

import json
import re
import uuid
from collections.abc import Sequence
from enum import Enum
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


class ParserState(Enum):
    IDLE = "idle"
    PARSING_NAME = "parsing_name"
    PARSING_ARGUMENTS = "parsing_arguments"
    PARSING_KEY = "parsing_key"
    WAITING_FOR_VALUE = "waiting_for_value"
    PARSING_VALUE = "parsing_value"


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["glm47", "glm4"])
class Glm47ToolParser(ToolParser):
    """
    Tool call parser for GLM-4.7 and GLM-4.7-Flash models.

    Supports GLM-4.7 tool call format:
    <tool_call>function_name
    <arg_key>param1</arg_key><arg_value>value1</arg_value>
    <arg_key>param2</arg_key><arg_value>value2</arg_value>
    </tool_call>

    Used when --enable-auto-tool-choice --tool-call-parser glm47 are set.
    """

    # Match entire tool call block
    TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    # Match function name and optional arguments
    # GLM47 format: <tool_call>func_name\n<arg_key>...</arg_key>...
    FUNC_DETAIL_PATTERN = re.compile(
        r"<tool_call>\s*([^\n<]+?)(?:\n|\s*)(<arg_key>.*?)?</tool_call>", re.DOTALL
    )

    # Match individual argument key-value pairs
    ARG_PATTERN = re.compile(
        r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
    )

    def __init__(self, tokenizer: Any | None = None):
        super().__init__(tokenizer)
        self.last_parsed_index = 0
        self.state = ParserState.IDLE
        self.current_tool_id = generate_tool_id()
        self.current_tool_name = ""
        self.current_arg_key = ""
        self.is_first_arg = True
        self.tool_call_index = 0

    def reset(self) -> None:
        """Reset parser state for a new request."""
        super().reset()
        self.last_parsed_index = 0
        self.state = ParserState.IDLE
        self.current_tool_id = generate_tool_id()
        self.current_tool_name = ""
        self.current_arg_key = ""
        self.is_first_arg = True
        self.tool_call_index = 0

    def _deserialize(self, value: str) -> Any:
        """Convert string value to appropriate Python type.

        Uses json.loads for type coercion, falls back to raw string.
        """
        value = value.strip()

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _get_tool_names(self, request: dict[str, Any] | None) -> set[str]:
        """Extract valid tool names from the request."""
        if not request or "tools" not in request:
            return set()
        return {
            t.get("function", {}).get("name", "")
            for t in request.get("tools", [])
            if isinstance(t, dict)
        }

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete GLM-4.7 model response.
        """
        tool_calls = []

        # Strip think tags using the base class method (handles both
        # full <think>...</think> and implicit ...</think> patterns)
        cleaned_text = self.strip_think_tags(model_output)

        # Get valid tool names for validation
        valid_names = self._get_tool_names(request)

        # Find all tool call blocks
        matches = self.FUNC_DETAIL_PATTERN.findall(cleaned_text)

        for match in matches:
            func_name = match[0].strip() if match[0] else ""
            args_section = match[1] if len(match) > 1 and match[1] else ""
            # Fix 2: Handle zero-argument tool calls - coalesce None to empty string
            if args_section is None:
                args_section = ""

            if not func_name:
                continue

            # Validate tool name against available tools if provided
            if valid_names and func_name not in valid_names:
                continue

            # Parse arguments
            arguments = {}
            if args_section:
                arg_matches = self.ARG_PATTERN.findall(args_section)
                for arg_key, arg_value in arg_matches:
                    key = arg_key.strip()
                    value = self._deserialize(arg_value)
                    if key:
                        arguments[key] = value

            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": func_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )

        # When tool calls are found, don't return reasoning text as content
        # GLM often outputs thinking/reasoning before tool calls without <think> tags
        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_text
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Extract tool calls from streaming GLM-4.7 model output using a state machine.
        """
        # 1. Skip thinking content (defensive)
        if "<think>" in current_text and "</think>" not in current_text:
            self.last_parsed_index = len(current_text)
            return None

        # 2. State Machine
        # We use a while loop to handle multiple transitions in one chunk
        while self.last_parsed_index < len(current_text):
            unparsed = current_text[self.last_parsed_index :]

            if self.state == ParserState.IDLE:
                start_tag = "<tool_call>"
                start_pos = current_text.find(start_tag, self.last_parsed_index)
                if start_pos != -1:
                    # Found start of tool call
                    self.state = ParserState.PARSING_NAME
                    self.last_parsed_index = start_pos + len(start_tag)
                    self.current_tool_id = generate_tool_id()
                    self.current_tool_name = ""
                    self.is_first_arg = True
                    continue  # Try next state
                else:
                    # Still IDLE, emit delta as content if no partial tag
                    # Check for partial <tool_call> at the very end
                    potential = "<tool_call>"
                    for i in range(len(potential) - 1, 0, -1):
                        if current_text.endswith(potential[:i]):
                            return None  # Buffer partial tag

                    # No tag, emit content
                    self.last_parsed_index = len(current_text)
                    return {"content": delta_text}

            if self.state == ParserState.PARSING_NAME:
                # End of name markers
                nl_pos = current_text.find("\n", self.last_parsed_index)
                ak_pos = current_text.find("<arg_key>", self.last_parsed_index)
                tc_end_pos = current_text.find("</tool_call>", self.last_parsed_index)

                markers = [m for m in [nl_pos, ak_pos, tc_end_pos] if m != -1]
                if not markers:
                    # Still parsing name
                    name_chunk = unparsed.strip()
                    if name_chunk:
                        self.current_tool_name += name_chunk
                    self.last_parsed_index = len(current_text)
                    return None

                # Found end of name
                early = min(markers)
                name_part = current_text[self.last_parsed_index : early].strip()
                self.current_tool_name += name_part
                self.last_parsed_index = early
                self.state = ParserState.PARSING_ARGUMENTS

                # Emit the first chunk of tool call (header)
                return {
                    "tool_calls": [
                        {
                            "index": self.tool_call_index,
                            "id": self.current_tool_id,
                            "type": "function",
                            "function": {
                                "name": self.current_tool_name,
                                "arguments": "{",
                            },
                        }
                    ]
                }

            if self.state == ParserState.PARSING_ARGUMENTS:
                # Look for <arg_key> or </tool_call>
                ak_pos = current_text.find("<arg_key>", self.last_parsed_index)
                tc_end_pos = current_text.find("</tool_call>", self.last_parsed_index)

                if ak_pos != -1 and (tc_end_pos == -1 or ak_pos < tc_end_pos):
                    # Found next argument
                    self.state = ParserState.PARSING_KEY
                    self.last_parsed_index = ak_pos + len("<arg_key>")
                    self.current_arg_key = ""
                    continue
                elif tc_end_pos != -1:
                    # Tool call finished!
                    self.state = ParserState.IDLE
                    self.last_parsed_index = tc_end_pos + len("</tool_call>")
                    self.tool_call_index += 1
                    return {
                        "tool_calls": [
                            {
                                "index": self.tool_call_index - 1,
                                "id": self.current_tool_id,
                                "function": {"arguments": "}"},
                            }
                        ]
                    }
                else:
                    # Wait for more
                    self.last_parsed_index = len(current_text)
                    return None

            if self.state == ParserState.PARSING_KEY:
                ak_end_tag = "</arg_key>"
                ak_end_pos = current_text.find(ak_end_tag, self.last_parsed_index)
                if ak_end_pos != -1:
                    # Key captured
                    self.current_arg_key += current_text[
                        self.last_parsed_index : ak_end_pos
                    ].strip()
                    self.last_parsed_index = ak_end_pos + len(ak_end_tag)
                    self.state = ParserState.WAITING_FOR_VALUE
                    continue  # Transition to WAITING_FOR_VALUE in same chunk
                else:
                    # Still capturing key
                    self.current_arg_key += unparsed.strip()
                    self.last_parsed_index = len(current_text)
                    return None

            if self.state == ParserState.WAITING_FOR_VALUE:
                av_tag = "<arg_value>"
                av_pos = current_text.find(av_tag, self.last_parsed_index)
                if av_pos != -1:
                    self.state = ParserState.PARSING_VALUE
                    self.last_parsed_index = av_pos + len(av_tag)

                    # Emit JSON prefix for this argument
                    prefix = "" if self.is_first_arg else ", "
                    self.is_first_arg = False
                    return {
                        "tool_calls": [
                            {
                                "index": self.tool_call_index,
                                "id": self.current_tool_id,
                                "function": {
                                    "arguments": f'{prefix}"{self.current_arg_key}": "'
                                },
                            }
                        ]
                    }
                else:
                    # Buffer if tail looks like start of <arg_value>
                    for i in range(len(av_tag) - 1, 0, -1):
                        if current_text.endswith(av_tag[:i]):
                            return None
                    # Otherwise keep waiting
                    self.last_parsed_index = len(current_text)
                    return None

            if self.state == ParserState.PARSING_VALUE:
                av_end_tag = "</arg_value>"
                av_end_pos = current_text.find(av_end_tag, self.last_parsed_index)
                if av_end_pos != -1:
                    # Value finished
                    val_chunk = current_text[self.last_parsed_index : av_end_pos]
                    self.last_parsed_index = av_end_pos + len(av_end_tag)
                    self.state = ParserState.PARSING_ARGUMENTS

                    return {
                        "tool_calls": [
                            {
                                "index": self.tool_call_index,
                                "id": self.current_tool_id,
                                "function": {"arguments": f'{val_chunk}"'},
                            }
                        ]
                    }
                else:
                    # Yield value chunk incrementally
                    val_chunk = unparsed
                    self.last_parsed_index = len(current_text)
                    return {
                        "tool_calls": [
                            {
                                "index": self.tool_call_index,
                                "id": self.current_tool_id,
                                "function": {"arguments": val_chunk},
                            }
                        ]
                    }

        return None

        # 2. State Machine Loop
        # We work on current_text[self.last_parsed_index:] to handle fragmentation
        unparsed = current_text[self.last_parsed_index :]
        if not unparsed:
            return None

        # Pattern to detect tool call start
        if self.state == ParserState.IDLE:
            # Look for <tool_call> tag
            start_tag = "<tool_call>"
            start_pos = current_text.find(start_tag, self.last_parsed_index)

            if start_pos != -1:
                # Transition to PARSING_NAME
                # Content before the tag should be emitted (if any)
                content_before = current_text[self.last_parsed_index : start_pos]
                self.state = ParserState.PARSING_NAME
                self.last_parsed_index = start_pos + len(start_tag)
                self.current_tool_id = generate_tool_id()
                self.current_tool_name = ""
                self.current_arguments = {}

                # If there was content before the tag, we should return it
                # and then process the rest in the next call or recurse?
                # For simplicity, we just advance the index and continue in this call.
                return self.extract_tool_calls_streaming(
                    previous_text, current_text, delta_text, request=request
                )
            else:
                # Still IDLE, emit delta as content
                # But we must be careful not to emit partial tags
                # If the tail of current_text looks like the start of <tool_call>, we buffer
                potential_start = "<tool_call"
                found_potential = False
                for i in range(len(potential_start), 0, -1):
                    if current_text.endswith(potential_start[:i]):
                        found_potential = True
                        break

                if found_potential:
                    # Buffer the potential tag
                    return None

                # No tag, emit content
                self.last_parsed_index = len(current_text)
                return {"content": delta_text}

        if self.state == ParserState.PARSING_NAME:
            # Look for function name (ends with newline or start of arg tag)
            # Or just wait until we see <arg_key> or </tool_call>
            end_name_pos = current_text.find("\n", self.last_parsed_index)
            arg_start_pos = current_text.find("<arg_key>", self.last_parsed_index)
            tc_end_pos = current_text.find("</tool_call>", self.last_parsed_index)

            # Find the earliest marker
            markers = [m for m in [end_name_pos, arg_start_pos, tc_end_pos] if m != -1]
            if not markers:
                # Still parsing name, just advance (but don't emit)
                # We can update self.current_tool_name incrementally if needed
                name_chunk = current_text[self.last_parsed_index :].strip()
                if name_chunk:
                    self.current_tool_name += name_chunk
                self.last_parsed_index = len(current_text)
                return None

            # Found a marker
            early_marker = min(markers)
            name_part = current_text[self.last_parsed_index : early_marker].strip()
            self.current_tool_name += name_part
            self.last_parsed_index = early_marker

            # Transition to PARSING_ARGUMENTS (or finish if </tool_call>)
            if early_marker == tc_end_pos:
                # Zero-argument tool call finished
                tool_calls = [
                    {
                        "index": self.tool_call_index,
                        "id": self.current_tool_id,
                        "type": "function",
                        "function": {
                            "name": self.current_tool_name,
                            "arguments": "{}",
                        },
                    }
                ]
                self.tool_call_index += 1
                self.state = ParserState.IDLE
                self.last_parsed_index = tc_end_pos + len("</tool_call>")
                return {"tool_calls": tool_calls}

            # Transition to PARSING_ARGUMENTS
            self.state = ParserState.PARSING_ARGUMENTS
            # Emit the tool call header (id and name)
            tool_calls = [
                {
                    "index": self.tool_call_index,
                    "id": self.current_tool_id,
                    "type": "function",
                    "function": {
                        "name": self.current_tool_name,
                        "arguments": "",  # Start arguments stream
                    },
                }
            ]
            # Don't increment tool_call_index yet, as we might send more deltas for same call
            return {"tool_calls": tool_calls}

        if self.state == ParserState.PARSING_ARGUMENTS:
            # Look for arguments or end tag
            tc_end_pos = current_text.find("</tool_call>", self.last_parsed_index)

            if tc_end_pos != -1:
                # Finished! Parse all arguments found in the whole text for this tool call
                # We extract the section between <tool_call> and </tool_call>
                # and use the existing extract_tool_calls logic to get clean JSON
                result = self.extract_tool_calls(
                    current_text[
                        current_text.rfind("<tool_call>", 0, tc_end_pos) : tc_end_pos
                        + 12
                    ],
                    request,
                )
                if result.tools_called and result.tool_calls:
                    tc = result.tool_calls[0]
                    tool_calls = [
                        {
                            "index": self.tool_call_index,
                            "id": self.current_tool_id,  # Use persistent ID
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                    ]
                    self.tool_call_index += 1
                    self.state = ParserState.IDLE
                    self.last_parsed_index = tc_end_pos + len("</tool_call>")
                    return {"tool_calls": tool_calls}

                # Fallback if parsing failed
                self.state = ParserState.IDLE
                self.last_parsed_index = tc_end_pos + len("</tool_call>")
                return None

            # Incremental arguments streaming:
            # We look for <arg_key>...<arg_value>...
            # But the request says: "Capture all new tokens appended beyond last_parsed_index
            # and yield them immediately within the function.arguments field"

            # If we see tags, we should probably hide them from the arguments string
            # and only yield what's between them.

            # Simple heuristic: if we are in PARSING_ARGUMENTS, any text not inside a tag is arguments
            # However, GLM format is XML-ish, so it's better to just buffer until </tool_call>
            # for valid JSON, UNLESS we want to stream raw XML (which the client might not like).

            # Given the strict requirement for incremental streaming:
            # We will yield the delta_text, but strip the XML tags
            clean_delta = delta_text.replace("<arg_key>", "").replace("</arg_key>", "")
            clean_delta = clean_delta.replace("<arg_value>", "").replace(
                "</arg_value>", ""
            )

            self.last_parsed_index = len(current_text)
            if clean_delta.strip():
                return {
                    "tool_calls": [
                        {
                            "index": self.tool_call_index,
                            "id": self.current_tool_id,
                            "function": {"arguments": clean_delta},
                        }
                    ]
                }
            return None

        return None
