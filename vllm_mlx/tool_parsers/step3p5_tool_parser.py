# SPDX-License-Identifier: Apache-2.0
"""Step3p5 XML tool call parser for vllm-mlx.

Step's chat template emits function calls as:

<tool_call>
<function=name>
<parameter=key>
value
</parameter>
</function>
</tool_call>

That protocol matches the existing XML/function parser shape, but Step gets a
separate registered parser so launch contracts do not inherit Qwen behavior by
name or implication.
"""

from __future__ import annotations

import json
from typing import Any

from .abstract_tool_parser import ExtractedToolCallInformation, ToolParserManager
from .qwen_tool_parser import QwenToolParser


@ToolParserManager.register_module(["step3p5", "step"])
class Step3p5ToolParser(QwenToolParser):
    """Parse Step XML tool calls and preserve native tool replay messages."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    extra_stop_tokens = ["</tool_call>"]

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        result = super().extract_tool_calls(model_output, request)
        if not result.tools_called:
            return result

        unique_calls = []
        previous_signature = None
        for call in result.tool_calls:
            signature = self._tool_call_signature(call)
            if signature == previous_signature:
                continue
            unique_calls.append(call)
            previous_signature = signature

        return ExtractedToolCallInformation(
            tools_called=bool(unique_calls),
            tool_calls=unique_calls,
            content=result.content,
        )

    @staticmethod
    def _tool_call_signature(call: dict[str, Any]) -> tuple[str, str]:
        raw_arguments = call.get("arguments", "")
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            arguments = raw_arguments
        normalized_arguments = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
        )
        return (str(call.get("name", "")), normalized_arguments)
