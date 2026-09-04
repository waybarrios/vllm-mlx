# SPDX-License-Identifier: Apache-2.0
"""Golden prompts for the DeepSeek-V4 encoder.

Generated from the reference ``encoding_dsv4.py`` published with the model
weights. The reference is not vendored into this repository, so its output is
frozen here: any drift in our encoder shows up as a failing equality assertion
rather than as a subtly malformed prompt at inference time.

Regenerate only against the reference implementation, never by pasting in what
our own encoder currently produces.
"""

GOLDEN_PROMPTS = {
    "single_user_thinking": (
        '[{"role": "user", "content": "What is 2+2?"}]',
        {"thinking_mode": "thinking"},
        "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜><think>",
    ),
    "single_user_chat": (
        '[{"role": "user", "content": "What is 2+2?"}]',
        {"thinking_mode": "chat"},
        "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜></think>",
    ),
    "system_and_user": (
        '[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}]',
        {"thinking_mode": "thinking"},
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>",
    ),
    "effort_high": (
        '[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}]',
        {"thinking_mode": "thinking", "reasoning_effort": "high"},
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\nYou are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>",
    ),
    "effort_max": (
        '[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}]',
        {"thinking_mode": "thinking", "reasoning_effort": "max"},
        "<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\nYou are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>",
    ),
    "effort_low": (
        '[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}]',
        {"thinking_mode": "thinking", "reasoning_effort": "low"},
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>",
    ),
    "effort_ignored_in_chat": (
        '[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}]',
        {"thinking_mode": "chat", "reasoning_effort": "max"},
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜></think>",
    ),
    "multiturn_drop": (
        '[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": "2 plus 2 is 4."}, {"role": "user", "content": "And 3+3?"}]',
        {"thinking_mode": "thinking"},
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜></think>4<｜end▁of▁sentence｜><｜User｜>And 3+3?<｜Assistant｜><think>",
    ),
    "multiturn_keep": (
        '[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": "2 plus 2 is 4."}, {"role": "user", "content": "And 3+3?"}]',
        {"thinking_mode": "thinking", "drop_thinking": False},
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>2 plus 2 is 4.</think>4<｜end▁of▁sentence｜><｜User｜>And 3+3?<｜Assistant｜><think>",
    ),
    "no_bos": (
        '[{"role": "user", "content": "What is 2+2?"}]',
        {"thinking_mode": "thinking", "add_default_bos_token": False},
        "<｜User｜>What is 2+2?<｜Assistant｜><think>",
    ),
    "tools_declared": (
        '[{"role": "system", "content": "You are a helpful assistant.", "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}, {"role": "user", "content": "What is 2+2?"}]',
        {"thinking_mode": "thinking"},
        '<｜begin▁of▁sentence｜>You are a helpful assistant.\n\n## Tools\n\nYou have access to a set of tools to help answer the user\'s question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="$TOOL_NAME">\n<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name="$TOOL_NAME2">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>What is 2+2?<｜Assistant｜><think>',
    ),
    "tool_roundtrip": (
        '[{"role": "system", "content": "You are a helpful assistant.", "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "", "reasoning_content": "I need the weather.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Prague\\", \\"days\\": 3}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "{\\"temp\\": 21}"}]',
        {"thinking_mode": "thinking"},
        '<｜begin▁of▁sentence｜>You are a helpful assistant.\n\n## Tools\n\nYou have access to a set of tools to help answer the user\'s question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="$TOOL_NAME">\n<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name="$TOOL_NAME2">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>What is 2+2?<｜Assistant｜><think>I need the weather.</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="city" string="true">Prague</｜DSML｜parameter>\n<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>{"temp": 21}</tool_result><｜Assistant｜><think>',
    ),
    "two_tool_results_merge": (
        '[{"role": "system", "content": "You are a helpful assistant.", "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "", "reasoning_content": "I need the weather.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Prague\\", \\"days\\": 3}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "{\\"temp\\": 21}"}, {"role": "tool", "tool_call_id": "call_1", "content": "{\\"temp\\": 21}"}]',
        {"thinking_mode": "thinking"},
        '<｜begin▁of▁sentence｜>You are a helpful assistant.\n\n## Tools\n\nYou have access to a set of tools to help answer the user\'s question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="$TOOL_NAME">\n<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name="$TOOL_NAME2">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>What is 2+2?<｜Assistant｜><think>I need the weather.</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="city" string="true">Prague</｜DSML｜parameter>\n<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>{"temp": 21}</tool_result>\n\n<tool_result>{"temp": 21}</tool_result><｜Assistant｜><think>',
    ),
}
