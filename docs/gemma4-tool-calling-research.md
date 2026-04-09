# Gemma 4 Tool Calling — Research & Fix Plan

**Date:** 2026-04-08
**Problem:** Gemma 4 tool calls don't work with Claude Code (or any OpenAI-compatible client)
**Root cause:** No Gemma 4 tool parser exists in vllm-mlx. Tool calls come through as raw text in `message.content` instead of structured `tool_calls` field.

## The Problem

When Claude Code sends a request with `tools` defined and Gemma 4 generates a tool call, the response looks like this:

```
# What Claude Code receives (broken):
{
  "choices": [{
    "message": {
      "content": "<|tool_call>call:read_file{<|\"|>path<|\"|>: <|\"|>/some/file.py<|\"|>}<tool_call|>",
      "tool_calls": null   ← EMPTY, should have the parsed call
    }
  }]
}

# What Claude Code expects (working, like Qwen produces):
{
  "choices": [{
    "message": {
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "read_file",
          "arguments": "{\"path\": \"/some/file.py\"}"
        }
      }]
    }
  }]
}
```

## Why It Happens

### Gemma 4's Native Tool Format

Gemma 4 uses special tokens for tool calls (different from all other model families):

| Token | Purpose |
|-------|---------|
| `<\|tool_call>` | Start of tool call block |
| `<tool_call\|>` | End of tool call block |
| `<\|"\|>` | Escaped quote (replaces `"` in arguments) |
| `<\|tool_result>` | Start of tool result (for multi-turn) |
| `<tool_result\|>` | End of tool result |

The format inside the delimiters is:
```
call:function_name{<|"|>key<|"|>: <|"|>value<|"|>, <|"|>key2<|"|>: <|"|>value2<|"|>}
```

This is NOT JSON — it uses the special `<|"|>` token instead of actual quote characters.

### What Other Models Do

- **Qwen3.5**: Uses `<tool_call>` / `</tool_call>` with standard JSON inside. The `qwen_tool_parser.py` in vllm-mlx handles this.
- **Hermes/Llama**: Uses `<tool_call>` with JSON. Has `hermes_tool_parser.py`.
- **DeepSeek**: Uses `<|tool▁call▁begin|>` format. Has `deepseek_tool_parser.py`.

All of these convert their native format into OpenAI-compatible `tool_calls` structure. **Gemma 4 has no such parser.**

## Existing Tool Parsers in vllm-mlx

```
vllm_mlx/tool_parsers/
├── abstract_tool_parser.py    ← base class, defines interface
├── auto_tool_parser.py        ← auto-detect which parser to use
├── qwen_tool_parser.py        ← Qwen models
├── hermes_tool_parser.py      ← Hermes/Llama
├── deepseek_tool_parser.py    ← DeepSeek
├── llama_tool_parser.py       ← Llama-specific
├── mistral_tool_parser.py     ← Mistral
├── kimi_tool_parser.py        ← Kimi/Moonshot
├── glm47_tool_parser.py       ← GLM models
├── harmony_tool_parser.py     ← Harmony
├── granite_tool_parser.py     ← IBM Granite
├── xlam_tool_parser.py        ← xLAM
├── nemotron_tool_parser.py    ← Nemotron
└── functionary_tool_parser.py ← Functionary
```

## Upstream Status

| Project | Gemma 4 tool parser? | Status |
|---------|---------------------|--------|
| **mlx-lm** (GitHub main) | Yes — merged 2026-04-08 | Available but NOT in PyPI 0.31.2 yet |
| **vllm** (GPU version) | Yes — `--tool-call-parser gemma4` | Working |
| **vllm-mlx** (our fork) | **NO** | Needs to be added |
| **Ollama** | Partial — streaming issues reported | Broken in some cases |
| **llama.cpp** | Via grammar-based tool calling | Works with config |

### mlx-lm Fix (Reference Implementation)

The fix merged into mlx-lm's main branch adds:
1. A `gemma4.py` parser module that handles the `<|tool_call>...<tool_call|>` delimiters and `<|"|>` escaping
2. Auto-detection in `_infer_tool_parser()` checking for both opening and closing markers
3. JSON conversion from the `call:func{key:value}` format to standard `{"name": "func", "arguments": {"key": "value"}}`

### vllm Fix (Reference Implementation)

vllm (GPU) has a `Gemma4ToolParser` class. Known issues:
- [#38837](https://github.com/vllm-project/vllm/issues/38837): `__init__()` missing `tools` parameter — 400 error on tool calls
- [#38946](https://github.com/vllm-project/vllm/issues/38946): Streaming produces invalid JSON due to delimiter splitting
- [#39043](https://github.com/vllm-project/vllm/issues/39043): Claude Code specifically broken with Gemma 4 tool calling

## Fix Options

### Option A: Add `gemma4_tool_parser.py` to vllm-mlx (Recommended)

Create a new parser following the same pattern as `qwen_tool_parser.py`:

1. **Parse**: Detect `<|tool_call>` and `<tool_call|>` delimiters in model output
2. **Unescape**: Replace `<|"|>` with `"` to get valid JSON
3. **Extract**: Parse the `call:function_name{args}` format
4. **Convert**: Output standard OpenAI `tool_calls` structure
5. **Register**: Add to `auto_tool_parser.py` detection and CLI `--tool-call-parser gemma4`

Key parsing logic:
```python
import re
import json

def parse_gemma4_tool_call(raw_text: str) -> list[dict]:
    """Parse Gemma 4 native tool call format into OpenAI tool_calls."""
    calls = []
    # Find all tool call blocks
    pattern = r'<\|tool_call>(.*?)<tool_call\|>'
    for match in re.finditer(pattern, raw_text, re.DOTALL):
        block = match.group(1).strip()
        # Unescape quotes
        block = block.replace('<|"|>', '"')
        # Parse call:function_name{json_args}
        call_match = re.match(r'call:(\w+)\s*(\{.*\})', block, re.DOTALL)
        if call_match:
            func_name = call_match.group(1)
            args_str = call_match.group(2)
            calls.append({
                "id": f"call_{hash(block) & 0xFFFFFF:06x}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": args_str
                }
            })
    return calls
```

Streaming considerations:
- The delimiters can split across SSE chunks
- Need to buffer until `<tool_call|>` is seen before parsing
- The vllm GPU version had bugs here ([#38946](https://github.com/vllm-project/vllm/issues/38946)) — learn from their mistakes

### Option B: Proxy-Layer Translation in HANK Arena

Add translation in `hank_llm_arena/proxy.py` that intercepts Gemma responses and rewrites tool calls. This would mean the proxy is no longer pure byte passthrough for Gemma models.

**Pros:** Doesn't require vllm-mlx changes, works immediately
**Cons:** Breaks the raw passthrough design, adds latency, needs to buffer streaming responses to detect tool calls, model-specific logic in the proxy

### Option C: Update mlx-lm to latest + hope vllm-mlx picks it up

Install mlx-lm from GitHub main (has the fix). Risk: may break vllm-mlx compatibility.

## Recommendation

**Option A** — add the parser to vllm-mlx. It follows the established pattern (14 other parsers exist), is the right architectural layer, and fixes it for all clients (not just our proxy). Use the mlx-lm and vllm GPU implementations as reference.

## Testing Plan

1. Start Gemma 4 with `--tool-call-parser gemma4`
2. Send a request with `tools` defined (e.g., a file read tool)
3. Verify `tool_calls` field is populated in the response
4. Verify streaming tool calls work (no split-delimiter bugs)
5. Test with Claude Code: `ANTHROPIC_BASE_URL=http://localhost:6969 claude` — verify tool use works end-to-end

## References

- [mlx-lm #1096 — Gemma 4 tool_calls field stays empty](https://github.com/ml-explore/mlx-lm/issues/1096)
- [vllm #39043 — Gemma 4 + Claude Code tool calling problems](https://github.com/vllm-project/vllm/issues/39043)
- [vllm #38837 — Gemma4ToolParser init missing tools param](https://github.com/vllm-project/vllm/issues/38837)
- [vllm #38946 — Streaming invalid JSON from delimiter splitting](https://github.com/vllm-project/vllm/issues/38946)
- [Google — Gemma 4 function calling docs](https://ai.google.dev/gemma/docs/capabilities/function-calling)
- [llama.cpp Gemma 4 tool calling fix](https://gist.github.com/daniel-farina/87dc1c394b94e45bb700d27e9ea03193)
