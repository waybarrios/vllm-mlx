# SPDX-License-Identifier: Apache-2.0
"""
Unified OpenAI-compatible API server for vllm-mlx.

This module provides a FastAPI server that exposes an OpenAI-compatible
API for LLM and MLLM (Multimodal Language Model) inference using MLX on Apple Silicon.

Supports two modes:
- Simple mode (default): Maximum throughput for single-user scenarios
- Batched mode: Continuous batching for multiple concurrent users

Features:
- Text-only LLM inference (mlx-lm)
- Multimodal MLLM inference with images and video (mlx-vlm)
- OpenAI-compatible chat/completions API
- Streaming responses
- MCP (Model Context Protocol) tool integration
- Tool calling (Qwen/Llama formats)

Usage:
    # Simple mode (maximum throughput)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Batched mode (for multiple concurrent users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

The server provides:
    - POST /v1/completions - Text completions
    - POST /v1/chat/completions - Chat completions (with multimodal support)
    - GET /v1/models - List available models
    - GET /health - Health check
    - GET /v1/mcp/tools - List MCP tools
    - GET /v1/mcp/servers - MCP server status
    - POST /v1/mcp/execute - Execute MCP tool
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import AsyncIterator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from starlette.routing import Match

# Import from new modular API
# Re-export for backwards compatibility with tests
from .api.anthropic_adapter import anthropic_to_openai
from .api.anthropic_models import (
    AnthropicRequest,
    AnthropicResponse,
    AnthropicResponseContentBlock,
    AnthropicUsage,
)
from .api.models import (
    AssistantMessage,  # noqa: F401
    ChatCompletionChoice,  # noqa: F401
    ChatCompletionChunk,  # noqa: F401
    ChatCompletionChunkChoice,  # noqa: F401
    ChatCompletionChunkDelta,  # noqa: F401
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,  # noqa: F401
    CompletionRequest,
    CompletionResponse,
    ContentPart,  # noqa: F401
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    ImageUrl,  # noqa: F401
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,  # noqa: F401
    MCPServersResponse,
    MCPToolInfo,  # noqa: F401
    MCPToolsResponse,
    Message,  # noqa: F401
    ModelInfo,  # noqa: F401
    ModelsResponse,
    ToolCall,
    Usage,  # noqa: F401
    VideoUrl,  # noqa: F401
)
from .api.responses_models import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallItem,
    ResponseFunctionCallOutputItem,
    ResponseFunctionTool,
    ResponseIncompleteDetails,
    ResponseInProgressEvent,
    ResponseMessageItem,
    ResponseObject,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseReasoningTextPart,
    ResponseTextContentPart,
    ResponsesRequest,
    ResponsesUsage,
)
from .api.tool_calling import (
    StreamingJsonFenceStripper,
    build_json_logits_processor,
    build_json_system_prompt,
    convert_tools_for_template,
    parse_json_output,
    parse_tool_calls,
)
from .api.utils import (
    SPECIAL_TOKENS_PATTERN,
    clean_output_text,
    extract_multimodal_content,
    is_mllm_model,  # noqa: F401
)
from .audio_limits import (
    DEFAULT_MAX_AUDIO_UPLOAD_BYTES,
    DEFAULT_MAX_AUDIO_UPLOAD_MB,
    DEFAULT_MAX_TTS_INPUT_CHARS,
    save_upload_with_limit,
    validate_tts_input_length,
)
from .engine import BaseEngine, BatchedEngine, GenerationOutput, SimpleEngine
from .endpoint_model_policies import (
    resolve_embedding_model_name,
    resolve_stt_model_name,
    resolve_tts_model_name,
)
from .metrics import metrics as _metrics
from .tool_parsers import ToolParserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
_engine: BaseEngine | None = None
_model_name: str | None = None
_model_path: str | None = (
    None  # Actual model path (for cache dir, not affected by --served-model-name)
)
_default_max_tokens: int = 32768
_default_timeout: float = 300.0  # Default request timeout in seconds (5 minutes)
_default_temperature: float | None = None  # Set via --default-temperature
_default_top_p: float | None = None  # Set via --default-top-p
_metrics_enabled = False
_max_audio_upload_bytes: int = DEFAULT_MAX_AUDIO_UPLOAD_BYTES
_max_tts_input_chars: int = DEFAULT_MAX_TTS_INPUT_CHARS

_FALLBACK_TEMPERATURE = 0.7
_FALLBACK_TOP_P = 0.9


def _resolve_temperature(request_value: float | None) -> float:
    """Resolve temperature: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if _default_temperature is not None:
        return _default_temperature
    return _FALLBACK_TEMPERATURE


def _resolve_top_p(request_value: float | None) -> float:
    """Resolve top_p: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if _default_top_p is not None:
        return _default_top_p
    return _FALLBACK_TOP_P


# Global MCP manager
_mcp_manager = None
_mcp_executor = None

# Global embedding engine (lazy loaded)
_embedding_engine = None
_embedding_model_locked: str | None = None  # Set when --embedding-model is used

# API key authentication
_api_key: str | None = None
_auth_warning_logged: bool = False

# Reasoning parser (for models like Qwen3, DeepSeek-R1)
_reasoning_parser = None  # ReasoningParser instance when enabled

# Tool calling configuration
_enable_auto_tool_choice: bool = False
_tool_call_parser: str | None = None  # Parser name: auto, mistral, qwen, llama, hermes
_tool_parser_instance = None  # Instantiated parser
_responses_store: OrderedDict[str, dict] = OrderedDict()
_RESPONSES_STORE_MAX_SIZE: int = 1000

# Pattern to strip leaked tool call markup from content output.
# Safety net: the tool parser should consume these, but if it doesn't
# (e.g. malformed JSON, stray closing tags), strip them before emitting.
_TOOL_MARKUP_PATTERN = re.compile(r"</?tool_call>|</?tool_call_reasoning>")
_STREAMING_TOOL_MARKERS = (
    "<tool_call>",
    "<|tool_call>",
    "<function=",
    "[Calling tool:",
    "[TOOL_CALLS]",
    "<minimax:tool_call>",
    '<invoke name="',
)


def _load_prefix_cache_from_disk() -> None:
    """Load prefix cache from disk during startup."""
    try:
        d = _get_cache_dir()
        logger.info(f"[lifespan] Loading prefix cache from {d}")
        loaded = _engine.load_cache_from_disk(d)
        if loaded > 0:
            logger.info(f"[lifespan] Loaded {loaded} prefix cache entries")
        else:
            logger.info("[lifespan] No prefix cache entries found on disk")
    except Exception as e:
        logger.warning(f"[lifespan] Failed to load cache from disk: {e}", exc_info=True)


def _save_prefix_cache_to_disk() -> None:
    """Save prefix cache to disk during shutdown."""
    try:
        d = _get_cache_dir()
        logger.info(f"[lifespan] Saving prefix cache to {d}")
        saved = _engine.save_cache_to_disk(d)
        if saved:
            logger.info(f"[lifespan] Saved prefix cache to {d}")
        else:
            logger.info("[lifespan] No cache to save")
    except Exception as e:
        logger.warning(f"[lifespan] Failed to save cache to disk: {e}", exc_info=True)


def _get_cache_dir() -> str:
    """Get cache persistence directory based on actual model path."""
    # Use _model_path (actual model path) not _model_name (which may be overridden
    # by --served-model-name). This ensures cache is shared regardless of served name.
    model_name = (
        _model_path if _model_path else (_model_name if _model_name else "default")
    )
    logger.info(
        f"[_get_cache_dir] _model_path={_model_path!r} type={type(_model_path)}"
    )
    # Sanitize model name for filesystem
    safe_name = str(model_name).replace("/", "--").replace("\\", "--")
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "vllm-mlx", "prefix_cache", safe_name
    )
    logger.info(f"[_get_cache_dir] cache_dir={cache_dir!r}")
    return cache_dir


async def lifespan(app: FastAPI):
    """FastAPI lifespan for startup/shutdown events."""
    global _engine, _mcp_manager

    # Startup: Start engine if loaded (needed for BatchedEngine in uvicorn's event loop)
    if _engine is not None and hasattr(_engine, "_loaded") and not _engine._loaded:
        await _engine.start()

    # Load persisted cache from disk (AFTER engine start — AsyncEngineCore must exist)
    if _engine is not None and hasattr(_engine, "load_cache_from_disk"):
        _load_prefix_cache_from_disk()

    # Initialize MCP if config provided
    mcp_config = os.environ.get("VLLM_MLX_MCP_CONFIG")
    if mcp_config:
        await init_mcp(mcp_config)

    yield

    # Shutdown: Save cache to disk BEFORE stopping engine
    if _engine is not None and hasattr(_engine, "save_cache_to_disk"):
        _save_prefix_cache_to_disk()

    # Shutdown: Close MCP connections and stop engine
    if _mcp_manager is not None:
        await _mcp_manager.stop()
        logger.info("MCP manager stopped")
    if _engine is not None:
        await _engine.stop()
        logger.info("Engine stopped")


app = FastAPI(
    title="vllm-mlx API",
    description="OpenAI-compatible API for MLX LLM/MLLM inference on Apple Silicon",
    version="0.2.1",
    lifespan=lifespan,
)

security = HTTPBearer(auto_error=False)


def _metrics_result_from_status(status_code: int) -> str:
    """Map HTTP-ish status codes to low-cardinality inference results."""
    if status_code == 499:
        return "client_closed"
    if status_code == 504:
        return "timeout"
    if status_code >= 500:
        return "error"
    return "success"


def _metrics_path_for_request(request: Request) -> str:
    """Prefer route templates over raw URLs to keep metrics cardinality bounded."""
    route = request.scope.get("route")
    if route is not None:
        path = getattr(route, "path", None)
        if path:
            return str(path)
    for candidate in app.router.routes:
        match, _ = candidate.matches(request.scope)
        if match in (Match.FULL, Match.PARTIAL):
            path = getattr(candidate, "path", None)
            if path:
                return str(path)
    return "__unmatched__"


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """Capture generic HTTP request metrics when enabled."""
    if not _metrics.enabled:
        return await call_next(request)

    method = request.method
    path = _metrics_path_for_request(request)
    if path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    _metrics.observe_http_start(method=method, path=path)
    try:
        response = await call_next(request)
    except Exception:
        _metrics.observe_http_finish(
            method=method,
            path=path,
            status_code=500,
            duration=time.perf_counter() - start_time,
        )
        raise

    _metrics.observe_http_finish(
        method=method,
        path=path,
        status_code=response.status_code,
        duration=time.perf_counter() - start_time,
    )
    return response


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, requests_per_minute: int = 60, enabled: bool = False):
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.window_size = 60.0  # 1 minute window
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed for client.

        Returns:
            (is_allowed, retry_after_seconds)
        """
        if not self.enabled:
            return True, 0

        current_time = time.time()
        window_start = current_time - self.window_size

        with self._lock:
            # Clean old requests outside window
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > window_start
            ]

            # Check rate limit
            if len(self._requests[client_id]) >= self.requests_per_minute:
                # Calculate retry-after
                oldest = min(self._requests[client_id])
                retry_after = int(oldest + self.window_size - current_time) + 1
                return False, max(1, retry_after)

            # Record this request
            self._requests[client_id].append(current_time)
            return True, 0


# Global rate limiter (disabled by default)
_rate_limiter = RateLimiter(requests_per_minute=60, enabled=False)


async def check_rate_limit(request: Request):
    """Rate limiting dependency."""
    # Use API key as client ID if available, otherwise use IP
    client_id = request.headers.get(
        "Authorization", request.client.host if request.client else "unknown"
    )

    allowed, retry_after = _rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if authentication is enabled."""
    global _auth_warning_logged

    if _api_key is None:
        # Log warning once about running without authentication
        if not _auth_warning_logged:
            logger.warning(
                "SECURITY WARNING: Server running without API key authentication. "
                "Anyone can access the API. Use --api-key to enable authentication."
            )
            _auth_warning_logged = True
        return True  # No auth required

    if credentials is None:
        raise HTTPException(status_code=401, detail="API key required")
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(credentials.credentials, _api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def get_engine() -> BaseEngine:
    """Get the loaded engine, raising error if not loaded."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _engine


def _coerce_tool_arguments(
    arguments_json: str, tool_name: str, tools: list[dict] | None
) -> str:
    """
    Coerce tool call arguments to match the tool schema.

    If a schema field expects "string" but the model produced an object/array,
    JSON-stringify the value. This fixes a common LLM failure mode where models
    output raw JSON objects instead of JSON strings for file content, etc.
    """
    if not tools:
        return arguments_json

    # Find the schema for this tool
    schema = None
    for tool in tools:
        if isinstance(tool, dict) and tool.get("function", {}).get("name") == tool_name:
            schema = tool["function"].get("parameters", {})
            break

    if not schema or "properties" not in schema:
        return arguments_json

    try:
        arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return arguments_json

    if not isinstance(arguments, dict):
        return arguments_json

    properties = schema.get("properties", {})
    changed = False

    for key, value in arguments.items():
        if key in properties:
            expected_type = properties[key].get("type")
            if expected_type == "string" and isinstance(value, (dict, list)):
                arguments[key] = json.dumps(value, ensure_ascii=False, indent=2)
                changed = True

    if changed:
        return json.dumps(arguments, ensure_ascii=False)

    return arguments_json


def _validate_model_name(request_model: str) -> None:
    """Validate that the request model name matches the served model."""
    if _model_name and request_model != _model_name:
        raise HTTPException(
            status_code=404,
            detail=f"The model `{request_model}` does not exist. "
            f"Available model: `{_model_name}`",
        )


def _parse_tool_calls_with_parser(
    output_text: str, request: ChatCompletionRequest | None = None
) -> tuple[str, list | None]:
    """
    Parse tool calls from model output using the configured parser.

    If --enable-auto-tool-choice is set with --tool-call-parser, uses the
    selected parser. Otherwise falls back to the generic parse_tool_calls.

    Args:
        output_text: The model output text
        request: The original request (for context)

    Returns:
        Tuple of (cleaned_text, tool_calls)
    """
    global _tool_parser_instance

    request_dict = request.model_dump() if request else None

    # tool_choice="none" means never return tool calls — skip all parsing
    if request is not None:
        tool_choice = getattr(request, "tool_choice", None)
        if tool_choice is None and request_dict:
            tool_choice = request_dict.get("tool_choice")
        if tool_choice == "none":
            return output_text, None

    # If auto tool choice is not enabled, use the generic parser
    if not _enable_auto_tool_choice or not _tool_call_parser:
        return parse_tool_calls(output_text, request_dict)

    # Initialize parser if needed
    if _tool_parser_instance is None:
        try:
            parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
            # Get tokenizer from engine if available
            tokenizer = None
            if _engine is not None and hasattr(_engine, "_tokenizer"):
                tokenizer = _engine._tokenizer
            _tool_parser_instance = parser_cls(tokenizer)
            logger.info(f"Initialized tool call parser: {_tool_call_parser}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize tool parser '{_tool_call_parser}': {e}"
            )
            logger.warning("Falling back to generic parser")
            return parse_tool_calls(output_text, request_dict)

    # Use the configured parser
    try:
        # Reset parser state between requests
        _tool_parser_instance.reset()
        result = _tool_parser_instance.extract_tool_calls(output_text, request_dict)
        if result.tools_called:
            tools = request_dict.get("tools") if request_dict else None
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=_coerce_tool_arguments(
                            tc["arguments"], tc["name"], tools
                        ),
                    ),
                )
                for tc in result.tool_calls
            ]
            return result.content or "", tool_calls
        else:
            # Fallback: specific parser didn't find tool calls,
            # try generic parser which handles more formats (e.g. Nemotron XML)
            return parse_tool_calls(output_text, request_dict)
    except Exception as e:
        logger.warning(f"Tool parser error: {e}")
        return parse_tool_calls(output_text, request_dict)


def _new_response_item_id(prefix: str) -> str:
    """Generate stable OpenAI-style item ids."""
    return f"{prefix}_{uuid.uuid4().hex}"


def _response_content_to_text(content) -> str:
    """Normalize Responses API content items into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    text_parts = []
    for part in content:
        if isinstance(part, dict):
            part_type = part.get("type")
            text = part.get("text", "")
        else:
            part_type = getattr(part, "type", None)
            text = getattr(part, "text", "")
        if part_type in {"text", "input_text", "output_text"}:
            text_parts.append(text)
    return "\n".join(part for part in text_parts if part)


def _responses_tools_to_chat_tools(
    tools: list[ResponseFunctionTool | dict],
) -> tuple[list[dict] | None, list[str]]:
    """Convert supported Responses tools and report unsupported tool types."""
    if not tools:
        return None, []

    supported: list[dict] = []
    unsupported: list[str] = []

    for tool in tools:
        if isinstance(tool, ResponseFunctionTool):
            tool_type = tool.type
            tool_name = tool.name
            tool_description = tool.description or ""
            tool_parameters = tool.parameters
        elif isinstance(tool, dict):
            tool_type = tool.get("type", "unknown")
            tool_name = tool.get("name", "")
            tool_description = tool.get("description", "")
            tool_parameters = tool.get("parameters", {})
        else:
            unsupported.append(type(tool).__name__)
            continue

        if tool_type == "function":
            supported.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": tool_parameters
                        or {"type": "object", "properties": {}},
                    },
                }
            )
        else:
            unsupported.append(tool_type)

    return supported or None, unsupported


def _responses_input_to_chat_messages(request: ResponsesRequest) -> list[dict]:
    """Convert Responses API input items into chat-completions-style messages."""
    messages: list[dict] = []

    if request.previous_response_id:
        previous = _responses_store.get(request.previous_response_id)
        if previous is None:
            raise HTTPException(
                status_code=404,
                detail=f"Previous response `{request.previous_response_id}` not found",
            )
        messages.extend(copy.deepcopy(previous["messages"]))

    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})

    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
        return messages

    for item in request.input:
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type == "message":
                role = item.get("role", "user")
                if role == "developer":
                    role = "system"
                messages.append(
                    {
                        "role": role,
                        "content": _response_content_to_text(item.get("content")),
                    }
                )
            elif item_type == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.get(
                                    "call_id", _new_response_item_id("call")
                                ),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", ""),
                                },
                            }
                        ],
                    }
                )
            elif item_type == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    }
                )
            elif item_type == "reasoning":
                parts = item.get("content", [])
                reasoning_text = "\n".join(
                    p.get("text", "") for p in parts if isinstance(p, dict)
                )
                if reasoning_text:
                    messages.append({"role": "assistant", "content": reasoning_text})
            else:
                logger.info(
                    "Skipping unsupported Responses input item type %r", item_type
                )
            continue

        if isinstance(item, ResponseMessageItem):
            role = item.role
            if role == "developer":
                role = "system"
            messages.append(
                {
                    "role": role,
                    "content": _response_content_to_text(item.content),
                }
            )
        elif isinstance(item, ResponseFunctionCallItem):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": item.arguments,
                            },
                        }
                    ],
                }
            )
        elif isinstance(item, ResponseFunctionCallOutputItem):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.call_id,
                    "content": item.output,
                }
            )
        elif isinstance(item, ResponseReasoningItem):
            reasoning_text = "\n".join(part.text for part in (item.content or []))
            if reasoning_text:
                messages.append({"role": "assistant", "content": reasoning_text})
        else:
            logger.info(
                "Skipping unsupported Responses input item type %r",
                getattr(item, "type", type(item).__name__),
            )

    return messages


def _responses_request_to_new_persisted_messages(
    request: ResponsesRequest,
) -> list[dict]:
    """Persist only the current request's replayable input items."""
    request_without_history = request.model_copy(
        update={"previous_response_id": None, "instructions": None},
        deep=True,
    )
    return _responses_input_to_chat_messages(request_without_history)


def _responses_request_to_persisted_messages(request: ResponsesRequest) -> list[dict]:
    """Persist replayable history for chained previous_response_id requests.

    Responses `instructions` are intentionally not replayed across
    `previous_response_id`, but replayable message items are.
    """
    messages: list[dict] = []
    if request.previous_response_id:
        previous = _responses_store.get(request.previous_response_id)
        if previous is None:
            raise HTTPException(
                status_code=404,
                detail=f"Previous response `{request.previous_response_id}` not found",
            )
        messages.extend(copy.deepcopy(previous["messages"]))
    messages.extend(_responses_request_to_new_persisted_messages(request))
    return messages


def _responses_request_to_chat_request(
    request: ResponsesRequest,
) -> ChatCompletionRequest:
    """Build a ChatCompletionRequest from a ResponsesRequest."""
    if request.text.format.type == "json_object":
        raise HTTPException(
            status_code=400,
            detail="Responses text.format.type='json_object' is not supported on this backend",
        )
    if request.reasoning is not None:
        logger.debug("Ignoring reasoning configuration (not supported on this backend)")

    tools, unsupported_tools = _responses_tools_to_chat_tools(request.tools)
    messages = _responses_input_to_chat_messages(request)
    if unsupported_tools:
        tool_list = ", ".join(sorted(set(unsupported_tools)))
        messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    "The following requested tool types are not available on this "
                    f"backend: {tool_list}. Do not call them."
                ),
            },
        )

    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
    merged_system_content = "\n\n".join(
        str(msg.get("content", "")).strip()
        for msg in system_messages
        if str(msg.get("content", "")).strip()
    )
    messages = (
        [{"role": "system", "content": merged_system_content}]
        if merged_system_content
        else []
    ) + non_system_messages

    return ChatCompletionRequest(
        model=request.model,
        messages=[Message(**msg) for msg in messages],
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        stream=False,
        tools=tools,
        tool_choice=request.tool_choice,
    )


def _build_responses_output_items(
    text: str | None,
    reasoning: str | None,
    tool_calls: list[ToolCall] | None,
) -> list[ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem]:
    """Convert parsed assistant output into Responses API output items."""
    output_items: list[
        ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem
    ] = []

    if reasoning:
        output_items.append(
            ResponseReasoningItem(
                id=_new_response_item_id("rs"),
                content=[ResponseReasoningTextPart(text=reasoning)],
            )
        )

    if text:
        output_items.append(
            ResponseMessageItem(
                id=_new_response_item_id("msg"),
                role="assistant",
                content=[ResponseTextContentPart(type="output_text", text=text)],
            )
        )

    for tool_call in tool_calls or []:
        output_items.append(
            ResponseFunctionCallItem(
                id=_new_response_item_id("fc"),
                call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
        )

    return output_items


def _response_output_items_to_chat_messages(output_items: list) -> list[dict]:
    """Persist assistant output in chat-completions form for previous_response_id."""
    assistant_text_parts: list[str] = []
    assistant_tool_calls: list[dict] = []

    for item in output_items:
        if isinstance(item, ResponseMessageItem):
            assistant_text_parts.append(_response_content_to_text(item.content))
        elif isinstance(item, ResponseFunctionCallItem):
            assistant_tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }
            )

    if not assistant_text_parts and not assistant_tool_calls:
        return []

    return [
        {
            "role": "assistant",
            "content": "".join(assistant_text_parts),
            "tool_calls": assistant_tool_calls or None,
        }
    ]


def _build_response_object(
    request: ResponsesRequest,
    output_items: list[
        ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem
    ],
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str | None,
    response_id: str | None = None,
) -> ResponseObject:
    """Build a full Responses API object."""
    response = ResponseObject(
        id=response_id or _new_response_item_id("resp"),
        model=_model_name or request.model,
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        metadata=request.metadata,
        output=output_items,
        parallel_tool_calls=request.parallel_tool_calls,
        previous_response_id=request.previous_response_id,
        text=request.text,
        tool_choice=request.tool_choice,
        tools=request.tools,
        top_p=_resolve_top_p(request.top_p),
        temperature=_resolve_temperature(request.temperature),
        truncation=request.truncation,
        user=request.user,
        store=request.store,
        usage=ResponsesUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    if finish_reason == "length":
        response.status = "incomplete"
        response.incomplete_details = ResponseIncompleteDetails(
            reason="max_output_tokens"
        )
    return response


def _prepare_responses_request(
    request: ResponsesRequest,
) -> tuple[BaseEngine, ChatCompletionRequest, list[dict], dict]:
    """Prepare a Responses request for execution on the chat engine."""
    _validate_model_name(request.model)
    engine = get_engine()
    chat_request = _responses_request_to_chat_request(request)

    if chat_request.messages:
        logger.info(
            f"[REQUEST] POST /v1/responses stream={request.stream} "
            f"model={request.model!r} items="
            f"{len(request.input) if isinstance(request.input, list) else 1} "
            f"tools={len(request.tools)}"
        )

    messages, images, videos = extract_multimodal_content(
        chat_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )

    chat_kwargs = {
        "max_tokens": chat_request.max_tokens or _default_max_tokens,
        "temperature": _resolve_temperature(chat_request.temperature),
        "top_p": _resolve_top_p(chat_request.top_p),
    }
    if request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(chat_request.tools)
    if images:
        chat_kwargs["images"] = images
    if videos:
        chat_kwargs["videos"] = videos

    return engine, chat_request, messages, chat_kwargs


async def _run_responses_request(
    request: ResponsesRequest,
    raw_request: Request,
) -> tuple[ResponseObject | None, list[dict]]:
    """Execute a Responses API request against the backend chat engine."""
    engine, chat_request, messages, chat_kwargs = _prepare_responses_request(request)

    timeout = _default_timeout
    output = await _wait_with_disconnect(
        engine.chat(messages=messages, **chat_kwargs),
        raw_request,
        timeout=timeout,
    )
    if output is None:
        return None, []

    cleaned_text, tool_calls = _parse_tool_calls_with_parser(output.text, chat_request)
    reasoning_text = None
    if _reasoning_parser and not tool_calls:
        reasoning_text, cleaned_text = _reasoning_parser.extract_reasoning(
            cleaned_text or output.text
        )

    output_items = _build_responses_output_items(
        clean_output_text(cleaned_text) if cleaned_text else None,
        reasoning_text,
        tool_calls,
    )
    response_object = _build_response_object(
        request=request,
        output_items=output_items,
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
        finish_reason=output.finish_reason,
    )

    persisted_messages = _responses_request_to_persisted_messages(request)
    persisted_messages.extend(_response_output_items_to_chat_messages(output_items))
    if request.store:
        _responses_store[response_object.id] = {
            "messages": copy.deepcopy(persisted_messages),
            "response": response_object.model_copy(deep=True),
        }
        while len(_responses_store) > _RESPONSES_STORE_MAX_SIZE:
            _responses_store.popitem(last=False)

    return response_object, persisted_messages


async def _stream_responses_request(request: ResponsesRequest) -> AsyncIterator[str]:
    """Execute a Responses API request and stream SSE events incrementally."""
    engine, chat_request, messages, chat_kwargs = _prepare_responses_request(request)

    response_id = _new_response_item_id("resp")
    sequence = 1
    base_response = _build_response_object(
        request=request,
        output_items=[],
        prompt_tokens=0,
        completion_tokens=0,
        finish_reason=None,
        response_id=response_id,
    )
    base_response.status = "in_progress"
    base_response.usage = None

    yield _responses_sse_event(
        "response.created",
        ResponseCreatedEvent(sequence_number=sequence, response=base_response),
    )
    sequence += 1
    yield _responses_sse_event(
        "response.in_progress",
        ResponseInProgressEvent(sequence_number=sequence, response=base_response),
    )
    sequence += 1

    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = None
    last_output = None
    raw_accumulated_text = ""
    accumulated_text = ""
    accumulated_reasoning = ""

    text_item_id: str | None = None
    text_output_index: int | None = None
    reasoning_item_id: str | None = None
    reasoning_output_index: int | None = None
    next_output_index = 0

    def _start_text_item() -> list[str]:
        nonlocal text_item_id, text_output_index, next_output_index, sequence
        events: list[str] = []
        if text_item_id is None:
            text_item_id = _new_response_item_id("msg")
            text_output_index = next_output_index
            next_output_index += 1
            events.append(
                _responses_sse_event(
                    "response.output_item.added",
                    ResponseOutputItemAddedEvent(
                        sequence_number=sequence,
                        output_index=text_output_index,
                        item=ResponseMessageItem(
                            id=text_item_id,
                            role="assistant",
                            status="in_progress",
                            content=[],
                        ),
                    ),
                )
            )
            sequence += 1
            events.append(
                _responses_sse_event(
                    "response.content_part.added",
                    ResponseContentPartAddedEvent(
                        sequence_number=sequence,
                        item_id=text_item_id,
                        output_index=text_output_index,
                        content_index=0,
                        part=ResponseTextContentPart(type="output_text", text=""),
                    ),
                )
            )
            sequence += 1
        return events

    def _start_reasoning_item() -> list[str]:
        nonlocal reasoning_item_id, reasoning_output_index, next_output_index, sequence
        events: list[str] = []
        if reasoning_item_id is None:
            reasoning_item_id = _new_response_item_id("rs")
            reasoning_output_index = next_output_index
            next_output_index += 1
            events.append(
                _responses_sse_event(
                    "response.output_item.added",
                    ResponseOutputItemAddedEvent(
                        sequence_number=sequence,
                        output_index=reasoning_output_index,
                        item=ResponseReasoningItem(
                            id=reasoning_item_id,
                            status="in_progress",
                            content=[],
                        ),
                    ),
                )
            )
            sequence += 1
            events.append(
                _responses_sse_event(
                    "response.content_part.added",
                    ResponseContentPartAddedEvent(
                        sequence_number=sequence,
                        item_id=reasoning_item_id,
                        output_index=reasoning_output_index,
                        content_index=0,
                        part=ResponseReasoningTextPart(text=""),
                    ),
                )
            )
            sequence += 1
        return events

    if _reasoning_parser:
        _reasoning_parser.reset_state()

    global _tool_parser_instance
    tool_parser = None
    tool_accumulated_text = ""
    tool_markup_possible = False
    if _enable_auto_tool_choice and _tool_call_parser:
        if _tool_parser_instance is None:
            try:
                parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
                tokenizer = None
                if _engine is not None and hasattr(_engine, "_tokenizer"):
                    tokenizer = _engine._tokenizer
                _tool_parser_instance = parser_cls(tokenizer)
                logger.info(
                    "Initialized tool call parser for responses streaming: %s",
                    _tool_call_parser,
                )
            except Exception as e:
                logger.warning(
                    "Failed to init tool parser for responses streaming: %s", e
                )
        if _tool_parser_instance is not None:
            tool_parser = _tool_parser_instance
            tool_parser.reset()

    async for output in engine.stream_chat(messages=messages, **chat_kwargs):
        last_output = output
        finish_reason = output.finish_reason
        if hasattr(output, "prompt_tokens") and output.prompt_tokens:
            prompt_tokens = output.prompt_tokens
        if hasattr(output, "completion_tokens") and output.completion_tokens:
            completion_tokens = output.completion_tokens

        delta_text = output.new_text or ""
        if not delta_text:
            continue

        previous_text = raw_accumulated_text
        raw_accumulated_text += delta_text

        if _reasoning_parser:
            delta_msg = _reasoning_parser.extract_reasoning_streaming(
                previous_text, raw_accumulated_text, delta_text
            )
            if delta_msg is None:
                continue

            if delta_msg.reasoning:
                for event in _start_reasoning_item():
                    yield event
                accumulated_reasoning += delta_msg.reasoning
                yield _responses_sse_event(
                    "response.reasoning_text.delta",
                    ResponseReasoningTextDeltaEvent(
                        sequence_number=sequence,
                        item_id=reasoning_item_id,
                        output_index=reasoning_output_index,
                        content_index=0,
                        delta=delta_msg.reasoning,
                    ),
                )
                sequence += 1

            if delta_msg.content:
                for event in _start_text_item():
                    yield event
                accumulated_text += delta_msg.content
                yield _responses_sse_event(
                    "response.output_text.delta",
                    ResponseOutputTextDeltaEvent(
                        sequence_number=sequence,
                        item_id=text_item_id,
                        output_index=text_output_index,
                        content_index=0,
                        delta=delta_msg.content,
                    ),
                )
                sequence += 1
            continue

        content = SPECIAL_TOKENS_PATTERN.sub("", delta_text)
        if tool_parser and delta_text:
            if not tool_markup_possible and "<" not in delta_text:
                tool_accumulated_text += delta_text
            else:
                if not tool_markup_possible:
                    tool_markup_possible = True
                tool_result = tool_parser.extract_tool_calls_streaming(
                    tool_accumulated_text,
                    tool_accumulated_text + delta_text,
                    delta_text,
                )
                tool_accumulated_text += delta_text
                if tool_result is None:
                    continue
                if "tool_calls" in tool_result:
                    continue
                content = tool_result.get("content", "")

        if not content:
            continue

        for event in _start_text_item():
            yield event
        accumulated_text += content
        yield _responses_sse_event(
            "response.output_text.delta",
            ResponseOutputTextDeltaEvent(
                sequence_number=sequence,
                item_id=text_item_id,
                output_index=text_output_index,
                content_index=0,
                delta=content,
            ),
        )
        sequence += 1

    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        raw_accumulated_text, chat_request
    )
    final_text = accumulated_text
    if cleaned_text is not None and not final_text and not tool_calls:
        final_text = clean_output_text(cleaned_text)

    reasoning_item = None
    if reasoning_item_id is not None:
        reasoning_item = ResponseReasoningItem(
            id=reasoning_item_id,
            status="completed",
            content=[ResponseReasoningTextPart(text=accumulated_reasoning)],
        )
        yield _responses_sse_event(
            "response.reasoning_text.done",
            ResponseReasoningTextDoneEvent(
                sequence_number=sequence,
                item_id=reasoning_item_id,
                output_index=reasoning_output_index,
                content_index=0,
                text=accumulated_reasoning,
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.content_part.done",
            ResponseContentPartDoneEvent(
                sequence_number=sequence,
                item_id=reasoning_item_id,
                output_index=reasoning_output_index,
                content_index=0,
                part=reasoning_item.content[0],
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.output_item.done",
            ResponseOutputItemDoneEvent(
                sequence_number=sequence,
                output_index=reasoning_output_index,
                item=reasoning_item,
            ),
        )
        sequence += 1

    text_item = None
    if text_item_id is not None or final_text:
        if text_item_id is None:
            for event in _start_text_item():
                yield event
        text_item = ResponseMessageItem(
            id=text_item_id,
            role="assistant",
            status="completed",
            content=[ResponseTextContentPart(type="output_text", text=final_text)],
        )
        yield _responses_sse_event(
            "response.output_text.done",
            ResponseOutputTextDoneEvent(
                sequence_number=sequence,
                item_id=text_item_id,
                output_index=text_output_index,
                content_index=0,
                text=final_text,
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.content_part.done",
            ResponseContentPartDoneEvent(
                sequence_number=sequence,
                item_id=text_item_id,
                output_index=text_output_index,
                content_index=0,
                part=text_item.content[0],
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.output_item.done",
            ResponseOutputItemDoneEvent(
                sequence_number=sequence,
                output_index=text_output_index,
                item=text_item,
            ),
        )
        sequence += 1

    function_call_items: list[ResponseFunctionCallItem] = []
    for tool_call in tool_calls or []:
        output_index = next_output_index
        next_output_index += 1
        item = ResponseFunctionCallItem(
            id=_new_response_item_id("fc"),
            call_id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
        function_call_items.append(item)
        yield _responses_sse_event(
            "response.output_item.added",
            ResponseOutputItemAddedEvent(
                sequence_number=sequence,
                output_index=output_index,
                item=item.model_copy(update={"status": "in_progress"}),
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.function_call_arguments.delta",
            ResponseFunctionCallArgumentsDeltaEvent(
                sequence_number=sequence,
                item_id=item.id,
                output_index=output_index,
                delta=item.arguments,
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.output_item.done",
            ResponseOutputItemDoneEvent(
                sequence_number=sequence,
                output_index=output_index,
                item=item,
            ),
        )
        sequence += 1

    output_items: list[
        ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem
    ] = []
    if reasoning_item is not None:
        output_items.append(reasoning_item)
    if text_item is not None:
        output_items.append(text_item)
    output_items.extend(function_call_items)

    response_object = _build_response_object(
        request=request,
        output_items=output_items,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        response_id=response_id,
    )

    if request.store and last_output is not None:
        persisted_messages = _responses_request_to_persisted_messages(request)
        persisted_messages.extend(_response_output_items_to_chat_messages(output_items))
        _responses_store[response_object.id] = {
            "messages": copy.deepcopy(persisted_messages),
            "response": response_object.model_copy(deep=True),
        }
        while len(_responses_store) > _RESPONSES_STORE_MAX_SIZE:
            _responses_store.popitem(last=False)

    yield _responses_sse_event(
        "response.completed",
        ResponseCompletedEvent(sequence_number=sequence, response=response_object),
    )


def _responses_sse_event(event_type: str, payload: BaseModel | dict) -> str:
    """Encode a Responses API SSE event."""
    data = (
        payload.model_dump_json()
        if isinstance(payload, BaseModel)
        else json.dumps(payload)
    )
    return f"event: {event_type}\ndata: {data}\n\n"


def _detect_native_tool_support() -> bool:
    """
    Detect if the active tool parser supports native tool format.

    Native format means role="tool" messages and tool_calls fields
    are preserved instead of being converted to text.

    Returns:
        True if native format should be preserved
    """
    if not _enable_auto_tool_choice or not _tool_call_parser:
        return False

    try:
        parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
        return parser_cls.supports_native_format()
    except KeyError:
        # Parser not found - this is a configuration error, log as error
        logger.error(
            f"Tool parser '{_tool_call_parser}' not found. "
            f"Available parsers: {ToolParserManager.list_registered()}"
        )
        return False
    except Exception as e:
        # Unexpected error during detection
        logger.warning(f"Failed to detect native tool support: {e}")
        return False


def _tool_choice_disabled(request: ChatCompletionRequest | None) -> bool:
    """Return True when tool_choice explicitly disables tool calling."""
    if request is None:
        return False

    tool_choice = getattr(request, "tool_choice", None)
    if tool_choice is None:
        request_dict = request.model_dump()
        tool_choice = request_dict.get("tool_choice")
    return tool_choice == "none"


def _get_streaming_tool_parser(request: ChatCompletionRequest | None):
    """Get a streaming-capable tool parser for this request.

    Uses the configured parser when auto tool choice is enabled, otherwise falls
    back to the generic auto parser so streaming still matches the generic
    non-streaming tool parsing behavior.
    """
    global _tool_parser_instance

    if request is None:
        return None
    if _tool_choice_disabled(request):
        return None

    tokenizer = None
    if _engine is not None and hasattr(_engine, "_tokenizer"):
        tokenizer = _engine._tokenizer

    if _enable_auto_tool_choice and _tool_call_parser:
        if _tool_parser_instance is None:
            try:
                parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
                _tool_parser_instance = parser_cls(tokenizer)
                logger.info(f"Initialized tool call parser: {_tool_call_parser}")
            except Exception as e:
                logger.warning(f"Failed to init tool parser for streaming: {e}")
                return None
        _tool_parser_instance.reset()
        return _tool_parser_instance

    if not getattr(request, "tools", None):
        return None

    try:
        parser_cls = ToolParserManager.get_tool_parser("auto")
        parser = parser_cls(tokenizer)
        parser.reset()
        return parser
    except Exception as e:
        logger.warning(f"Failed to init generic streaming tool parser: {e}")
        return None


def _streaming_tool_markup_possible(text: str) -> bool:
    """Heuristic marker check to avoid parser work on ordinary text chunks."""
    return any(marker in text for marker in _STREAMING_TOOL_MARKERS)


def load_embedding_model(
    model_name: str | None,
    *,
    lock: bool = False,
    reuse_existing: bool = True,
) -> None:
    """Load or reuse the embedding model engine when configured."""
    global _embedding_engine, _embedding_model_locked

    if not model_name:
        return

    if lock:
        _embedding_model_locked = model_name

    if (
        reuse_existing
        and _embedding_engine is not None
        and _embedding_engine.model_name == model_name
    ):
        return

    from .embedding import EmbeddingEngine

    _embedding_engine = EmbeddingEngine(model_name)
    _embedding_engine.load()


def load_model(
    model_name: str,
    use_batching: bool = False,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int = 32768,
    force_mllm: bool = False,
    gpu_memory_utilization: float = 0.90,
    served_model_name: str | None = None,
    trust_remote_code: bool = False,
    mtp: bool = False,
    prefill_step_size: int = 2048,
    specprefill_enabled: bool = False,
    specprefill_threshold: int = 8192,
    specprefill_keep_pct: float = 0.3,
    specprefill_draft_model: str = None,
):
    """
    Load a model (auto-detects MLLM vs LLM).

    Args:
        model_name: HuggingFace model name or local path
        use_batching: Use continuous batching (BatchedEngine) vs simple mode (SimpleEngine)
        scheduler_config: Scheduler config for batched mode
        stream_interval: Tokens to batch before streaming (batched mode only)
        max_tokens: Default max tokens for generation
        force_mllm: Force loading as MLLM even if not auto-detected
        trust_remote_code: Allow HuggingFace remote code execution during model/tokenizer loading
        mtp: Enable native MTP speculative decoding (SimpleEngine only)
        prefill_step_size: Chunk size for prompt prefill processing (default: 2048)
        specprefill_enabled: Enable SpecPrefill (SimpleEngine only)
        specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill (default: 8192)
        specprefill_keep_pct: Fraction of tokens to keep (default: 0.3)
        specprefill_draft_model: Path to small draft model for SpecPrefill scoring
    """
    global _engine, _model_name, _model_path, _default_max_tokens, _tool_parser_instance

    _default_max_tokens = max_tokens
    _model_path = model_name
    _model_name = served_model_name or model_name
    # Reset tool parser instance when model is reloaded (tokenizer may change)
    _tool_parser_instance = None

    if force_mllm:
        logger.info("Force MLLM mode enabled via --mllm flag")

    if use_batching:
        logger.info(f"Loading model with BatchedEngine: {model_name}")
        _engine = BatchedEngine(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=force_mllm,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        # BatchedEngine will be started in lifespan (uvicorn's event loop)
        # Just log for now
        logger.info(f"Model loaded (batched mode): {model_name}")
    else:
        logger.info(f"Loading model with SimpleEngine: {model_name}")
        _engine = SimpleEngine(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            force_mllm=force_mllm,
            mtp=mtp,
            prefill_step_size=prefill_step_size,
            specprefill_enabled=specprefill_enabled,
            specprefill_threshold=specprefill_threshold,
            specprefill_keep_pct=specprefill_keep_pct,
            specprefill_draft_model=specprefill_draft_model,
        )
        # Start SimpleEngine synchronously (no background loop)
        # Use new_event_loop() for Python 3.10+ compatibility (get_event_loop() is deprecated)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_engine.start())
        model_type = "MLLM" if _engine.is_mllm else "LLM"
        logger.info(f"{model_type} model loaded (simple mode): {model_name}")

    # Set native tool format support on the engine (thread-safe via instance property)
    _engine.preserve_native_tool_format = _detect_native_tool_support()
    if _engine.preserve_native_tool_format:
        logger.info(f"Native tool format enabled for parser: {_tool_call_parser}")

    logger.info(f"Default max tokens: {_default_max_tokens}")


def get_usage(output: GenerationOutput) -> Usage:
    """Extract usage metrics from GenerationOutput."""
    total_prompt_tokens = (
        output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
    )
    total_completion_tokens = (
        output.completion_tokens if hasattr(output, "completion_tokens") else 0
    )
    return Usage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint (disabled by default)."""
    if not _metrics.enabled:
        raise HTTPException(status_code=404, detail="Metrics endpoint is disabled")

    payload, content_type = _metrics.render_metrics(
        engine=_engine,
        mcp_manager=_mcp_manager,
    )
    return Response(content=payload, headers={"Content-Type": content_type})


@app.get("/health")
async def health():
    """Health check endpoint."""
    mcp_info = None
    if _mcp_manager is not None:
        connected = sum(
            1 for s in _mcp_manager.get_server_status() if s.state.value == "connected"
        )
        total = len(_mcp_manager.get_server_status())
        mcp_info = {
            "enabled": True,
            "servers_connected": connected,
            "servers_total": total,
            "tools_available": len(_mcp_manager.get_all_tools()),
        }

    return {
        "status": "healthy",
        "model_loaded": _engine is not None,
        "model_name": _model_name,
        "model_type": "mllm" if (_engine and _engine.is_mllm) else "llm",
        "mcp": mcp_info,
    }


@app.get("/v1/status", dependencies=[Depends(verify_api_key)])
async def status():
    """Real-time status with per-request details for debugging and monitoring."""
    if _engine is None:
        return {"status": "not_loaded", "model": None, "requests": []}

    stats = _engine.get_stats()

    return {
        "status": "running" if stats.get("running") else "stopped",
        "model": _model_name,
        "uptime_s": round(stats.get("uptime_seconds", 0), 1),
        "steps_executed": stats.get("steps_executed", 0),
        "num_running": stats.get("num_running", 0),
        "num_waiting": stats.get("num_waiting", 0),
        "total_requests_processed": stats.get("num_requests_processed", 0),
        "total_prompt_tokens": stats.get("total_prompt_tokens", 0),
        "total_completion_tokens": stats.get("total_completion_tokens", 0),
        "metal": {
            "active_memory_gb": stats.get("metal_active_memory_gb"),
            "peak_memory_gb": stats.get("metal_peak_memory_gb"),
            "cache_memory_gb": stats.get("metal_cache_memory_gb"),
        },
        "cache": stats.get("memory_aware_cache")
        or stats.get("paged_cache")
        or stats.get("prefix_cache"),
        "requests": stats.get("requests", []),
    }


@app.get("/v1/cache/stats", dependencies=[Depends(verify_api_key)])
async def cache_stats():
    """Get cache statistics for debugging and monitoring."""
    try:
        from mlx_vlm.utils import (
            get_multimodal_kv_cache_stats,
            get_pil_cache_stats,
            get_pixel_values_cache_stats,
        )

        return {
            "multimodal_kv_cache": get_multimodal_kv_cache_stats(),
            "pixel_values_cache": get_pixel_values_cache_stats(),
            "pil_image_cache": get_pil_cache_stats(),
        }
    except ImportError:
        return {"error": "Cache stats not available (mlx_vlm not loaded)"}


@app.delete("/v1/cache", dependencies=[Depends(verify_api_key)])
async def clear_cache():
    """Clear all caches."""
    try:
        from mlx_vlm.utils import (
            clear_multimodal_kv_cache,
            clear_pixel_values_cache,
        )

        clear_multimodal_kv_cache()
        clear_pixel_values_cache()
        return {
            "status": "cleared",
            "caches": ["multimodal_kv", "pixel_values", "pil_image"],
        }
    except ImportError:
        return {"error": "Cache clear not available (mlx_vlm not loaded)"}


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelsResponse:
    """List available models."""
    models = []
    if _model_name:
        models.append(ModelInfo(id=_model_name))
    return ModelsResponse(data=models)


# =============================================================================
# Embeddings Endpoint
# =============================================================================


@app.post(
    "/v1/embeddings",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Create embeddings for the given input text(s).

    OpenAI-compatible embeddings API supporting single or batch inputs.

    Single text:
    ```json
    {
      "model": "mlx-community/all-MiniLM-L6-v2-4bit",
      "input": "The quick brown fox jumps over the lazy dog"
    }
    ```

    Batch of texts:
    ```json
    {
      "model": "mlx-community/embeddinggemma-300m-6bit",
      "input": [
        "I love machine learning",
        "Deep learning is fascinating",
        "Neural networks are powerful"
      ]
    }
    ```

    Response:
    ```json
    {
      "object": "list",
      "data": [
        {"object": "embedding", "index": 0, "embedding": [0.023, -0.982, ...]},
        {"object": "embedding", "index": 1, "embedding": [0.112, -0.543, ...]},
        {"object": "embedding", "index": 2, "embedding": [0.876, 0.221, ...]}
      ],
      "model": "mlx-community/embeddinggemma-300m-6bit",
      "usage": {"prompt_tokens": 24, "total_tokens": 24}
    }
    ```

    Supported request-time models:
    - mlx-community/all-MiniLM-L6-v2-4bit (fast, compact)
    - mlx-community/embeddinggemma-300m-6bit (high quality)
    - mlx-community/bge-large-en-v1.5-4bit (best for English)
    - mlx-community/multilingual-e5-small-mlx
    - mlx-community/multilingual-e5-large-mlx
    - mlx-community/bert-base-uncased-mlx
    - mlx-community/ModernBERT-base-mlx

    Other embedding models must be pinned explicitly with --embedding-model at
    server startup.
    """
    global _embedding_engine
    tracker = _metrics.track_inference("embeddings", stream=False)

    try:
        # Resolve model name before any lazy-load path is reached.
        model_name = resolve_embedding_model_name(
            request.model,
            locked_model=_embedding_model_locked,
        )

        # Lazy-load or swap embedding engine
        load_embedding_model(model_name, lock=False, reuse_existing=True)

        # Normalise input to list
        texts = request.input if isinstance(request.input, list) else [request.input]

        if not texts:
            raise HTTPException(status_code=400, detail="Input must not be empty")

        start_time = time.perf_counter()

        # Count tokens for usage reporting
        prompt_tokens = _embedding_engine.count_tokens(texts)

        # Generate embeddings (batch)
        embeddings = _embedding_engine.embed(texts)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Embeddings: {len(texts)} inputs, {prompt_tokens} tokens in {elapsed:.2f}s"
        )

        # Build OpenAI-compatible response with ordered indices
        data = [
            EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(embeddings)
        ]

        response = EmbeddingResponse(
            data=data,
            model=model_name,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )
        tracker.finish(
            result="success",
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
        )
        return response

    except ImportError:
        tracker.finish(result="error")
        raise HTTPException(
            status_code=503,
            detail=(
                "mlx-embeddings not installed. Install with: pip install mlx-embeddings"
            ),
        )
    except HTTPException as exc:
        tracker.finish(result=_metrics_result_from_status(exc.status_code))
        raise
    except Exception as e:
        tracker.finish(result="error")
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MCP Endpoints
# =============================================================================


@app.get("/v1/mcp/tools", dependencies=[Depends(verify_api_key)])
async def list_mcp_tools() -> MCPToolsResponse:
    """List all available MCP tools."""
    if _mcp_manager is None:
        return MCPToolsResponse(tools=[], count=0)

    tools = []
    for tool in _mcp_manager.get_all_tools():
        tools.append(
            MCPToolInfo(
                name=tool.full_name,
                description=tool.description,
                server=tool.server_name,
                parameters=tool.input_schema,
            )
        )

    return MCPToolsResponse(tools=tools, count=len(tools))


@app.get("/v1/mcp/servers", dependencies=[Depends(verify_api_key)])
async def list_mcp_servers() -> MCPServersResponse:
    """Get status of all MCP servers."""
    if _mcp_manager is None:
        return MCPServersResponse(servers=[])

    servers = []
    for status in _mcp_manager.get_server_status():
        servers.append(
            MCPServerInfo(
                name=status.name,
                state=status.state.value,
                transport=status.transport.value,
                tools_count=status.tools_count,
                error=status.error,
            )
        )

    return MCPServersResponse(servers=servers)


@app.post("/v1/mcp/execute", dependencies=[Depends(verify_api_key)])
async def execute_mcp_tool(request: MCPExecuteRequest) -> MCPExecuteResponse:
    """Execute an MCP tool."""
    global _mcp_executor

    if _mcp_manager is None:
        raise HTTPException(
            status_code=503, detail="MCP not configured. Start server with --mcp-config"
        )

    if _mcp_executor is None:
        from vllm_mlx.mcp import ToolExecutor

        _mcp_executor = ToolExecutor(_mcp_manager)

    tool_call = {
        "id": f"mcp-{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": request.tool_name,
            "arguments": request.arguments,
        },
    }
    result, _ = (await _mcp_executor.execute_tool_calls([tool_call], parallel=False))[0]

    return MCPExecuteResponse(
        tool_name=result.tool_name,
        content=result.content,
        is_error=result.is_error,
        error_message=result.error_message,
    )


# =============================================================================
# Audio Endpoints
# =============================================================================

# Global audio engines (lazy loaded)
_stt_engine = None
_tts_engine = None


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def create_transcription(
    file: UploadFile,
    model: str = "whisper-large-v3",
    language: str | None = None,
    response_format: str = "json",
):
    """
    Transcribe audio to text (OpenAI Whisper API compatible).

    Supported models:
    - whisper-large-v3 (multilingual, best quality)
    - whisper-large-v3-turbo (faster)
    - whisper-medium, whisper-small (lighter)
    - parakeet-tdt-0.6b-v2 (English, fastest)
    """
    global _stt_engine
    tracker = _metrics.track_inference("audio_transcriptions", stream=False)

    try:
        from .audio.stt import STTEngine  # Lazy import - optional feature

        model_name = resolve_stt_model_name(model)

        # Load engine if needed
        if _stt_engine is None or _stt_engine.model_name != model_name:
            _stt_engine = STTEngine(model_name)
            _stt_engine.load()

        # Stream uploaded file to disk under a hard size cap.
        tmp_path = await save_upload_with_limit(
            file,
            max_bytes=_max_audio_upload_bytes,
            default_suffix=".wav",
        )

        try:
            result = _stt_engine.transcribe(tmp_path, language=language)
        finally:
            os.unlink(tmp_path)

        if response_format == "text":
            tracker.finish(result="success")
            return result.text

        tracker.finish(result="success")
        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
        }

    except ImportError:
        tracker.finish(result="error")
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except HTTPException as exc:
        tracker.finish(result=_metrics_result_from_status(exc.status_code))
        raise
    except Exception as e:
        tracker.finish(result="error")
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(
    model: str = "kokoro",
    input: str = "",
    voice: str = "af_heart",
    speed: float = 1.0,
    response_format: str = "wav",
):
    """
    Generate speech from text (OpenAI TTS API compatible).

    Supported models:
    - kokoro (fast, lightweight)
    - chatterbox (multilingual, expressive)
    - vibevoice (realtime)
    - voxcpm (Chinese/English)
    """
    global _tts_engine
    tracker = _metrics.track_inference("audio_speech", stream=False)

    try:
        from .audio.tts import TTSEngine  # Lazy import - optional feature

        model_name = resolve_tts_model_name(model)
        validate_tts_input_length(input, max_chars=_max_tts_input_chars)

        # Load engine if needed
        if _tts_engine is None or _tts_engine.model_name != model_name:
            _tts_engine = TTSEngine(model_name)
            _tts_engine.load()

        audio = _tts_engine.generate(input, voice=voice, speed=speed)
        audio_bytes = _tts_engine.to_bytes(audio, format=response_format)

        content_type = (
            "audio/wav" if response_format == "wav" else f"audio/{response_format}"
        )
        tracker.finish(result="success")
        return Response(content=audio_bytes, media_type=content_type)

    except ImportError:
        tracker.finish(result="error")
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except HTTPException as exc:
        tracker.finish(result=_metrics_result_from_status(exc.status_code))
        raise
    except Exception as e:
        tracker.finish(result="error")
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(model: str = "kokoro"):
    """List available voices for a TTS model."""
    from .audio.tts import CHATTERBOX_VOICES, KOKORO_VOICES

    if "kokoro" in model.lower():
        return {"voices": KOKORO_VOICES}
    elif "chatterbox" in model.lower():
        return {"voices": CHATTERBOX_VOICES}
    else:
        return {"voices": ["default"]}


# =============================================================================
# Streaming disconnect detection
# =============================================================================


async def _ensure_sse_terminal(
    generator: AsyncIterator[str],
    terminal_frame: str,
) -> AsyncIterator[str]:
    """Guarantee that *terminal_frame* is emitted exactly once at the end of
    *generator*, even if the generator raises mid-stream.

    If the inner generator already yields the terminal frame on its happy path,
    the wrapper detects it and avoids double-emission.  If the generator raises
    before reaching the terminal, the wrapper emits it in the ``finally`` block.
    """
    emitted = False
    try:
        async for chunk in generator:
            if chunk == terminal_frame:
                emitted = True
            yield chunk
    except Exception as e:
        logger.error(f"Streaming error, ensuring terminal frame: {e}")
    finally:
        if not emitted:
            yield terminal_frame


async def _disconnect_guard(
    generator: AsyncIterator[str],
    raw_request: Request,
    poll_interval: float = 0.5,
    heartbeat_interval: float = 5.0,
) -> AsyncIterator[str]:
    """Wrap streaming generator to abort on client disconnect.

    Uses asyncio racing: each __anext__() on the inner generator is
    raced against a disconnect poller.  When neither completes within
    ``heartbeat_interval`` seconds, an SSE comment is yielded as a
    heartbeat.  This forces an ASGI write which triggers broken-pipe
    detection — without heartbeats, ``is_disconnected()`` stays False
    during long prefill because no data is written to the socket.

    On disconnect, the cancellation propagates to stream_outputs()
    finally-block → abort_request() → abort_prefill().
    """
    import time as _time

    _t0 = _time.monotonic()

    def _elapsed():
        return f"{_time.monotonic() - _t0:.1f}s"

    logger.info(
        f"[disconnect_guard] START poll={poll_interval}s heartbeat={heartbeat_interval}s"
    )

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = await raw_request.is_disconnected()
            if poll_count % 10 == 0 or is_disc:
                logger.info(
                    f"[disconnect_guard] poll #{poll_count} "
                    f"disconnected={is_disc} elapsed={_elapsed()}"
                )
            if is_disc:
                return

    chunk_count = 0
    heartbeat_count = 0
    disconnect_task: asyncio.Task | None = None
    anext_task: asyncio.Task | None = None
    try:
        aiter = generator.__aiter__()
        disconnect_task = asyncio.create_task(_wait_disconnect())
        anext_task = None
        while True:
            if anext_task is None:
                anext_task = asyncio.ensure_future(aiter.__anext__())

            done, _ = await asyncio.wait(
                [anext_task, disconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=heartbeat_interval,
            )

            if disconnect_task in done:
                logger.info(
                    f"[disconnect_guard] CLIENT DISCONNECTED after "
                    f"{chunk_count} chunks, {heartbeat_count} heartbeats, "
                    f"elapsed={_elapsed()}"
                )
                anext_task.cancel()
                try:
                    await anext_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
                break

            if anext_task in done:
                try:
                    chunk = anext_task.result()
                except StopAsyncIteration:
                    logger.info(
                        f"[disconnect_guard] generator exhausted normally, "
                        f"{chunk_count} chunks, elapsed={_elapsed()}"
                    )
                    break
                except Exception as exc:
                    logger.error(
                        f"[disconnect_guard] generator raised {type(exc).__name__}: {exc}, "
                        f"after {chunk_count} chunks, elapsed={_elapsed()}"
                    )
                    break
                chunk_count += 1
                if chunk_count == 1:
                    logger.info(
                        f"[disconnect_guard] first chunk arrived, elapsed={_elapsed()}"
                    )
                yield chunk
                anext_task = None
                continue

            # Timeout — no chunk and no disconnect detected yet.
            # Send SSE comment as heartbeat to force an ASGI write.
            # If the client has disconnected, this write will fail and
            # the next is_disconnected() poll will return True.
            heartbeat_count += 1
            yield ": heartbeat\n\n"

    except GeneratorExit:
        logger.info(
            f"[disconnect_guard] GeneratorExit after {chunk_count} chunks, elapsed={_elapsed()}"
        )
    finally:
        if disconnect_task and not disconnect_task.done():
            disconnect_task.cancel()
        if anext_task and not anext_task.done():
            anext_task.cancel()
        # NOTE: Do NOT call generator.aclose() here.  With run_in_executor,
        # scheduler.step() runs in a background thread.  aclose() would throw
        # GeneratorExit into the async-generator chain, which can trigger
        # mlx::core::eval on the main thread while the executor thread is also
        # mid-eval → Metal assertion failure → SIGABRT.
        #
        # Instead, rely on the task cancellation propagation:
        #   anext_task.cancel() → CancelledError in stream_outputs()
        #   → finally block → abort_request() → request removed from scheduler
        logger.info(
            f"[disconnect_guard] CLEANUP done, {chunk_count} chunks, "
            f"{heartbeat_count} heartbeats, elapsed={_elapsed()}"
        )


async def _wait_with_disconnect(
    coro,
    raw_request: Request,
    timeout: float,
    poll_interval: float = 0.5,
):
    """Run a coroutine with both timeout and client disconnect detection.

    For non-streaming requests where _disconnect_guard() can't be used.
    Races the coroutine against a disconnect poller, same pattern as
    _disconnect_guard but for awaitable (non-generator) coroutines.
    """
    import time as _time

    _t0 = _time.monotonic()

    task = asyncio.ensure_future(coro)

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = await raw_request.is_disconnected()
            if poll_count % 10 == 0 or is_disc:
                logger.info(
                    f"[disconnect_guard] poll #{poll_count} "
                    f"disconnected={is_disc} elapsed={_time.monotonic() - _t0:.1f}s"
                )
            if is_disc:
                return

    disconnect_task = asyncio.create_task(_wait_disconnect())

    try:
        done, _ = await asyncio.wait(
            [task, disconnect_task],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            # Timeout
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {timeout:.1f} seconds",
            )

        if disconnect_task in done:
            # Client disconnected
            logger.info(
                f"[disconnect_guard] CLIENT DISCONNECTED (non-stream) "
                f"elapsed={_time.monotonic() - _t0:.1f}s"
            )
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            return None  # Signal to caller that client disconnected

        # Task completed
        return task.result()

    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
        if not task.done():
            task.cancel()


# =============================================================================
# Completion Endpoints
# =============================================================================


@app.post(
    "/v1/completions", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Create a text completion."""
    _validate_model_name(request.model)
    engine = get_engine()
    tracker = _metrics.track_inference("completions", stream=request.stream)

    # Handle single prompt or list of prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    # --- Detailed request logging ---
    prompt_preview = prompts[0][:200] if prompts else "(empty)"
    prompt_len = sum(len(p) for p in prompts)
    logger.info(
        f"[REQUEST] POST /v1/completions stream={request.stream} "
        f"max_tokens={request.max_tokens} temp={request.temperature} "
        f"top_p={request.top_p} top_k={request.top_k} min_p={request.min_p} "
        f"presence_penalty={request.presence_penalty} "
        f"repetition_penalty={request.repetition_penalty} "
        f"prompt_chars={prompt_len} prompt_preview={prompt_preview!r}"
    )

    # Resolve repetition penalty for completions
    comp_rep_penalty = request.repetition_penalty

    if request.stream:
        return StreamingResponse(
            _disconnect_guard(
                _ensure_sse_terminal(
                    stream_completion(
                        engine,
                        prompts[0],
                        request,
                        repetition_penalty=comp_rep_penalty,
                        metrics_tracker=tracker,
                    ),
                    "data: [DONE]\n\n",
                ),
                raw_request,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or _default_timeout
    choices = []
    total_completion_tokens = 0
    total_prompt_tokens = 0

    for i, prompt in enumerate(prompts):
        generate_kwargs = {
            "prompt": prompt,
            "max_tokens": request.max_tokens or _default_max_tokens,
            "temperature": _resolve_temperature(request.temperature),
            "top_p": _resolve_top_p(request.top_p),
            "top_k": request.top_k or 0,
            "min_p": request.min_p or 0.0,
            "presence_penalty": request.presence_penalty or 0.0,
            "stop": request.stop,
        }
        if comp_rep_penalty is not None:
            generate_kwargs["repetition_penalty"] = comp_rep_penalty
        if request.specprefill is not None:
            generate_kwargs["specprefill"] = request.specprefill
        if request.specprefill_keep_pct is not None:
            generate_kwargs["specprefill_keep_pct"] = request.specprefill_keep_pct

        try:
            output = await _wait_with_disconnect(
                engine.generate(**generate_kwargs),
                raw_request,
                timeout=timeout,
            )
        except HTTPException as exc:
            tracker.finish(result=_metrics_result_from_status(exc.status_code))
            raise
        if output is None:
            tracker.finish(
                result="client_closed",
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            )
            return Response(status_code=499)  # Client closed request

        choices.append(
            CompletionChoice(
                index=i,
                text=output.text,
                finish_reason=output.finish_reason,
            )
        )
        total_completion_tokens += output.completion_tokens
        total_prompt_tokens += (
            output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
        )

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Completion: {total_prompt_tokens} prompt + {total_completion_tokens} completion tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    tracker.finish(
        result="success",
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
    )
    return CompletionResponse(
        model=_model_name,
        choices=choices,
        usage=Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )


@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create a chat completion (supports multimodal content for VLM models).

    OpenAI-compatible multimodal format for images:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
    ```

    Video support:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }]
    ```

    Structured output (JSON mode):
    ```json
    response_format={"type": "json_object"}
    ```

    Structured output (JSON Schema):
    ```json
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "schema": {"type": "object", "properties": {...}}
        }
    }
    ```
    """
    _validate_model_name(request.model)
    engine = get_engine()
    tracker = _metrics.track_inference("chat_completions", stream=request.stream)

    # --- Detailed request logging ---
    n_msgs = len(request.messages)
    msg_roles = [m.role for m in request.messages]
    total_chars = 0
    last_user_preview = ""
    for m in request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    has_tools = bool(request.tools)
    n_tools = len(request.tools) if request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/chat/completions stream={request.stream} "
        f"model={request.model!r} max_tokens={request.max_tokens} "
        f"temp={request.temperature} top_p={request.top_p} "
        f"top_k={request.top_k} min_p={request.min_p} "
        f"presence_penalty={request.presence_penalty} "
        f"repetition_penalty={request.repetition_penalty} "
        f"msgs={n_msgs} roles={msg_roles} "
        f"total_chars={total_chars} tools={n_tools} "
        f"response_format={request.response_format}"
    )
    logger.info(f"[REQUEST] last user message preview: {last_user_preview!r}")

    # For MLLM models, keep original messages with embedded images
    # (MLLM.chat() extracts images from message content internally)
    if engine.is_mllm:
        # Convert Pydantic messages to dicts, excluding None fields
        # to prevent chat templates from misinterpreting key presence
        # (e.g. image_url: null on text parts triggers Qwen3-VL crash)
        messages = []
        for msg in request.messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                raw = dict(msg)
                msg_dict = {k: v for k, v in raw.items() if v is not None}
            messages.append(msg_dict)
        images, videos = [], []  # MLLM extracts these from messages
        logger.debug(f"MLLM: Processing {len(messages)} messages")
        # Convert tool_call arguments from JSON string to dict so that
        # chat templates can iterate them (e.g. GLM-4.6V calls .items()).
        # The LLM path does this inside extract_multimodal_content(), but
        # the MLLM path bypasses that function.
        if engine.preserve_native_tool_format:
            for msg_dict in messages:
                for tc in msg_dict.get("tool_calls") or []:
                    func = tc.get("function") or {}
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            pass
        messages = _normalize_messages(messages)
    else:
        # For LLM, extract text, images, and videos separately
        messages, images, videos = extract_multimodal_content(
            request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )
        messages = _normalize_messages(messages)

    has_media = bool(images or videos)
    if engine.is_mllm and not has_media:
        # MLLM extracts media from messages directly, so images/videos are
        # always empty. Check message content for video/image types instead.
        for msg in request.messages:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    item_type = (
                        item.type
                        if hasattr(item, "type")
                        else (item.get("type", "") if isinstance(item, dict) else "")
                    )
                    if item_type in ("image_url", "image", "video", "video_url"):
                        has_media = True
                        break
            if has_media:
                break

    # Handle response_format - inject system prompt if needed
    response_format = request.response_format
    json_logits_processor = None
    if response_format:
        json_instruction = build_json_system_prompt(response_format)
        if json_instruction:
            # Inject JSON instruction into messages
            messages = _inject_json_instruction(messages, json_instruction)

        # Build a grammar-guided logits processor so the model *cannot*
        # emit invalid JSON.  If the optional ``lm-format-enforcer``
        # dependency is missing, or the tokenizer cannot be adapted, this
        # returns ``None`` and we fall back to prompt-only guidance plus
        # post-hoc validation.  ``tools`` + ``response_format`` is an
        # undefined combination in the OpenAI spec — we skip the
        # constraint in that case so tool-call markup can still be
        # emitted.
        if not (request.tools and request.tool_choice != "none"):
            tokenizer_obj = _get_engine_tokenizer(engine)
            if tokenizer_obj is not None:
                try:
                    json_logits_processor = build_json_logits_processor(
                        response_format, tokenizer_obj
                    )
                except Exception as exc:
                    logger.warning("Failed to build JSON logits processor: %s", exc)
                    json_logits_processor = None
                if json_logits_processor is not None:
                    logger.info(
                        "Constrained decoding enabled for response_format.type=%s",
                        (
                            getattr(response_format, "type", None)
                            if not isinstance(response_format, dict)
                            else response_format.get("type")
                        ),
                    )

    # Resolve repetition penalty
    rep_penalty = request.repetition_penalty

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": request.max_tokens or _default_max_tokens,
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "top_k": request.top_k or 0,
        "min_p": request.min_p or 0.0,
        "presence_penalty": request.presence_penalty or 0.0,
        "repetition_penalty": request.repetition_penalty or 1.0,
    }
    if rep_penalty is not None:
        chat_kwargs["repetition_penalty"] = rep_penalty

    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        if request.video_fps:
            chat_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            chat_kwargs["video_max_frames"] = request.video_max_frames

    # SpecPrefill: per-request overrides
    if request.specprefill is not None:
        chat_kwargs["specprefill"] = request.specprefill
    if request.specprefill_keep_pct is not None:
        chat_kwargs["specprefill_keep_pct"] = request.specprefill_keep_pct
    if request.chat_template_kwargs:
        chat_kwargs["chat_template_kwargs"] = dict(request.chat_template_kwargs)

    # Enable/disable thinking mode per request
    if request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = request.enable_thinking

    # Add tools if provided
    if request.tools and request.tool_choice != "none":
        chat_kwargs["tools"] = convert_tools_for_template(request.tools)

    # Wire constrained decoding logits processor if available.  Must be
    # appended after all other kwargs because the engine may merge it with
    # penalty processors internally.
    if json_logits_processor is not None:
        existing = chat_kwargs.get("logits_processors") or []
        chat_kwargs["logits_processors"] = list(existing) + [json_logits_processor]
        # Constrained decoding is incompatible with reasoning parsers:
        # the model cannot emit <think> tags when forced to produce JSON.
        # Suppress thinking so the reasoning parser doesn't capture the
        # JSON output as reasoning_content.
        if request.enable_thinking is None:
            request.enable_thinking = False
            chat_kwargs["enable_thinking"] = False

    if request.stream:
        return StreamingResponse(
            _disconnect_guard(
                _ensure_sse_terminal(
                    stream_chat_completion(
                        engine,
                        messages,
                        request,
                        metrics_tracker=tracker,
                        **chat_kwargs,
                    ),
                    "data: [DONE]\n\n",
                ),
                raw_request,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or _default_timeout

    try:
        output = await _wait_with_disconnect(
            engine.chat(messages=messages, **chat_kwargs),
            raw_request,
            timeout=timeout,
        )
    except HTTPException as exc:
        tracker.finish(result=_metrics_result_from_status(exc.status_code))
        raise
    if output is None:
        tracker.finish(result="client_closed")
        return Response(status_code=499)  # Client closed request

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Parse tool calls from output using configured parser
    # Skip tool parsing when request has no tools — otherwise the parser
    # can misinterpret JSON output (e.g. response_format) as tool calls.
    if request.tools:
        cleaned_text, tool_calls = _parse_tool_calls_with_parser(output.text, request)
    else:
        cleaned_text, tool_calls = output.text, None

    # Extract reasoning content (strips channel tokens before JSON extraction)
    # Skip reasoning parser when enable_thinking=False (no think tags expected)
    reasoning_text = None
    if _reasoning_parser and request.enable_thinking is not False:
        # Always use original output.text for reasoning extraction so
        # <think> content is preserved even when tool calls are present.
        text_to_parse = output.text
        reasoning_text, remaining_text = _reasoning_parser.extract_reasoning(
            text_to_parse
        )
        # Only update cleaned_text from reasoning parser when no tool calls
        # (tool parser already set cleaned_text appropriately)
        if not tool_calls:
            cleaned_text = remaining_text

    # Process response_format if specified (after reasoning parser cleaned the text)
    if response_format and not tool_calls:
        json_input = cleaned_text or output.text
        _, parsed_json, is_valid, error = parse_json_output(json_input, response_format)
        if parsed_json is not None:
            # Return JSON as string
            cleaned_text = json.dumps(parsed_json)
        if not is_valid:
            if json_logits_processor is not None:
                # Constrained decoding was active yet validation still failed.
                # This is unexpected and indicates a bug in the grammar
                # integration — log at error level so it surfaces in CI/logs.
                logger.error("Constrained decoding produced invalid JSON: %s", error)
            else:
                logger.warning(f"JSON validation failed: {error}")

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    tracker.finish(
        result="success",
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
    )
    return ChatCompletionResponse(
        model=_model_name,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=clean_output_text(cleaned_text) if cleaned_text else None,
                    reasoning=reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            total_tokens=output.prompt_tokens + output.completion_tokens,
        ),
    )


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize message roles and merge consecutive same-role messages.

    1. Maps non-standard roles to standard ones (e.g. ``developer`` -> ``system``).
    2. Merges consecutive same-role messages to satisfy chat template constraints
       (Qwen 3.5, Llama, etc. require alternating roles).

    Only merges when both messages have string content. Messages with list
    content (multimodal) are left as-is to preserve image/video attachments.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        New list with normalized roles and consecutive same-role messages merged.
    """
    # OpenAI Responses API uses "developer" instead of "system".
    # Map it so chat templates don't fail and fall back to raw prefill.
    _ROLE_MAP = {"developer": "system"}

    if not messages:
        return messages

    merged = [messages[0].copy()]
    if merged[0]["role"] in _ROLE_MAP:
        merged[0]["role"] = _ROLE_MAP[merged[0]["role"]]
    for msg in messages[1:]:
        prev = merged[-1]
        role = _ROLE_MAP.get(msg["role"], msg["role"])
        if (
            role == prev["role"]
            and isinstance(prev.get("content"), str)
            and isinstance(msg.get("content"), str)
        ):
            # Merge string content with double newline separator
            prev["content"] = prev["content"] + "\n\n" + msg["content"]
            logger.debug(
                f"Merged consecutive {role} messages "
                f"({len(prev['content'])} chars total)"
            )
        else:
            copy = msg.copy()
            copy["role"] = role
            merged.append(copy)

    mapped_roles = sum(1 for m in messages if m["role"] in _ROLE_MAP)
    merged_count = len(messages) - len(merged)
    if mapped_roles or merged_count:
        parts = []
        if mapped_roles:
            parts.append(f"mapped {mapped_roles} role(s)")
        if merged_count:
            parts.append(f"merged {len(messages)} -> {len(merged)}")
        logger.info(f"Normalized messages: {', '.join(parts)}")

    return merged


def _get_engine_tokenizer(engine) -> object | None:
    """
    Return the tokenizer backing ``engine``, if exposed.

    Different engine classes store the tokenizer under different attributes.
    We try the common ones and return ``None`` if nothing matches, so that
    optional features like constrained decoding can degrade gracefully.
    """
    for attr in ("_tokenizer", "tokenizer", "_processor", "processor"):
        tok = getattr(engine, attr, None)
        if tok is not None:
            return tok
    return None


@app.post(
    "/v1/responses",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_response(request: ResponsesRequest, raw_request: Request):
    """Create a Responses API response."""
    if request.stream:
        return StreamingResponse(
            _disconnect_guard(_stream_responses_request(request), raw_request),
            media_type="text/event-stream",
        )

    response_object, _persisted_messages = await _run_responses_request(
        request, raw_request
    )
    if response_object is None:
        return Response(status_code=499)

    return response_object


def _inject_json_instruction(messages: list, instruction: str) -> list:
    """
    Inject JSON instruction into messages.

    If a system message exists, append to it. Otherwise, prepend a new system message.
    """
    messages = list(messages)  # Make a copy

    # Find existing system message
    system_idx = None
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            system_idx = i
            break

    if system_idx is not None:
        # Append to existing system message
        msg = messages[system_idx]
        if isinstance(msg, dict):
            existing = msg.get("content", "")
            msg["content"] = f"{existing}\n\n{instruction}"
        else:
            existing = getattr(msg, "content", "") or ""
            msg.content = f"{existing}\n\n{instruction}"
    else:
        # Prepend new system message
        messages.insert(0, {"role": "system", "content": instruction})

    return messages


# =============================================================================
# Anthropic Messages API Endpoints
# =============================================================================


def _convert_anthropic_stop_reason(openai_reason: str | None) -> str:
    """Convert OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    return mapping.get(openai_reason or "", "end_turn")


@app.post(
    "/v1/messages", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def create_anthropic_message(
    request: Request,
):
    """
    Anthropic Messages API endpoint.

    Translates Anthropic-format requests to OpenAI format, runs inference
    through the existing engine, and converts the response back.

    Supports both streaming and non-streaming modes.
    """
    engine = get_engine()
    tracker = _metrics.track_inference("anthropic_messages", stream=False)

    # Parse the raw body to handle Anthropic request format.
    # Some clients (e.g. Claude Code) may send JSON with invalid escape
    # sequences like \s, \d in regex patterns within tool definitions.
    # Python's json.loads is strict per RFC 8259 and rejects these.
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        if "Invalid \\escape" in str(e):
            raw = await request.body()
            # Replace lone backslashes (not valid JSON escapes) with \\
            body = json.loads(re.sub(rb'\\(?!["\\/bfnrtu])', rb"\\\\", raw))
        else:
            raise
    anthropic_request = AnthropicRequest(**body)

    _validate_model_name(anthropic_request.model)

    # --- Detailed request logging ---
    n_msgs = len(anthropic_request.messages)
    total_chars = 0
    last_user_preview = ""
    for m in anthropic_request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    sys_chars = len(anthropic_request.system) if anthropic_request.system else 0
    n_tools = len(anthropic_request.tools) if anthropic_request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/messages (anthropic) stream={anthropic_request.stream} "
        f"model={anthropic_request.model!r} max_tokens={anthropic_request.max_tokens} "
        f"msgs={n_msgs} total_chars={total_chars} system_chars={sys_chars} "
        f"tools={n_tools}"
    )
    logger.info(f"[REQUEST] last user message preview: {last_user_preview!r}")

    # Convert Anthropic request -> OpenAI request
    openai_request = anthropic_to_openai(anthropic_request)

    if anthropic_request.stream:
        _anthropic_terminal = (
            f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        )
        return StreamingResponse(
            _disconnect_guard(
                _ensure_sse_terminal(
                    _stream_anthropic_messages(
                        engine,
                        openai_request,
                        anthropic_request,
                        metrics_tracker=tracker,
                    ),
                    _anthropic_terminal,
                ),
                request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming: run inference through existing engine
    messages, images, videos = extract_multimodal_content(
        openai_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )
    messages = _normalize_messages(messages)

    # Handle response_format (propagated from Anthropic request via adapter):
    # inject prompt-level instruction AND wire a grammar-guided logits
    # processor so the model cannot emit invalid JSON.
    response_format = openai_request.response_format
    json_logits_processor = None
    if response_format:
        json_instruction = build_json_system_prompt(response_format)
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)
        if not (openai_request.tools and openai_request.tool_choice != "none"):
            tokenizer_obj = _get_engine_tokenizer(engine)
            if tokenizer_obj is not None:
                try:
                    json_logits_processor = build_json_logits_processor(
                        response_format, tokenizer_obj
                    )
                except Exception as exc:
                    logger.warning("Failed to build JSON logits processor: %s", exc)
                    json_logits_processor = None
                if json_logits_processor is not None:
                    logger.info(
                        "Constrained decoding enabled for Anthropic "
                        "response_format.type=%s",
                        (
                            getattr(response_format, "type", None)
                            if not isinstance(response_format, dict)
                            else response_format.get("type")
                        ),
                    )

    chat_kwargs = {
        "max_tokens": openai_request.max_tokens or _default_max_tokens,
        "temperature": openai_request.temperature,
        "top_p": openai_request.top_p,
        "top_k": openai_request.top_k or 0,
        "min_p": openai_request.min_p or 0.0,
        "presence_penalty": openai_request.presence_penalty or 0.0,
        "repetition_penalty": openai_request.repetition_penalty or 1.0,
    }

    if openai_request.tools and openai_request.tool_choice != "none":
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)

    if json_logits_processor is not None:
        existing = chat_kwargs.get("logits_processors") or []
        chat_kwargs["logits_processors"] = list(existing) + [json_logits_processor]
        # Suppress thinking: constrained decoding prevents <think> tags.
        chat_kwargs["enable_thinking"] = False

    start_time = time.perf_counter()
    timeout = _default_timeout

    try:
        output = await _wait_with_disconnect(
            engine.chat(messages=messages, **chat_kwargs),
            request,
            timeout=timeout,
        )
    except HTTPException as exc:
        tracker.finish(result=_metrics_result_from_status(exc.status_code))
        raise
    if output is None:
        tracker.finish(result="client_closed")
        return Response(status_code=499)  # Client closed request

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Anthropic messages: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Parse tool calls (skip when no tools to avoid misinterpreting output)
    if openai_request.tools:
        cleaned_text, tool_calls = _parse_tool_calls_with_parser(
            output.text, openai_request
        )
    else:
        cleaned_text, tool_calls = output.text, None

    # Extract reasoning if parser is configured — skip when constrained
    # decoding was active (no <think> tags possible).
    reasoning_text = None
    if _reasoning_parser and not tool_calls and json_logits_processor is None:
        text_to_parse = cleaned_text or output.text
        reasoning_text, cleaned_text = _reasoning_parser.extract_reasoning(
            text_to_parse
        )

    # Post-hoc response_format validation / normalization (safety net even
    # when constrained decoding was active).
    if response_format and not tool_calls:
        json_input = cleaned_text or output.text
        _, parsed_json, is_valid, error = parse_json_output(json_input, response_format)
        if parsed_json is not None:
            cleaned_text = json.dumps(parsed_json)
        if not is_valid:
            if json_logits_processor is not None:
                logger.error(
                    "Constrained decoding produced invalid JSON on "
                    "Anthropic endpoint: %s",
                    error,
                )
            else:
                logger.warning(
                    "JSON validation failed on Anthropic endpoint: %s", error
                )

    # Clean output text
    final_content = None
    if cleaned_text:
        final_content = clean_output_text(cleaned_text)

    # Build Anthropic content blocks directly (with thinking support)
    content_blocks = []

    if reasoning_text:
        content_blocks.append(
            AnthropicResponseContentBlock(type="thinking", thinking=reasoning_text)
        )

    if final_content:
        content_blocks.append(
            AnthropicResponseContentBlock(type="text", text=final_content)
        )

    if tool_calls:
        for tc in tool_calls:
            try:
                tool_input = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                tool_input = {}
            content_blocks.append(
                AnthropicResponseContentBlock(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=tool_input,
                )
            )

    if not content_blocks:
        content_blocks.append(AnthropicResponseContentBlock(type="text", text=""))

    stop_reason = _convert_anthropic_stop_reason(
        "tool_calls" if tool_calls else output.finish_reason
    )

    anthropic_response = AnthropicResponse(
        model=_model_name,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=AnthropicUsage(
            input_tokens=output.prompt_tokens,
            output_tokens=output.completion_tokens,
        ),
    )
    tracker.finish(
        result="success",
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
    )
    return Response(
        content=anthropic_response.model_dump_json(exclude_none=True),
        media_type="application/json",
    )


@app.post(
    "/v1/messages/count_tokens",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def count_anthropic_tokens(request: Request):
    """
    Count tokens for an Anthropic Messages API request.

    Uses the model's tokenizer for accurate counting.
    Claude Code calls this endpoint for token budgeting.
    Note: Don't parse via AnthropicRequest — count_tokens requests
    from Claude Code don't include max_tokens.
    """
    body = await request.json()

    engine = get_engine()
    tokenizer = engine.tokenizer

    total_tokens = 0

    # System message
    system = body.get("system", "")
    if isinstance(system, str) and system:
        total_tokens += len(tokenizer.encode(system))
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    total_tokens += len(tokenizer.encode(text))

    # Messages
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            if content:
                total_tokens += len(tokenizer.encode(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        total_tokens += len(tokenizer.encode(text))
                    # tool_use input
                    if block.get("input"):
                        total_tokens += len(
                            tokenizer.encode(json.dumps(block["input"]))
                        )
                    # tool_result content
                    sub_content = block.get("content", "")
                    if isinstance(sub_content, str) and sub_content:
                        total_tokens += len(tokenizer.encode(sub_content))
                    elif isinstance(sub_content, list):
                        for item in sub_content:
                            if isinstance(item, dict):
                                item_text = item.get("text", "")
                                if item_text:
                                    total_tokens += len(tokenizer.encode(item_text))

    # Tools
    for tool in body.get("tools", []):
        name = tool.get("name", "")
        if name:
            total_tokens += len(tokenizer.encode(name))
        desc = tool.get("description", "")
        if desc:
            total_tokens += len(tokenizer.encode(desc))
        if tool.get("input_schema"):
            total_tokens += len(tokenizer.encode(json.dumps(tool["input_schema"])))

    return {"input_tokens": total_tokens}


def _emit_content_pieces(
    pieces: list[tuple[str, str]],
    current_block_type: str | None,
    block_index: int,
) -> tuple[list[str], str | None, int]:
    """Emit Anthropic SSE events for content pieces from the think router.

    Handles block type transitions (thinking <-> text), emitting
    content_block_start/stop/delta events as needed.

    Args:
        pieces: List of (block_type, text) from StreamingThinkRouter
        current_block_type: Current open block type, or None
        block_index: Current block index

    Returns:
        Tuple of (events, updated_block_type, updated_block_index)
    """
    events = []
    for block_type, text in pieces:
        if block_type != current_block_type:
            # Close previous block if open
            if current_block_type is not None:
                events.append(
                    f"event: content_block_stop\ndata: "
                    f"{json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                )
                block_index += 1
            # Start new block
            current_block_type = block_type
            content_block = (
                {"type": block_type, "text": ""}
                if block_type == "text"
                else {"type": block_type, "thinking": ""}
            )
            events.append(
                f"event: content_block_start\ndata: "
                f"{json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': content_block})}\n\n"
            )
        # Emit delta
        delta_key = "thinking" if block_type == "thinking" else "text"
        delta_type = "thinking_delta" if block_type == "thinking" else "text_delta"
        delta_event = {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": delta_type, delta_key: text},
        }
        events.append(
            f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
        )
    return events, current_block_type, block_index


async def _stream_anthropic_messages(
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    anthropic_request: AnthropicRequest,
    metrics_tracker=None,
) -> AsyncIterator[str]:
    """
    Stream Anthropic Messages API SSE events.

    Converts OpenAI streaming chunks to Anthropic event format:
    message_start -> content_block_start -> content_block_delta* ->
    content_block_stop -> message_delta -> message_stop

    When a reasoning parser is active, emits a ``thinking`` content block
    (index 0) for reasoning tokens and a ``text`` content block (index 1)
    for the actual response, matching the Anthropic extended thinking format.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    start_time = time.perf_counter()
    result_label = "success"
    prompt_tokens = 0

    # Extract messages for engine
    messages, images, videos = extract_multimodal_content(
        openai_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )
    messages = _normalize_messages(messages)

    chat_kwargs = {
        "max_tokens": openai_request.max_tokens or _default_max_tokens,
        "temperature": openai_request.temperature,
        "top_p": openai_request.top_p,
        "top_k": openai_request.top_k or 0,
        "min_p": openai_request.min_p or 0.0,
        "presence_penalty": openai_request.presence_penalty or 0.0,
        "repetition_penalty": openai_request.repetition_penalty or 1.0,
    }

    if openai_request.tools and openai_request.tool_choice != "none":
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)

    # Wire constrained decoding if response_format was requested via the
    # Anthropic extension field and tools are not also requested.
    response_format = getattr(openai_request, "response_format", None)
    if response_format is not None and not (
        openai_request.tools and openai_request.tool_choice != "none"
    ):
        json_instruction = build_json_system_prompt(response_format)
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)
        tokenizer_obj = _get_engine_tokenizer(engine)
        if tokenizer_obj is not None:
            try:
                processor = build_json_logits_processor(response_format, tokenizer_obj)
            except Exception as exc:
                logger.warning(
                    "Failed to build JSON logits processor (Anthropic stream): %s",
                    exc,
                )
                processor = None
            if processor is not None:
                chat_kwargs["logits_processors"] = [processor]
                chat_kwargs["enable_thinking"] = False

    # Emit message_start
    message_start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": _model_name,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    use_reasoning = _reasoning_parser is not None and not chat_kwargs.get(
        "logits_processors"
    )

    if use_reasoning:
        _reasoning_parser.reset_state()

    # Block index tracking: with reasoning parser we use index 0 for
    # thinking and index 1 for text; without parser, index 0 for text.
    thinking_block_started = False
    text_block_started = False
    thinking_index = 0
    text_index = 1 if use_reasoning else 0

    if not use_reasoning:
        # No reasoning parser — start text block immediately
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        text_block_started = True

    # Stream content deltas
    accumulated_text = ""
    completion_tokens = 0

    # Tool call streaming suppression — prevents raw tool markup from leaking
    # as text_delta events. Mirrors the OpenAI streaming path logic.
    tool_parser = None
    tool_accumulated_text = ""
    tool_markup_possible = False
    tool_parser = _get_streaming_tool_parser(openai_request)

    try:
        async for output in engine.stream_chat(messages=messages, **chat_kwargs):
            if metrics_tracker is not None:
                metrics_tracker.observe_ttft()
            delta_text = output.new_text

            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens

            # Track token counts
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens

            if not delta_text:
                continue

            # Filter special tokens
            filtered = SPECIAL_TOKENS_PATTERN.sub("", delta_text)
            if not filtered:
                continue

            if not use_reasoning:
                # Simple path — no reasoning parsing
                accumulated_text += filtered
                content_to_emit = filtered

                # Filter tool call markup during streaming
                if tool_parser and content_to_emit:
                    if (
                        not tool_markup_possible
                        and not _streaming_tool_markup_possible(
                            tool_accumulated_text + content_to_emit
                        )
                    ):
                        tool_accumulated_text += content_to_emit
                    else:
                        if not tool_markup_possible:
                            tool_markup_possible = True
                        tool_previous = tool_accumulated_text
                        tool_accumulated_text += content_to_emit
                        tool_result = tool_parser.extract_tool_calls_streaming(
                            tool_previous, tool_accumulated_text, content_to_emit
                        )
                        if tool_result is None or "tool_calls" in tool_result:
                            # Inside tool markup or tool calls detected — suppress
                            continue
                        content_to_emit = tool_result.get("content", "")
                        if content_to_emit:
                            content_to_emit = _TOOL_MARKUP_PATTERN.sub(
                                "", content_to_emit
                            )
                        if not content_to_emit:
                            continue

                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content_to_emit}})}\n\n"
                continue

            # Reasoning parser path
            previous_text = accumulated_text
            accumulated_text += filtered
            delta_msg = _reasoning_parser.extract_reasoning_streaming(
                previous_text, accumulated_text, filtered
            )

            if delta_msg is None:
                continue

            if delta_msg.reasoning:
                if not thinking_block_started:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': thinking_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                    thinking_block_started = True
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': thinking_index, 'delta': {'type': 'thinking_delta', 'thinking': delta_msg.reasoning}})}\n\n"

            if delta_msg.content:
                content_to_emit = delta_msg.content

                # Filter tool call markup during streaming
                if tool_parser and content_to_emit:
                    if (
                        not tool_markup_possible
                        and not _streaming_tool_markup_possible(
                            tool_accumulated_text + content_to_emit
                        )
                    ):
                        tool_accumulated_text += content_to_emit
                    else:
                        if not tool_markup_possible:
                            tool_markup_possible = True
                        tool_previous = tool_accumulated_text
                        tool_accumulated_text += content_to_emit
                        tool_result = tool_parser.extract_tool_calls_streaming(
                            tool_previous, tool_accumulated_text, content_to_emit
                        )
                        if tool_result is None or "tool_calls" in tool_result:
                            # Inside tool markup or tool calls detected — suppress
                            continue
                        content_to_emit = tool_result.get("content", "")
                        if content_to_emit:
                            content_to_emit = _TOOL_MARKUP_PATTERN.sub(
                                "", content_to_emit
                            )
                        if not content_to_emit:
                            continue

                if thinking_block_started and not text_block_started:
                    # Close thinking block, open text block
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    text_block_started = True
                elif not text_block_started:
                    # No thinking was emitted, start text block at index 0
                    text_index = 0
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    text_block_started = True
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': content_to_emit}})}\n\n"

        # Close any open thinking block that was never followed by text
        if thinking_block_started and not text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
            # Emit empty text block so response always has text content
            text_index = thinking_index + 1
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            text_block_started = True

        # Check for tool calls in accumulated text
        _, tool_calls = _parse_tool_calls_with_parser(accumulated_text, openai_request)

        # Close text block
        if text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_index})}\n\n"

        # If there are tool calls, emit tool_use blocks
        next_index = (text_index + 1) if text_block_started else 0
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                tool_index = next_index + i
                try:
                    tool_input = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    tool_input = {}

                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': tool_index, 'content_block': {'type': 'tool_use', 'id': tc.id, 'name': tc.function.name, 'input': {}}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(tool_input)}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index})}\n\n"

        # Determine stop reason
        stop_reason = "tool_use" if tool_calls else "end_turn"

        # Emit message_delta with stop_reason and usage
        message_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": completion_tokens},
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

        # Log throughput
        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Anthropic messages (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        # Emit message_stop
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    except HTTPException as exc:
        result_label = _metrics_result_from_status(exc.status_code)
        raise
    except (asyncio.CancelledError, GeneratorExit):
        result_label = "cancelled"
        raise
    except Exception:
        result_label = "error"
        raise
    finally:
        if metrics_tracker is not None:
            metrics_tracker.finish(
                result=result_label,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )


# =============================================================================
# Streaming Helpers
# =============================================================================


async def stream_completion(
    engine: BaseEngine,
    prompt: str,
    request: CompletionRequest,
    repetition_penalty: float | None = None,
    metrics_tracker=None,
) -> AsyncIterator[str]:
    """Stream completion response."""
    result = "success"
    prompt_tokens = 0
    completion_tokens = 0
    generate_kwargs = {
        "prompt": prompt,
        "max_tokens": request.max_tokens or _default_max_tokens,
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "top_k": request.top_k or 0,
        "min_p": request.min_p or 0.0,
        "presence_penalty": request.presence_penalty or 0.0,
        "stop": request.stop,
    }
    if repetition_penalty is not None:
        generate_kwargs["repetition_penalty"] = repetition_penalty
    if request.specprefill is not None:
        generate_kwargs["specprefill"] = request.specprefill
    if request.specprefill_keep_pct is not None:
        generate_kwargs["specprefill_keep_pct"] = request.specprefill_keep_pct

    try:
        async for output in engine.stream_generate(**generate_kwargs):
            if metrics_tracker is not None:
                metrics_tracker.observe_ttft()
            prompt_tokens = (
                output.prompt_tokens
                if hasattr(output, "prompt_tokens")
                else prompt_tokens
            )
            completion_tokens = (
                output.completion_tokens
                if hasattr(output, "completion_tokens")
                else completion_tokens
            )
            data = {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": _model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": output.new_text,
                        "finish_reason": (
                            output.finish_reason if output.finished else None
                        ),
                    }
                ],
            }
            if output.finished:
                data["usage"] = get_usage(output).model_dump()
            yield f"data: {json.dumps(data)}\n\n"
    except HTTPException as exc:
        result = _metrics_result_from_status(exc.status_code)
        raise
    except (asyncio.CancelledError, GeneratorExit):
        result = "cancelled"
        raise
    except Exception:
        result = "error"
        raise
    finally:
        yield "data: [DONE]\n\n"
        if metrics_tracker is not None:
            metrics_tracker.finish(
                result=result,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )


async def stream_chat_completion(
    engine: BaseEngine,
    messages: list,
    request: ChatCompletionRequest,
    metrics_tracker=None,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    result_label = "success"

    # Check if we should include usage in the final chunk
    include_usage = request.stream_options and request.stream_options.include_usage

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=response_id,
        model=_model_name,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role="assistant"),
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Track if we need to add <think> prefix for thinking models (when no reasoning parser)
    # The template adds <think> to the prompt, so the model output starts inside the think block
    is_thinking_model = (
        "nemotron" in (engine.model_name or "").lower() and not _reasoning_parser
    )
    think_prefix_sent = False

    # Reset reasoning parser state for this stream
    if _reasoning_parser:
        _reasoning_parser.reset_state()

    # Track accumulated text for reasoning parser
    accumulated_text = ""

    # Track token counts for usage reporting
    prompt_tokens = 0
    completion_tokens = 0
    last_output = None

    # Response-format streaming filter — strip markdown code fences from
    # content when client asked for JSON. Non-streaming path strips fences
    # via ``parse_json_output``; without this, streaming clients see
    # ``"```json{...}```"`` instead of ``"{...}"`` for models that wrap
    # their structured output in markdown (e.g. Gemma 4).
    fence_stripper: StreamingJsonFenceStripper | None = None
    _rf = getattr(request, "response_format", None)
    _rf_type = None
    if _rf is not None:
        _rf_type = getattr(_rf, "type", None)
        if _rf_type is None and isinstance(_rf, dict):
            _rf_type = _rf.get("type")
    if _rf_type in ("json_object", "json_schema"):
        fence_stripper = StreamingJsonFenceStripper()

    # Tool call streaming state
    tool_parser = None
    tool_accumulated_text = ""
    tool_calls_detected = False
    tool_markup_possible = False  # Fast path: skip parsing until markers appear
    tool_parser = _get_streaming_tool_parser(request)

    try:
        # Stream content
        async for output in engine.stream_chat(messages=messages, **kwargs):
            if metrics_tracker is not None:
                metrics_tracker.observe_ttft()
            delta_text = output.new_text
            last_output = output

            # Track token counts from output (updated each chunk)
            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens

            # Use reasoning parser if enabled (skip when enable_thinking=False)
            if (
                _reasoning_parser
                and delta_text
                and request.enable_thinking is not False
            ):
                previous_text = accumulated_text
                accumulated_text += delta_text
                delta_msg = _reasoning_parser.extract_reasoning_streaming(
                    previous_text, accumulated_text, delta_text
                )

                if delta_msg is None:
                    # Skip this chunk (e.g., <think> token itself)
                    continue

                content = delta_msg.content
                reasoning = delta_msg.reasoning

                # Some models (e.g. MiniMax) wrap tool calls in <think>
                # blocks, so reasoning parser captures tool call XML as
                # reasoning while content stays None.  Redirect reasoning
                # to the content stream so the tool parser can handle it.
                if tool_parser and reasoning and not content:
                    _check = tool_accumulated_text + reasoning
                    if (
                        "<minimax:tool_call>" in _check
                        or "<tool_call>" in _check
                        or '<invoke name="' in _check
                    ):
                        content = reasoning
                        reasoning = None

                # Tool call parsing on content portion
                if tool_parser and content:
                    if (
                        not tool_markup_possible
                        and not _streaming_tool_markup_possible(
                            tool_accumulated_text + content
                        )
                    ):
                        tool_accumulated_text += content
                        # Suppress whitespace-only content when tools are active;
                        # avoids emitting stray newlines before tool call XML.
                        if not content.strip():
                            continue
                    else:
                        if not tool_markup_possible:
                            tool_markup_possible = True
                        tool_previous = tool_accumulated_text
                        tool_accumulated_text += content
                        tool_result = tool_parser.extract_tool_calls_streaming(
                            tool_previous, tool_accumulated_text, content
                        )

                        if tool_result is None:
                            # Inside tool markup - suppress content output
                            if reasoning:
                                # Still emit reasoning while buffering tool call
                                chunk = ChatCompletionChunk(
                                    id=response_id,
                                    model=_model_name,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            delta=ChatCompletionChunkDelta(
                                                reasoning=reasoning,
                                            ),
                                            finish_reason=None,
                                        )
                                    ],
                                    usage=None,
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"
                            continue

                        if "tool_calls" in tool_result:
                            # Emit structured tool calls
                            tool_calls_detected = True
                            # Coerce arguments against tool schemas
                            tools = (
                                request.model_dump().get("tools")
                                if request and request.tools
                                else None
                            )
                            if tools:
                                for tc in tool_result["tool_calls"]:
                                    fn = tc.get("function", {})
                                    if "arguments" in fn and "name" in fn:
                                        fn["arguments"] = _coerce_tool_arguments(
                                            fn["arguments"], fn["name"], tools
                                        )
                            chunk = ChatCompletionChunk(
                                id=response_id,
                                model=_model_name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        delta=ChatCompletionChunkDelta(
                                            tool_calls=tool_result["tool_calls"],
                                            reasoning=reasoning,
                                        ),
                                        finish_reason=(
                                            "tool_calls" if output.finished else None
                                        ),
                                    )
                                ],
                                usage=get_usage(output) if output.finished else None,
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            continue

                        # Normal content from tool parser
                        content = tool_result.get("content", "")
                        # Strip any leaked tool markup tags
                        if content:
                            content = _TOOL_MARKUP_PATTERN.sub("", content)

                # Strip markdown code fences when response_format is set.
                if fence_stripper is not None and not tool_calls_detected:
                    content = fence_stripper.feed(content) if content else ""
                    if output.finished:
                        flush = fence_stripper.finalize()
                        if flush:
                            content = content + flush

                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                content=content if content else None,
                                reasoning=reasoning,
                            ),
                            finish_reason=(
                                "tool_calls"
                                if (output.finished and tool_calls_detected)
                                else (output.finish_reason if output.finished else None)
                            ),
                        )
                    ],
                    usage=get_usage(output) if output.finished else None,
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            else:
                # Standard path without reasoning parsing
                content = delta_text

                # Filter special tokens that may leak into streaming output
                if content:
                    content = SPECIAL_TOKENS_PATTERN.sub("", content)

                # Add <think> prefix on first content chunk for thinking models
                if is_thinking_model and not think_prefix_sent and content:
                    content = "<think>" + content
                    think_prefix_sent = True

                # Tool call streaming parsing
                if tool_parser and delta_text:
                    # Fast path: skip full parsing until likely tool markup appears.
                    # This preserves the cheap path for ordinary text while still
                    # allowing generic streaming tool parsing when no explicit
                    # parser flags are configured.
                    if (
                        not tool_markup_possible
                        and not _streaming_tool_markup_possible(
                            tool_accumulated_text + delta_text
                        )
                    ):
                        tool_accumulated_text += delta_text
                        # No tool markup yet, fall through to normal chunk emission
                    else:
                        if not tool_markup_possible:
                            tool_markup_possible = True
                        tool_previous = tool_accumulated_text
                        tool_accumulated_text += delta_text
                        tool_result = tool_parser.extract_tool_calls_streaming(
                            tool_previous, tool_accumulated_text, delta_text
                        )

                        if tool_result is None:
                            # Inside tool markup - suppress output
                            continue

                        if "tool_calls" in tool_result:
                            # Emit structured tool calls
                            tool_calls_detected = True
                            # Coerce arguments against tool schemas
                            tools = (
                                request.model_dump().get("tools")
                                if request and request.tools
                                else None
                            )
                            if tools:
                                for tc in tool_result["tool_calls"]:
                                    fn = tc.get("function", {})
                                    if "arguments" in fn and "name" in fn:
                                        fn["arguments"] = _coerce_tool_arguments(
                                            fn["arguments"], fn["name"], tools
                                        )
                            chunk = ChatCompletionChunk(
                                id=response_id,
                                model=_model_name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        delta=ChatCompletionChunkDelta(
                                            tool_calls=tool_result["tool_calls"]
                                        ),
                                        finish_reason=(
                                            "tool_calls" if output.finished else None
                                        ),
                                    )
                                ],
                                usage=get_usage(output) if output.finished else None,
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            continue

                        # Normal content from tool parser
                        content = tool_result.get("content", "")
                        # Strip any leaked tool markup tags
                        if content:
                            content = _TOOL_MARKUP_PATTERN.sub("", content)

                # Strip markdown code fences when response_format is set.
                if fence_stripper is not None and not tool_calls_detected:
                    content = fence_stripper.feed(content) if content else ""
                    if output.finished:
                        flush = fence_stripper.finalize()
                        if flush:
                            content = content + flush

                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                content=content if content else None
                            ),
                            finish_reason=(
                                "tool_calls"
                                if (output.finished and tool_calls_detected)
                                else (output.finish_reason if output.finished else None)
                            ),
                        )
                    ],
                    usage=get_usage(output) if output.finished else None,
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # Fallback: if tool parser accumulated text but never emitted tool_calls
        # (e.g., </tool_call> never arrived, or <function= block still incomplete)
        if (
            tool_parser
            and tool_accumulated_text
            and not tool_calls_detected
            and _streaming_tool_markup_possible(tool_accumulated_text)
        ):
            final_parse_result = tool_parser.extract_tool_calls(tool_accumulated_text)
            if final_parse_result.tools_called:
                tools = (
                    request.model_dump().get("tools")
                    if request and request.tools
                    else None
                )
                tool_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                tool_calls=[
                                    {
                                        "index": i,
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tc["name"],
                                            "arguments": _coerce_tool_arguments(
                                                tc["arguments"], tc["name"], tools
                                            ),
                                        },
                                    }
                                    for i, tc in enumerate(
                                        final_parse_result.tool_calls
                                    )
                                ]
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
                yield f"data: {tool_chunk.model_dump_json()}\n\n"

        # Safety-net validation: if response_format was requested, verify the
        # accumulated output still parses.  When constrained decoding is active
        # this should always succeed; if it fails we log loudly (error) so we
        # notice grammar-integration regressions.  When constrained decoding was
        # *not* active (optional dep missing, incompatible tokenizer, combined
        # with tools), we log at warning level only — the prompt-only path is
        # best-effort.
        if (
            getattr(request, "response_format", None) is not None
            and not tool_calls_detected
        ):
            try:
                _, _parsed, _is_valid, _err = parse_json_output(
                    accumulated_text, request.response_format
                )
                if not _is_valid:
                    # Determine whether constrained decoding was wired up.  We
                    # passed the processor through ``kwargs`` so its presence is
                    # the signal.
                    has_constrained = any(
                        p.__class__.__name__ == "JSONSchemaLogitsProcessor"
                        for p in (kwargs.get("logits_processors") or [])
                    )
                    if has_constrained:
                        logger.error(
                            "Streaming constrained decoding produced invalid JSON: %s",
                            _err,
                        )
                    else:
                        logger.warning("Streaming JSON validation failed: %s", _err)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Streaming JSON validation raised: %s", exc)

        # Log throughput
        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Chat completion (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        # Send final chunk with usage if requested
        if include_usage:
            usage_chunk = ChatCompletionChunk(
                id=response_id,
                model=_model_name,
                choices=[],  # Empty choices for usage-only chunk
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"
    except HTTPException as exc:
        result_label = _metrics_result_from_status(exc.status_code)
        raise
    except (asyncio.CancelledError, GeneratorExit):
        result_label = "cancelled"
        raise
    except Exception:
        result_label = "error"
        raise
    finally:
        if metrics_tracker is not None:
            metrics_tracker.finish(
                result=result_label,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )


# =============================================================================
# MCP Initialization
# =============================================================================


async def init_mcp(config_path: str):
    """Initialize MCP manager from config file."""
    global _mcp_manager, _mcp_executor

    try:
        from vllm_mlx.mcp import (
            MCPClientManager,
            ToolExecutor,
            ToolSandbox,
            load_mcp_config,
        )

        config = load_mcp_config(config_path)
        _mcp_manager = MCPClientManager(config)
        await _mcp_manager.start()

        sandbox = ToolSandbox(allowed_high_risk_tools=config.allowed_high_risk_tools)
        _mcp_executor = ToolExecutor(_mcp_manager, sandbox=sandbox)

        logger.info(f"MCP initialized with {len(_mcp_manager.get_all_tools())} tools")

    except ImportError:
        logger.error("MCP SDK not installed. Install with: pip install mcp")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize MCP: {e}")
        raise


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the server."""
    parser = create_parser()
    args = parser.parse_args()

    # Set global configuration
    global _api_key, _default_timeout, _rate_limiter, _metrics_enabled
    global _default_temperature, _default_top_p
    global _max_audio_upload_bytes, _max_tts_input_chars
    _api_key = args.api_key
    _default_timeout = args.timeout
    _metrics_enabled = args.enable_metrics
    _metrics.configure(enabled=args.enable_metrics)
    if args.default_temperature is not None:
        _default_temperature = args.default_temperature
    if args.default_top_p is not None:
        _default_top_p = args.default_top_p
    _max_audio_upload_bytes = args.max_audio_upload_mb * 1024 * 1024
    _max_tts_input_chars = args.max_tts_input_chars

    # Configure rate limiter
    if args.rate_limit > 0:
        _rate_limiter = RateLimiter(requests_per_minute=args.rate_limit, enabled=True)
        logger.info(
            f"Rate limiting enabled: {args.rate_limit} requests/minute per client"
        )

    # Security summary at startup
    logger.info("=" * 60)
    logger.info("SECURITY CONFIGURATION")
    logger.info("=" * 60)
    if _api_key:
        logger.info("  Authentication: ENABLED (API key required)")
    else:
        logger.warning("  Authentication: DISABLED - Use --api-key to enable")
    if args.rate_limit > 0:
        logger.info(f"  Rate limiting: ENABLED ({args.rate_limit} req/min)")
    else:
        logger.warning("  Rate limiting: DISABLED - Use --rate-limit to enable")
    logger.info(f"  Request timeout: {args.timeout}s")
    if args.enable_metrics:
        logger.info("  Metrics: ENABLED (/metrics, unauthenticated)")
    else:
        logger.info("  Metrics: DISABLED - Use --enable-metrics to expose /metrics")
    if args.trust_remote_code:
        logger.warning("  Remote code loading: ENABLED (--trust-remote-code)")
    else:
        logger.info("  Remote code loading: DISABLED (default)")
    logger.info(
        f"  Audio upload limit: {args.max_audio_upload_mb} MiB, "
        f"TTS input limit: {args.max_tts_input_chars} chars"
    )
    logger.info("=" * 60)

    # Set MCP config for lifespan
    if args.mcp_config:
        os.environ["VLLM_MLX_MCP_CONFIG"] = args.mcp_config

    # Initialize reasoning parser if specified
    if args.reasoning_parser:
        global _reasoning_parser
        from .reasoning import get_parser

        parser_cls = get_parser(args.reasoning_parser)
        _reasoning_parser = parser_cls()
        logger.info(f"Reasoning parser enabled: {args.reasoning_parser}")

    # Pre-load embedding model if specified
    load_embedding_model(args.embedding_model, lock=True)

    # Load model before starting server
    load_model(
        args.model,
        use_batching=args.continuous_batching,
        max_tokens=args.max_tokens,
        force_mllm=args.mllm,
        trust_remote_code=args.trust_remote_code,
    )

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)


def create_parser() -> argparse.ArgumentParser:
    """Create the standalone server CLI parser."""
    parser = argparse.ArgumentParser(
        description="vllm-mlx OpenAI-compatible server for LLM and MLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with simple mode (maximum throughput)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Start with continuous batching (for multiple users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Model to load (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force loading as MLLM (multimodal language model)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow HuggingFace remote code execution during model/tokenizer loading",
    )
    parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Enable continuous batching for multiple concurrent users",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Default max tokens for generation",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (if not set, no auth required)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Default request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Expose Prometheus metrics on /metrics (disabled by default)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            f"Options: {', '.join(reasoning_choices)}."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Pre-load an embedding model at startup (e.g. mlx-community/all-MiniLM-L6-v2-4bit)",
    )
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Default temperature for generation when not specified in request",
    )
    parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Default top_p for generation when not specified in request",
    )
    parser.add_argument(
        "--max-audio-upload-mb",
        type=int,
        default=DEFAULT_MAX_AUDIO_UPLOAD_MB,
        help="Maximum size of uploaded audio files in MiB (default: 25)",
    )
    parser.add_argument(
        "--max-tts-input-chars",
        type=int,
        default=DEFAULT_MAX_TTS_INPUT_CHARS,
        help="Maximum number of characters accepted by /v1/audio/speech (default: 4096)",
    )
    return parser


if __name__ == "__main__":
    main()
