# SPDX-License-Identifier: Apache-2.0
"""
API models, utilities, and tool calling support for vllm-mlx.

This module provides shared components used by the server:
- Pydantic models for OpenAI-compatible API
- Utility functions for text processing and model detection
- Tool calling parsing and conversion
"""

from .models import (
    # Content types
    ImageUrl,
    VideoUrl,
    AudioUrl,
    ContentPart,
    Message,
    # Tool calling
    FunctionCall,
    ToolCall,
    ToolDefinition,
    # Structured output
    ResponseFormat,
    ResponseFormatJsonSchema,
    # Chat requests/responses
    ChatCompletionRequest,
    ChatCompletionChoice,
    ChatCompletionResponse,
    AssistantMessage,
    # Completion requests/responses
    CompletionRequest,
    CompletionChoice,
    CompletionResponse,
    # Common
    Usage,
    ModelInfo,
    ModelsResponse,
    # MCP
    MCPToolInfo,
    MCPToolsResponse,
    MCPServerInfo,
    MCPServersResponse,
    MCPExecuteRequest,
    MCPExecuteResponse,
    # Audio
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioSpeechRequest,
    AudioSeparationRequest,
    # Embeddings
    EmbeddingRequest,
    EmbeddingData,
    EmbeddingUsage,
    EmbeddingResponse,
)

from .utils import (
    clean_output_text,
    is_mllm_model,
    is_vlm_model,
    extract_multimodal_content,
    MLLM_PATTERNS,
    SPECIAL_TOKENS_PATTERN,
)

from .tool_calling import (
    parse_tool_calls,
    convert_tools_for_template,
    # Structured output
    parse_json_output,
    validate_json_schema,
    extract_json_from_text,
    build_json_system_prompt,
)

__all__ = [
    # Models
    "ImageUrl",
    "VideoUrl",
    "AudioUrl",
    "ContentPart",
    "Message",
    "FunctionCall",
    "ToolCall",
    "ToolDefinition",
    "ResponseFormat",
    "ResponseFormatJsonSchema",
    "ChatCompletionRequest",
    "ChatCompletionChoice",
    "ChatCompletionResponse",
    "AssistantMessage",
    "CompletionRequest",
    "CompletionChoice",
    "CompletionResponse",
    "Usage",
    "ModelInfo",
    "ModelsResponse",
    "MCPToolInfo",
    "MCPToolsResponse",
    "MCPServerInfo",
    "MCPServersResponse",
    "MCPExecuteRequest",
    "MCPExecuteResponse",
    # Audio
    "AudioTranscriptionRequest",
    "AudioTranscriptionResponse",
    "AudioSpeechRequest",
    "AudioSeparationRequest",
    # Embeddings
    "EmbeddingRequest",
    "EmbeddingData",
    "EmbeddingUsage",
    "EmbeddingResponse",
    # Utils
    "clean_output_text",
    "is_mllm_model",
    "is_vlm_model",
    "extract_multimodal_content",
    "MLLM_PATTERNS",
    "SPECIAL_TOKENS_PATTERN",
    # Tool calling
    "parse_tool_calls",
    "convert_tools_for_template",
    # Structured output
    "parse_json_output",
    "validate_json_schema",
    "extract_json_from_text",
    "build_json_system_prompt",
]
