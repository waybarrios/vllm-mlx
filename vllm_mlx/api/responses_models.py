# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for the OpenAI-compatible Responses API.

This intentionally implements the subset needed for local coding-agent
workflows: text messages, function tools, function call outputs, and SSE
streaming events. The object and event shapes follow the conventions used by
OpenAI's gpt-oss reference server and llama.cpp's OpenAI-compatible server.
"""

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class ResponseTextFormat(BaseModel):
    """Output text format configuration."""

    type: Literal["text", "json_object"] = "text"


class ResponseTextConfig(BaseModel):
    """Text output configuration."""

    format: ResponseTextFormat = Field(default_factory=ResponseTextFormat)


class ResponseReasoningConfig(BaseModel):
    """Reasoning configuration."""

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None


class ResponseTextContentPart(BaseModel):
    """A text content part for message items."""

    type: Literal["text", "input_text", "output_text"] = "output_text"
    text: str
    annotations: list[dict] = Field(default_factory=list)
    logprobs: list[dict] = Field(default_factory=list)


class ResponseReasoningTextPart(BaseModel):
    """A reasoning text content part."""

    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


class ResponseReasoningSummaryTextPart(BaseModel):
    """A reasoning summary item."""

    type: Literal["summary_text"] = "summary_text"
    text: str


class ResponseMessageItem(BaseModel):
    """A Responses API message item."""

    id: str | None = None
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant", "developer"] = "assistant"
    content: str | list[ResponseTextContentPart] = Field(default_factory=list)
    status: Literal["in_progress", "completed", "incomplete"] | None = "completed"


class ResponseReasoningItem(BaseModel):
    """A reasoning output item."""

    id: str | None = None
    type: Literal["reasoning"] = "reasoning"
    summary: list[ResponseReasoningSummaryTextPart] = Field(default_factory=list)
    content: list[ResponseReasoningTextPart] = Field(default_factory=list)
    status: Literal["in_progress", "completed", "incomplete"] | None = "completed"


class ResponseFunctionCallItem(BaseModel):
    """A function call output item."""

    id: str | None = None
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str
    status: Literal["in_progress", "completed", "incomplete"] = "completed"


class ResponseFunctionCallOutputItem(BaseModel):
    """A tool result item passed back into a later request."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


class ResponseFunctionTool(BaseModel):
    """A function tool definition."""

    type: Literal["function"] = "function"
    name: str
    description: str | None = ""
    parameters: dict = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    strict: bool = False


class ResponsesInputTokenDetails(BaseModel):
    """Input token breakdown."""

    cached_tokens: int = 0


class ResponsesOutputTokenDetails(BaseModel):
    """Output token breakdown."""

    reasoning_tokens: int = 0


class ResponsesUsage(BaseModel):
    """Responses API token usage."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: ResponsesInputTokenDetails = Field(
        default_factory=ResponsesInputTokenDetails
    )
    output_tokens_details: ResponsesOutputTokenDetails = Field(
        default_factory=ResponsesOutputTokenDetails
    )


class ResponseError(BaseModel):
    """Error payload."""

    code: str
    message: str


class ResponseIncompleteDetails(BaseModel):
    """Incomplete response details."""

    reason: str


class ResponsesRequest(BaseModel):
    """Request payload for /v1/responses."""

    model: str
    input: (
        str
        | list[
            ResponseMessageItem
            | ResponseReasoningItem
            | ResponseFunctionCallItem
            | ResponseFunctionCallOutputItem
            | dict
        ]
    )
    instructions: str | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    tools: list[ResponseFunctionTool | dict] = Field(default_factory=list)
    tool_choice: str | dict | None = "auto"
    parallel_tool_calls: bool = True
    previous_response_id: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    metadata: dict = Field(default_factory=dict)
    text: ResponseTextConfig = Field(default_factory=ResponseTextConfig)
    reasoning: ResponseReasoningConfig | None = None
    store: bool = True
    truncation: str = "disabled"
    user: str | None = None


class ResponseObject(BaseModel):
    """Response object for /v1/responses."""

    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: Literal["completed", "failed", "incomplete", "in_progress"] = "completed"
    background: bool = False
    error: ResponseError | None = None
    incomplete_details: ResponseIncompleteDetails | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: dict = Field(default_factory=dict)
    model: str
    output: list[
        ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem
    ] = Field(default_factory=list)
    parallel_tool_calls: bool = True
    previous_response_id: str | None = None
    text: ResponseTextConfig = Field(default_factory=ResponseTextConfig)
    tool_choice: str | dict | None = "auto"
    tools: list[ResponseFunctionTool | dict] = Field(default_factory=list)
    top_p: float = 1.0
    temperature: float | None = None
    truncation: str = "disabled"
    usage: ResponsesUsage | None = None
    user: str | None = None
    store: bool = True

    @computed_field
    @property
    def output_text(self) -> str:
        """Concatenate assistant text content into the convenience field."""
        text_parts: list[str] = []
        for item in self.output:
            if not isinstance(item, ResponseMessageItem):
                continue
            if isinstance(item.content, str):
                text_parts.append(item.content)
                continue
            for part in item.content:
                if part.type == "output_text":
                    text_parts.append(part.text)
        return "".join(text_parts)


class ResponsesEventBase(BaseModel):
    """Base event fields."""

    sequence_number: int


class ResponseCreatedEvent(ResponsesEventBase):
    type: Literal["response.created"] = "response.created"
    response: ResponseObject


class ResponseInProgressEvent(ResponsesEventBase):
    type: Literal["response.in_progress"] = "response.in_progress"
    response: ResponseObject


class ResponseCompletedEvent(ResponsesEventBase):
    type: Literal["response.completed"] = "response.completed"
    response: ResponseObject


class ResponseOutputItemAddedEvent(ResponsesEventBase):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int
    item: ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem


class ResponseOutputItemDoneEvent(ResponsesEventBase):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int
    item: ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem


class ResponseContentPartAddedEvent(ResponsesEventBase):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: str
    output_index: int
    content_index: int
    part: ResponseTextContentPart | ResponseReasoningTextPart


class ResponseContentPartDoneEvent(ResponsesEventBase):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str
    output_index: int
    content_index: int
    part: ResponseTextContentPart | ResponseReasoningTextPart


class ResponseOutputTextDeltaEvent(ResponsesEventBase):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str
    output_index: int
    content_index: int
    delta: str
    logprobs: list[dict] = Field(default_factory=list)


class ResponseOutputTextDoneEvent(ResponsesEventBase):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str
    output_index: int
    content_index: int
    text: str
    logprobs: list[dict] = Field(default_factory=list)


class ResponseReasoningTextDeltaEvent(ResponsesEventBase):
    type: Literal["response.reasoning_text.delta"] = "response.reasoning_text.delta"
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseReasoningTextDoneEvent(ResponsesEventBase):
    type: Literal["response.reasoning_text.done"] = "response.reasoning_text.done"
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseFunctionCallArgumentsDeltaEvent(ResponsesEventBase):
    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    item_id: str
    output_index: int
    delta: str
