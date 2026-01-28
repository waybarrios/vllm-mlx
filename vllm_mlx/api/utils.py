# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for text processing and model detection.
"""

import re
from typing import List, Tuple

from .models import Message

# =============================================================================
# Special Token Patterns
# =============================================================================

# Pattern to match special tokens that should be removed from output
# Keeps <think>...</think> blocks intact for reasoning models
SPECIAL_TOKENS_PATTERN = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|"
    r"<\|end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|"
    r"</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]"
)


def clean_output_text(text: str) -> str:
    """
    Clean model output by removing special tokens.

    Keeps <think>...</think> blocks intact for reasoning models.
    Adds opening <think> tag if missing (happens when thinking is enabled
    in the prompt template but the tag is part of the prompt, not output).

    Args:
        text: Raw model output

    Returns:
        Cleaned text with special tokens removed
    """
    if not text:
        return text
    text = SPECIAL_TOKENS_PATTERN.sub("", text)
    text = text.strip()

    # Add opening <think> tag if response has closing but not opening
    # This happens when enable_thinking=True in the chat template
    if "</think>" in text and not text.lstrip().startswith("<think>"):
        text = "<think>" + text

    return text


# =============================================================================
# Model Detection
# =============================================================================

# Patterns that indicate a multimodal language model (MLLM/VLM)
MLLM_PATTERNS = [
    "-VL-",
    "-VL/",
    "VL-",  # Qwen-VL, Qwen2-VL, Qwen3-VL, etc.
    "llava",
    "LLaVA",  # LLaVA models
    "idefics",
    "Idefics",  # Idefics models
    "paligemma",
    "PaliGemma",  # PaliGemma
    "gemma-3",
    "gemma3",  # Gemma 3 (multimodal)
    "medgemma",
    "MedGemma",  # MedGemma (medical multimodal with SigLIP vision encoder)
    "pixtral",
    "Pixtral",  # Pixtral
    "molmo",
    "Molmo",  # Molmo
    "phi3-vision",
    "phi-3-vision",  # Phi-3 Vision
    "cogvlm",
    "CogVLM",  # CogVLM
    "internvl",
    "InternVL",  # InternVL
    "deepseek-vl",
    "DeepSeek-VL",  # DeepSeek-VL
]


def is_mllm_model(model_name: str) -> bool:
    """
    Check if model name indicates a multimodal language model.

    Args:
        model_name: HuggingFace model name or local path

    Returns:
        True if model is detected as MLLM/VLM
    """
    model_lower = model_name.lower()
    for pattern in MLLM_PATTERNS:
        if pattern.lower() in model_lower:
            return True
    return False


# Backwards compatibility alias
is_vlm_model = is_mllm_model


# =============================================================================
# Multimodal Content Extraction
# =============================================================================


def extract_multimodal_content(
    messages: List[Message],
) -> Tuple[List[dict], List[str], List[str]]:
    """
    Extract text content, images, and videos from OpenAI-format messages.

    Handles:
    - Simple text messages
    - Multimodal messages with images/videos
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool")

    Args:
        messages: List of Message objects

    Returns:
        Tuple of (processed_messages, images, videos)
        - processed_messages: List of {"role": str, "content": str}
        - images: List of image URLs/paths/base64
        - videos: List of video URLs/paths/base64
    """
    processed_messages = []
    images = []
    videos = []

    for msg in messages:
        # Handle both dict and Pydantic model messages
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content")
        else:
            role = msg.role
            content = msg.content

        # Handle tool response messages (role="tool")
        if role == "tool":
            # Format tool result as assistant context
            if isinstance(msg, dict):
                tool_call_id = msg.get("tool_call_id", "") or ""
            else:
                tool_call_id = getattr(msg, "tool_call_id", None) or ""
            tool_content = content if content else ""
            processed_messages.append(
                {
                    "role": "user",  # mlx-lm expects user/assistant roles
                    "content": f"[Tool Result ({tool_call_id})]: {tool_content}",
                }
            )
            continue

        # Handle assistant messages with tool_calls
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        else:
            tool_calls = getattr(msg, "tool_calls", None)

        if role == "assistant" and tool_calls:
            # Format tool calls as part of the assistant message
            tool_calls_text = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    tool_calls_text.append(f"[Calling tool: {name}({args})]")

            text = content if content else ""
            if tool_calls_text:
                text = (text + "\n" if text else "") + "\n".join(tool_calls_text)

            processed_messages.append({"role": role, "content": text})
            continue

        # Handle None content
        if content is None:
            processed_messages.append({"role": role, "content": ""})
            continue

        if isinstance(content, str):
            # Simple text message
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multimodal message - extract text and media
            text_parts = []
            for item in content:
                # Handle both Pydantic models and dicts
                if hasattr(item, "model_dump"):
                    item = item.model_dump()
                elif hasattr(item, "dict"):
                    item = item.dict()

                item_type = item.get("type", "")

                if item_type == "text":
                    text_parts.append(item.get("text", ""))

                elif item_type == "image_url":
                    img_url = item.get("image_url", {})
                    if isinstance(img_url, str):
                        images.append(img_url)
                    elif isinstance(img_url, dict):
                        images.append(img_url.get("url", ""))

                elif item_type == "image":
                    images.append(item.get("image", item.get("url", "")))

                elif item_type == "video":
                    videos.append(item.get("video", item.get("url", "")))

                elif item_type == "video_url":
                    vid_url = item.get("video_url", {})
                    if isinstance(vid_url, str):
                        videos.append(vid_url)
                    elif isinstance(vid_url, dict):
                        videos.append(vid_url.get("url", ""))

            # Combine text parts
            combined_text = "\n".join(text_parts) if text_parts else ""
            processed_messages.append({"role": role, "content": combined_text})
        else:
            # Unknown format, try to convert
            processed_messages.append({"role": role, "content": str(content)})

    return processed_messages, images, videos
