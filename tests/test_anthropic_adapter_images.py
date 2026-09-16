"""Anthropic image blocks must reach the model.

Before this, an image block parsed cleanly and was then dropped, so a vision
request returned HTTP 200 with a confident text-only answer. vLLM's own
Anthropic surface converts image blocks, so this was also a parity gap.
"""

import base64

import pytest

from vllm_mlx.api.anthropic_adapter import _convert_message
from vllm_mlx.api.anthropic_models import AnthropicMessage

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()


def parts_of(msg):
    """Normalize content parts to dicts (pydantic coerces them to ContentPart)."""
    out = []
    for p in msg.content:
        out.append(p if isinstance(p, dict) else p.model_dump(exclude_none=True))
    return out


def url_of(part):
    u = part["image_url"]
    return u if isinstance(u, str) else u["url"]


def _img(media_type="image/png", data=PNG):
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _msg(role, content):
    return AnthropicMessage(role=role, content=content)


def test_image_block_becomes_an_image_url_part():
    out = _convert_message(_msg("user", [_img()]))
    assert len(out) == 1
    parts = parts_of(out[0])
    assert isinstance(parts, list), "content must be multimodal, not a string"
    assert parts[0]["type"] == "image_url"
    assert url_of(parts[0]) == f"data:image/png;base64,{PNG}"


def test_media_type_is_preserved_not_assumed_png():
    parts = parts_of(_convert_message(_msg("user", [_img("image/jpeg")]))[0])
    assert url_of(parts[0]).startswith("data:image/jpeg;base64,")


def test_text_and_image_interleaving_is_preserved():
    # For document work, "here is the page" vs "now answer this" ordering changes
    # the task, so order must survive the conversion.
    out = _convert_message(
        _msg(
            "user",
            [
                {"type": "text", "text": "before"},
                _img(),
                {"type": "text", "text": "after"},
            ],
        )
    )
    parts = parts_of(out[0])
    kinds = [p["type"] for p in parts]
    assert kinds == ["text", "image_url", "text"]
    assert parts[0]["text"] == "before"
    assert parts[2]["text"] == "after"


def test_url_source_is_supported():
    out = _convert_message(
        _msg(
            "user",
            [{"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}],
        )
    )
    assert url_of(parts_of(out[0])[0]) == "https://x/y.png"


def test_unknown_source_type_raises_rather_than_dropping():
    # Silent dropping is the exact failure this branch exists to end.
    with pytest.raises(ValueError, match="unsupported image source type"):
        _convert_message(_msg("user", [{"type": "image", "source": {"type": "magic"}}]))


def test_empty_image_data_raises():
    with pytest.raises(ValueError, match="no data"):
        _convert_message(_msg("user", [_img(data="")]))


def test_text_only_path_is_unchanged():
    # The hot path must stay a plain string, not become a one-element list.
    out = _convert_message(_msg("user", [{"type": "text", "text": "hello"}]))
    assert out[0].content == "hello"
    assert _convert_message(_msg("user", "plain"))[0].content == "plain"


def test_assistant_images_are_not_dropped():
    out = _convert_message(_msg("assistant", [_img()]))
    assert isinstance(out[0].content, list)
    assert parts_of(out[0])[0]["type"] == "image_url"


def test_image_alongside_tool_result_keeps_both():
    out = _convert_message(
        _msg(
            "user",
            [
                _img(),
                {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
            ],
        )
    )
    roles = [m.role for m in out]
    assert "user" in roles and "tool" in roles
    user_msg = next(m for m in out if m.role == "user")
    assert parts_of(user_msg)[0]["type"] == "image_url"


def test_assistant_image_with_tool_use_preserves_both():
    msg = AnthropicMessage(
        role="assistant",
        content=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": PNG,
                },
            },
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "zoom",
                "input": {"region": "top-left"},
            },
        ],
    )
    (converted,) = _convert_message(msg)
    assert converted.role == "assistant"
    assert (
        converted.tool_calls and converted.tool_calls[0]["function"]["name"] == "zoom"
    )
    kinds = [part["type"] for part in parts_of(converted)]
    assert "image_url" in kinds
