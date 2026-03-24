# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine MLLM + MTP per-request routing."""


def test_has_media_content_text_only():
    from vllm_mlx.engine.batched import _has_media_content

    assert _has_media_content([{"role": "user", "content": "Hello"}]) is False


def test_has_media_content_with_image():
    from vllm_mlx.engine.batched import _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        }
    ]
    assert _has_media_content(messages) is True


def test_has_media_content_with_video():
    from vllm_mlx.engine.batched import _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "file:///tmp/v.mp4"}}
            ],
        }
    ]
    assert _has_media_content(messages) is True


def test_has_media_content_empty():
    from vllm_mlx.engine.batched import _has_media_content

    assert _has_media_content([]) is False


def test_has_media_content_string_content():
    """String content (not list) should return False."""
    from vllm_mlx.engine.batched import _has_media_content

    assert _has_media_content([{"role": "user", "content": "Just text"}]) is False


def test_has_media_content_audio():
    from vllm_mlx.engine.batched import _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": "data:audio/wav;base64,..."},
                }
            ],
        }
    ]
    assert _has_media_content(messages) is True


def test_has_media_content_multi_turn():
    """Media in earlier turns should still be detected."""
    from vllm_mlx.engine.batched import _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        },
        {"role": "assistant", "content": "I see an image."},
        {"role": "user", "content": "Tell me more about it."},
    ]
    assert _has_media_content(messages) is True


def test_has_media_content_text_list():
    """List content with only text parts should return False."""
    from vllm_mlx.engine.batched import _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }
    ]
    assert _has_media_content(messages) is False
