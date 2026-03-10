# SPDX-License-Identifier: Apache-2.0
"""Tests for video support in MLLM chat/stream_chat."""

import pytest

from vllm_mlx.models.mllm import (
    MLXMultimodalLM,
    is_base64_video,
    process_video_input,
    smart_nframes,
    FRAME_FACTOR,
    MIN_FRAMES,
    MAX_FRAMES,
    DEFAULT_FPS,
)


class TestSmartNframes:
    """Verify frame count alignment and clamping."""

    def test_basic_calculation(self):
        # 300 frames at 30fps = 10s video, at 2fps target = 20 frames
        result = smart_nframes(300, 30.0, target_fps=2.0)
        assert result == 20
        assert result % FRAME_FACTOR == 0

    def test_clamps_to_min(self):
        # Very short video: 6 frames at 30fps
        result = smart_nframes(6, 30.0, target_fps=2.0)
        assert result >= MIN_FRAMES
        assert result % FRAME_FACTOR == 0

    def test_clamps_to_max(self):
        # Very long video: 100000 frames
        result = smart_nframes(100000, 30.0, target_fps=2.0, max_frames=64)
        assert result <= 64
        assert result % FRAME_FACTOR == 0

    def test_result_always_even(self):
        for total in [5, 7, 11, 13, 100, 999]:
            result = smart_nframes(total, 30.0)
            assert result % FRAME_FACTOR == 0, f"Odd frame count {result} for total={total}"


class TestVideoUrlParsing:
    """Verify video_url content type extraction from OpenAI messages."""

    def _extract_video_inputs(self, messages):
        """Simulate the first-pass video extraction from chat()."""
        _msg_video_inputs = {}
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")
                if item_type == "video":
                    _msg_video_inputs.setdefault(msg_idx, []).append(
                        item.get("video", item.get("url", ""))
                    )
                elif item_type == "video_url":
                    vid_url = item.get("video_url", {})
                    if isinstance(vid_url, str):
                        _msg_video_inputs.setdefault(msg_idx, []).append(vid_url)
                    elif isinstance(vid_url, dict):
                        url = vid_url.get("url", "")
                        if url:
                            _msg_video_inputs.setdefault(msg_idx, []).append(url)
        return _msg_video_inputs

    def test_video_url_dict_format(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}},
                    {"type": "text", "text": "Describe this video"},
                ],
            }
        ]
        result = self._extract_video_inputs(messages)
        assert 0 in result
        assert result[0] == ["https://example.com/video.mp4"]

    def test_video_url_string_format(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": "https://example.com/video.mp4"},
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]
        result = self._extract_video_inputs(messages)
        assert result[0] == ["https://example.com/video.mp4"]

    def test_video_type(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/path/to/video.mp4"},
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]
        result = self._extract_video_inputs(messages)
        assert result[0] == ["/path/to/video.mp4"]

    def test_no_video(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                ],
            }
        ]
        result = self._extract_video_inputs(messages)
        assert len(result) == 0

    def test_mixed_media(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                    {"type": "video_url", "video_url": {"url": "https://example.com/vid.mp4"}},
                    {"type": "text", "text": "Compare"},
                ],
            }
        ]
        result = self._extract_video_inputs(messages)
        # Only video extracted, not image
        assert result[0] == ["https://example.com/vid.mp4"]


class TestTranslateMessages:
    """Verify OpenAI format to process_vision_info format translation."""

    def _make_model(self):
        """Create an unloaded model instance for testing translation."""
        model = MLXMultimodalLM.__new__(MLXMultimodalLM)
        model._loaded = False
        model._video_native = True
        return model

    def test_text_only_passthrough(self):
        model = self._make_model()
        messages = [{"role": "user", "content": "Hello"}]
        result = model._translate_messages_for_native_video(messages, 2.0, 128)
        assert result[0]["content"] == "Hello"

    def test_video_url_translated(self):
        import tempfile
        import os

        # Create a temp file to act as a "video"
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            video_path = f.name

        try:
            model = self._make_model()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": "Describe"},
                    ],
                }
            ]
            result = model._translate_messages_for_native_video(messages, 2.0, 128)
            content = result[0]["content"]

            # Should have video and text items
            types = [item["type"] for item in content]
            assert "video" in types
            assert "text" in types

            # Video item should have fps and max_frames
            video_item = next(i for i in content if i["type"] == "video")
            assert video_item["fps"] == 2.0
            assert video_item["max_frames"] == 128
        finally:
            os.unlink(video_path)


class TestIsBase64Video:
    def test_detects_base64_video(self):
        assert is_base64_video("data:video/mp4;base64,AAAA") is True

    def test_rejects_non_video(self):
        assert is_base64_video("data:image/jpeg;base64,AAAA") is False
        assert is_base64_video("https://example.com/video.mp4") is False
