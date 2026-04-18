# SPDX-License-Identifier: Apache-2.0
"""Tests for audio endpoint resource limits."""

from pathlib import Path

import pytest
from fastapi import HTTPException

from vllm_mlx.audio_limits import (
    DEFAULT_MAX_AUDIO_UPLOAD_MB,
    DEFAULT_MAX_TTS_INPUT_CHARS,
    save_upload_with_limit,
    validate_tts_input_length,
)


class FakeUpload:
    def __init__(self, chunks: list[bytes], filename: str = "audio.wav"):
        self._chunks = list(chunks)
        self.filename = filename

    async def read(self, _size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class TestAudioUploadLimits:
    @pytest.mark.asyncio
    async def test_save_upload_with_limit_writes_file(self):
        upload = FakeUpload([b"a" * 8, b"b" * 4])

        path = await save_upload_with_limit(upload, max_bytes=32)

        try:
            assert Path(path).read_bytes() == b"a" * 8 + b"b" * 4
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_save_upload_with_limit_rejects_oversize_and_cleans_up(self):
        upload = FakeUpload([b"a" * 16, b"b" * 16, b"c"])

        with pytest.raises(HTTPException) as exc_info:
            await save_upload_with_limit(upload, max_bytes=32)

        assert exc_info.value.status_code == 413
        assert "Audio upload too large" in exc_info.value.detail


class TestTTSInputLimits:
    def test_validate_tts_input_length_accepts_short_text(self):
        validate_tts_input_length("hello", max_chars=16)

    def test_validate_tts_input_length_rejects_oversized_text(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_tts_input_length("x" * 17, max_chars=16)

        assert exc_info.value.status_code == 413
        assert "TTS input too long" in exc_info.value.detail


class TestAudioLimitParsers:
    def test_top_level_cli_exposes_audio_limit_flags(self):
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "serve",
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "--max-audio-upload-mb",
                "12",
                "--max-tts-input-chars",
                "2048",
            ]
        )

        assert args.max_audio_upload_mb == 12
        assert args.max_tts_input_chars == 2048

    def test_standalone_server_parser_defaults(self):
        from vllm_mlx.server import create_parser

        parser = create_parser()
        args = parser.parse_args(
            ["--model", "mlx-community/Llama-3.2-3B-Instruct-4bit"]
        )

        assert args.max_audio_upload_mb == DEFAULT_MAX_AUDIO_UPLOAD_MB
        assert args.max_tts_input_chars == DEFAULT_MAX_TTS_INPUT_CHARS
