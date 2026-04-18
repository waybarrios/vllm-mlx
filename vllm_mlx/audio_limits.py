# SPDX-License-Identifier: Apache-2.0
"""Resource limits for optional audio endpoints."""

import os
import tempfile
from pathlib import Path
from typing import Protocol

from fastapi import HTTPException

DEFAULT_MAX_AUDIO_UPLOAD_MB = 25
DEFAULT_MAX_AUDIO_UPLOAD_BYTES = DEFAULT_MAX_AUDIO_UPLOAD_MB * 1024 * 1024
DEFAULT_MAX_TTS_INPUT_CHARS = 4096
UPLOAD_CHUNK_SIZE = 1024 * 1024


class AsyncReadableUpload(Protocol):
    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...


async def save_upload_with_limit(
    file: AsyncReadableUpload,
    *,
    max_bytes: int,
    default_suffix: str = ".wav",
    chunk_size: int = UPLOAD_CHUNK_SIZE,
) -> str:
    """
    Stream an uploaded file to disk while enforcing a hard byte limit.

    This prevents large audio uploads from being buffered entirely in memory.
    """
    suffix = Path(file.filename or "").suffix or default_suffix
    total_bytes = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        try:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Audio upload too large: {total_bytes} bytes exceeds "
                            f"the configured limit of {max_bytes} bytes."
                        ),
                    )
                tmp.write(chunk)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    return tmp_path


def validate_tts_input_length(text: str, *, max_chars: int) -> None:
    """Reject oversized TTS requests before synthesis starts."""
    if len(text) > max_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                f"TTS input too long: {len(text)} characters exceeds the configured "
                f"limit of {max_chars} characters."
            ),
        )
