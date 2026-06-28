# SPDX-License-Identifier: Apache-2.0
"""
MLX Multimodal Language Model (MLLM) wrapper.

This module provides a wrapper around mlx-vlm for multimodal inference,
supporting vision, audio, and video understanding on Apple Silicon.

Features:
- OpenAI-compatible API format for images and video
- Smart video frame extraction with configurable FPS
- Base64 and URL image support
- Streaming generation
- MLLM KV cache for repeated image/video+prompt combinations
"""

import atexit
import base64
from importlib.metadata import PackageNotFoundError, version
import json
import ipaddress
import logging
import math
import os
import socket
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests

from vllm_mlx.mllm_cache import MLLMPrefixCacheManager

logger = logging.getLogger(__name__)


class TempFileManager:
    """Thread-safe manager for tracking and cleaning up temporary files."""

    def __init__(self):
        self._files: set[str] = set()
        self._lock = threading.Lock()
        atexit.register(self.cleanup_all)

    def register(self, path: str) -> str:
        """Register a temp file for tracking. Returns the path for convenience."""
        with self._lock:
            self._files.add(path)
        return path

    def cleanup(self, path: str) -> bool:
        """Clean up a specific temp file. Returns True if successful."""
        with self._lock:
            if path in self._files:
                self._files.discard(path)
        try:
            if os.path.exists(path):
                os.unlink(path)
                logger.debug(f"Cleaned up temp file: {path}")
                return True
        except OSError as e:
            logger.warning(f"Failed to clean up temp file {path}: {e}")
        return False

    def cleanup_all(self) -> int:
        """Clean up all tracked temp files. Returns count of cleaned files."""
        with self._lock:
            files_to_clean = list(self._files)
            self._files.clear()

        cleaned = 0
        for path in files_to_clean:
            try:
                if os.path.exists(path):
                    os.unlink(path)
                    cleaned += 1
            except OSError:
                pass

        if cleaned:
            logger.info(f"Cleaned up {cleaned} temp files")
        return cleaned


# Global temp file manager
_temp_manager = TempFileManager()


def cleanup_temp_file(path: str) -> bool:
    """Clean up a specific temporary file."""
    return _temp_manager.cleanup(path)


def cleanup_all_temp_files() -> int:
    """Clean up all tracked temporary files. Returns count of cleaned files."""
    return _temp_manager.cleanup_all()


# Video processing constants
FRAME_FACTOR = 2  # Frames must be divisible by this
DEFAULT_FPS = 2.0  # Default frames per second for video
MIN_FRAMES = 4
MAX_FRAMES = 128  # Practical limit for most MLLMs
IMAGE_FACTOR = 28  # For smart resize

# Security: File size limits (in bytes)
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB max for images
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB max for videos
MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100 MB max for audio
MAX_BASE64_IMAGE_LENGTH = 30 * 1024 * 1024  # 30 MB base64 string (~22 MB decoded)
MAX_BASE64_VIDEO_LENGTH = 700 * 1024 * 1024  # 700 MB base64 string (~500 MB decoded)
MAX_BASE64_AUDIO_LENGTH = 140 * 1024 * 1024  # 140 MB base64 string (~100 MB decoded)


class FileSizeExceededError(Exception):
    """Raised when a downloaded file exceeds the size limit."""

    pass


class UnsafeRemoteURLError(ValueError):
    """Raised when a remote media URL targets an unsafe destination."""

    def __init__(
        self,
        message: str,
        *,
        public_message: str = "Remote media URL is not allowed",
    ) -> None:
        super().__init__(message)
        self.public_message = public_message


def _normalize_content_part(item: object) -> object:
    """Convert Pydantic content parts into plain Python objects."""
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    if hasattr(item, "dict"):
        return {k: v for k, v in item.dict().items() if v is not None}
    return item


def _extract_media_url(item: dict, item_type: str) -> str:
    if item_type == "image_url":
        media_value = item.get("image_url", {})
    elif item_type == "video_url":
        media_value = item.get("video_url", {})
    elif item_type == "audio_url":
        media_value = item.get("audio_url", {})
    elif item_type in {"image", "video", "audio"}:
        media_value = item.get(item_type, item.get("url", ""))
    else:
        return ""

    if isinstance(media_value, dict):
        media_value = media_value.get("url", "")
    return media_value if isinstance(media_value, str) else ""


def _text_content_part(text: str) -> dict[str, str]:
    return {"type": "text", "text": text, "content": text}


def _append_text_content_part(
    built_parts: list[dict[str, str]], text_parts: list[str], text: str
) -> None:
    if not text:
        return
    built_parts.append(_text_content_part(text))
    text_parts.append(text)


def _build_string_mllm_message_content(content: str, role: str) -> tuple[object, bool]:
    if not content:
        return "", False
    if role == "assistant":
        return content, True
    return [_text_content_part(content)], True


def _append_ordered_mllm_content_part(
    raw_item: object,
    *,
    built_parts: list[dict[str, str]],
    text_parts: list[str],
    all_image_urls: list[str],
    video_frame_count: int,
) -> int:
    item = _normalize_content_part(raw_item)
    if isinstance(item, str):
        _append_text_content_part(built_parts, text_parts, item)
        return video_frame_count

    if not isinstance(item, dict):
        return video_frame_count

    item_type = item.get("type", "")
    if item_type in {"text", "input_text"}:
        _append_text_content_part(
            built_parts, text_parts, item.get("text", "") or item.get("content", "")
        )
    elif item_type in {"image_url", "image"}:
        media_url = _extract_media_url(item, item_type)
        if media_url:
            all_image_urls.append(media_url)
        built_parts.append({"type": "image"})
    elif item_type in {"audio_url", "audio"}:
        # Audio inputs are collected once by _collect_audio_inputs before
        # message reconstruction; this helper only preserves placeholder order.
        built_parts.append({"type": "audio"})
    elif item_type in {"video", "video_url"}:
        # Native video models bypass this helper. For fallback frame extraction,
        # preserve the video position by inserting that message's frames here.
        built_parts.extend({"type": "image"} for _ in range(video_frame_count))
        return 0
    return video_frame_count


def _build_ordered_mllm_message_content(
    content: object,
    *,
    role: str,
    all_image_urls: list[str],
    video_frame_count: int = 0,
) -> tuple[object, bool]:
    """Build template content while preserving OpenAI media/text part order."""
    if isinstance(content, str):
        return _build_string_mllm_message_content(content, role)

    if not isinstance(content, list):
        return "", False

    built_parts: list[dict[str, str]] = []
    text_parts: list[str] = []
    remaining_video_frames = video_frame_count

    for raw_item in content:
        remaining_video_frames = _append_ordered_mllm_content_part(
            raw_item,
            built_parts=built_parts,
            text_parts=text_parts,
            all_image_urls=all_image_urls,
            video_frame_count=remaining_video_frames,
        )

    if role == "assistant":
        text = "".join(text_parts)
        return text, bool(text)

    return built_parts, bool(built_parts)


def _build_mllm_chat_messages(
    messages: list[dict],
    *,
    all_image_urls: list[str],
    video_frame_counts: dict[int, int],
) -> list[dict]:
    """Build chat-template messages without reordering multimodal content parts."""
    chat_messages: list[dict] = []
    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        if not isinstance(role, str):
            role = str(role)

        content, has_content = _build_ordered_mllm_message_content(
            msg.get("content", ""),
            role=role,
            all_image_urls=all_image_urls,
            video_frame_count=video_frame_counts.get(msg_idx, 0),
        )
        if has_content:
            chat_messages.append({"role": role, "content": content})
    return chat_messages


@dataclass
class MultimodalInput:
    """Input for multimodal generation."""

    prompt: str
    images: list[str] = field(default_factory=list)  # Paths, URLs, or base64
    videos: list[str] = field(default_factory=list)  # Paths
    audio: list[str] = field(default_factory=list)  # Paths


@dataclass
class MLLMOutput:
    """Output from multimodal language model."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    mtp_drafts: int = 0
    mtp_accepted: int = 0


def load_gemma4_assistant_drafter(model_path: str):
    """Load a Gemma 4 assistant drafter for mlx-vlm speculative decoding."""
    try:
        import mlx.core as mx
        from mlx_vlm.speculative.drafters import gemma4_assistant as arch
    except ImportError as exc:
        raise ImportError(
            "Gemma 4 assistant drafter support requires an mlx-vlm build that "
            "provides mlx_vlm.speculative.drafters.gemma4_assistant."
        ) from exc

    try:
        mlx_vlm_version = version("mlx-vlm")
    except PackageNotFoundError:
        mlx_vlm_version = "unknown"
    logger.info(
        "Loading Gemma 4 assistant drafter from %s using mlx-vlm %s",
        model_path,
        mlx_vlm_version,
    )

    path = Path(model_path)
    config_path = path / "config.json"
    weight_paths = sorted(path.glob("*.safetensors"))
    if not config_path.exists():
        raise FileNotFoundError(f"Gemma 4 assistant config not found: {config_path}")
    if not weight_paths:
        raise FileNotFoundError(f"Gemma 4 assistant weights not found: {path}")

    config = arch.ModelConfig.from_dict(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    model = arch.Model(config)
    weights = {}
    for weight_path in weight_paths:
        weights.update(mx.load(str(weight_path)))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    model.eval()
    return model


_DRAFT_KWARG_NAMES = ("draft_model", "draft_kind", "draft_block_size")


def _count_draft_tokens(draft_tokens) -> int:
    """Best-effort drafted-token count for an mlx-vlm drafter output."""
    shape = getattr(draft_tokens, "shape", None)
    if shape:
        try:
            return max(int(shape[-1]), 0)
        except (TypeError, ValueError):
            pass
    try:
        return max(len(draft_tokens), 0)
    except TypeError:
        return 0


def _install_draft_metrics_hooks(draft_model) -> None:
    """Record actual drafted token counts from mlx-vlm assistant drafters."""
    if draft_model is None or getattr(draft_model, "_vllm_mlx_metrics_hooked", False):
        return

    if not hasattr(draft_model, "_vllm_mlx_draft_counts"):
        draft_model._vllm_mlx_draft_counts = []

    draft_block = getattr(draft_model, "draft_block", None)
    if callable(draft_block):

        def draft_block_with_metrics(*args, **kwargs):
            draft_tokens = draft_block(*args, **kwargs)
            draft_model._vllm_mlx_draft_counts.append(_count_draft_tokens(draft_tokens))
            return draft_tokens

        draft_model.draft_block = draft_block_with_metrics

    reset = getattr(draft_model, "reset", None)
    if callable(reset):

        def reset_with_metrics(*args, **kwargs):
            draft_model._vllm_mlx_draft_counts = []
            return reset(*args, **kwargs)

        draft_model.reset = reset_with_metrics

    draft_model._vllm_mlx_metrics_hooked = True


def is_base64_image(s: str) -> bool:
    """Check if string is base64-encoded image data."""
    return s.startswith("data:image/") or (
        len(s) > 100 and not s.startswith(("http://", "https://", "/"))
    )


def is_url(s: str) -> bool:
    """Check if string is a URL."""
    return s.startswith(("http://", "https://"))


def is_base64_video(s: str) -> bool:
    """Check if string is base64-encoded video data."""
    return s.startswith("data:video/")


def is_base64_audio(s: str) -> bool:
    """Check if string is base64-encoded audio data."""
    return s.startswith("data:audio/")


def decode_base64_image(
    base64_string: str, max_length: int = MAX_BASE64_IMAGE_LENGTH
) -> bytes:
    """
    Decode base64 image to bytes.

    Args:
        base64_string: Base64 encoded image (optionally with data URL prefix)
        max_length: Maximum allowed length of base64 string

    Returns:
        Decoded image bytes

    Raises:
        FileSizeExceededError: If base64 string exceeds max_length
    """
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 image data exceeds maximum size: {len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )

    # Handle data URL format: data:image/jpeg;base64,/9j/4AAQ...
    if base64_string.startswith("data:"):
        # Extract the base64 part after the comma
        _, data = base64_string.split(",", 1)
        return base64.b64decode(data)
    return base64.b64decode(base64_string)


def _validate_url_safety(url: str) -> None:
    """Reject remote URLs that target local or private network resources."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeRemoteURLError(
            f"Unsupported remote media URL scheme: {parsed.scheme or '<missing>'}"
        )

    hostname = parsed.hostname
    if not hostname:
        raise UnsafeRemoteURLError("Remote media URL must include a hostname")

    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise UnsafeRemoteURLError(
            f"Remote media URL targets a blocked host: {hostname}"
        )

    try:
        resolved_ips = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            addrinfo = socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise UnsafeRemoteURLError(
                f"Failed to resolve remote media host {hostname}: {exc}"
            ) from exc
        resolved_ips = [ipaddress.ip_address(info[4][0]) for info in addrinfo]

    blocked_ips = [str(ip) for ip in resolved_ips if not ip.is_global]
    if blocked_ips:
        raise UnsafeRemoteURLError(
            f"Remote media URL resolves to blocked address(es): {', '.join(sorted(set(blocked_ips)))}"
        )


def _request_with_safe_redirects(
    method: str,
    url: str,
    *,
    timeout: int,
    headers: dict[str, str],
    stream: bool = False,
    max_redirects: int = 5,
):
    """Issue a requests call while validating every redirect target."""
    current_url = url
    for _ in range(max_redirects + 1):
        _validate_url_safety(current_url)
        response = requests.request(
            method,
            current_url,
            timeout=timeout,
            headers=headers,
            allow_redirects=False,
            verify=True,
            stream=stream,
        )
        if not response.is_redirect and not response.is_permanent_redirect:
            return response

        location = response.headers.get("location")
        response.close()
        if not location:
            raise UnsafeRemoteURLError(
                f"Remote media URL redirect missing Location header: {current_url}"
            )
        current_url = urljoin(current_url, location)

    raise UnsafeRemoteURLError(
        f"Remote media URL exceeded redirect limit ({max_redirects}): {url}"
    )


def download_image(url: str, timeout: int = 30, max_size: int = MAX_IMAGE_SIZE) -> str:
    """
    Download image from URL and return local path.

    Args:
        url: Image URL
        timeout: Download timeout in seconds
        max_size: Maximum allowed file size in bytes

    Returns:
        Local file path to downloaded image

    Raises:
        FileSizeExceededError: If image exceeds max_size
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    # First, make a HEAD request to check Content-Length
    try:
        head_response = _request_with_safe_redirects(
            "HEAD", url, timeout=timeout, headers=headers
        )
        content_length = head_response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise FileSizeExceededError(
                f"Image at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
                f"{max_size / 1024 / 1024:.1f} MB limit"
            )
    except requests.RequestException:
        # HEAD request failed, proceed with GET and check during download
        pass

    response = _request_with_safe_redirects(
        "GET", url, timeout=timeout, headers=headers, stream=True
    )
    response.raise_for_status()

    # Check Content-Length header from GET response
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise FileSizeExceededError(
            f"Image at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
            f"{max_size / 1024 / 1024:.1f} MB limit"
        )

    # Determine extension from content type or URL
    content_type = response.headers.get("content-type", "")
    if "jpeg" in content_type or "jpg" in content_type:
        ext = ".jpg"
    elif "png" in content_type:
        ext = ".png"
    elif "gif" in content_type:
        ext = ".gif"
    elif "webp" in content_type:
        ext = ".webp"
    else:
        # Try to get from URL
        path = urlparse(response.url).path
        ext = Path(path).suffix or ".jpg"

    # Save to temp file with size checking during download
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    downloaded_size = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileSizeExceededError(
                    f"Image at {url} exceeds maximum size during download: "
                    f"{downloaded_size / 1024 / 1024:.1f} MB > {max_size / 1024 / 1024:.1f} MB limit"
                )
            temp_file.write(chunk)
        temp_file.close()
    except FileSizeExceededError:
        raise
    except Exception:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise

    return _temp_manager.register(temp_file.name)


_VIDEO_EXT_MAP: dict[str, str] = {
    "mp4": ".mp4",
    "webm": ".webm",
    "avi": ".avi",
    "mov": ".mov",
    "quicktime": ".mov",
    "mkv": ".mkv",
}

_AUDIO_EXT_MAP: dict[str, str] = {
    "wav": ".wav",
    "mpeg": ".mp3",
    "mp3": ".mp3",
    "flac": ".flac",
    "ogg": ".ogg",
    "webm": ".webm",
    "mp4": ".m4a",
    "m4a": ".m4a",
    "aac": ".m4a",
}


def _download_media(
    url: str,
    media_type: str,
    ext_map: dict[str, str],
    default_ext: str,
    timeout: int,
    max_size: int,
) -> str:
    """Download media from URL, enforce size limits, and return a local temp path."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    logger.info(f"Downloading {media_type} from: {url}")

    try:
        head_response = _request_with_safe_redirects(
            "HEAD", url, timeout=timeout, headers=headers
        )
        content_length = head_response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise FileSizeExceededError(
                f"{media_type.capitalize()} at {url} exceeds maximum size: "
                f"{int(content_length) / 1024 / 1024:.1f} MB > {max_size / 1024 / 1024:.1f} MB limit"
            )
    except requests.RequestException:
        pass

    response = _request_with_safe_redirects(
        "GET", url, timeout=timeout, headers=headers, stream=True
    )
    response.raise_for_status()

    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise FileSizeExceededError(
            f"{media_type.capitalize()} at {url} exceeds maximum size: "
            f"{int(content_length) / 1024 / 1024:.1f} MB > {max_size / 1024 / 1024:.1f} MB limit"
        )

    content_type = response.headers.get("content-type", "").lower()
    ext = default_ext
    for key, mapped_ext in ext_map.items():
        if key in content_type:
            ext = mapped_ext
            break
    else:
        path_ext = Path(urlparse(response.url).path).suffix
        if path_ext:
            ext = path_ext

    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    downloaded_size = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileSizeExceededError(
                    f"{media_type.capitalize()} at {url} exceeds maximum size during download: "
                    f"{downloaded_size / 1024 / 1024:.1f} MB > {max_size / 1024 / 1024:.1f} MB limit"
                )
            temp_file.write(chunk)
        temp_file.close()
    except FileSizeExceededError:
        raise
    except Exception:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise

    file_size = Path(temp_file.name).stat().st_size
    logger.info(
        f"{media_type.capitalize()} downloaded: {temp_file.name} ({file_size / 1024 / 1024:.1f} MB)"
    )

    return _temp_manager.register(temp_file.name)


def download_video(url: str, timeout: int = 120, max_size: int = MAX_VIDEO_SIZE) -> str:
    """Download video from URL and return local path."""
    return _download_media(url, "video", _VIDEO_EXT_MAP, ".mp4", timeout, max_size)


def download_audio(url: str, timeout: int = 120, max_size: int = MAX_AUDIO_SIZE) -> str:
    """Download audio from URL and return local path."""
    return _download_media(url, "audio", _AUDIO_EXT_MAP, ".wav", timeout, max_size)


def decode_base64_video(
    base64_string: str, max_length: int = MAX_BASE64_VIDEO_LENGTH
) -> str:
    """
    Decode base64 video to temp file and return path.

    Supports format: data:video/mp4;base64,AAAA...

    Args:
        base64_string: Base64-encoded video with data URL prefix
        max_length: Maximum allowed length of base64 string

    Returns:
        Local file path to decoded video

    Raises:
        FileSizeExceededError: If base64 string exceeds max_length
    """
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 video data exceeds maximum size: {len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )

    # Extract format and data
    if base64_string.startswith("data:video/"):
        # Format: data:video/mp4;base64,AAAA...
        header, data = base64_string.split(",", 1)
        # Extract extension from header (e.g., "data:video/mp4;base64" -> "mp4")
        format_part = header.split(";")[0]  # "data:video/mp4"
        ext = "." + format_part.split("/")[-1]  # ".mp4"
    else:
        # Assume mp4 if no header
        data = base64_string
        ext = ".mp4"

    # Decode, save, and register for cleanup
    video_bytes = base64.b64decode(data)
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(video_bytes)
    temp_file.close()

    logger.info(
        f"Base64 video decoded: {temp_file.name} ({len(video_bytes) / 1024 / 1024:.1f} MB)"
    )

    return _temp_manager.register(temp_file.name)


def decode_base64_audio(
    base64_string: str, max_length: int = MAX_BASE64_AUDIO_LENGTH
) -> str:
    """
    Decode base64 audio to temp file and return path.

    Supports format: data:audio/wav;base64,AAAA...
    """
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 audio data exceeds maximum size: {len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )

    if base64_string.startswith("data:audio/"):
        header, data = base64_string.split(",", 1)
        format_part = header.split(";")[0]
        ext = "." + format_part.split("/")[-1]
    else:
        data = base64_string
        ext = ".wav"

    audio_bytes = base64.b64decode(data)
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(audio_bytes)
    temp_file.close()
    return _temp_manager.register(temp_file.name)


def process_video_input(video: str | dict) -> str:
    """
    Process video input in various formats and return local path.

    Supports:
    - URL (http/https)
    - Base64 encoded string (data:video/mp4;base64,...)
    - OpenAI format dict: {"url": "..."} or {"url": "data:video/...;base64,..."}

    Args:
        video: Video input in any supported format

    Returns:
        Local file path to video
    """
    # Handle dict format (OpenAI style)
    if isinstance(video, dict):
        url = video.get("url", video.get("video_url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        video = url

    if not video:
        raise ValueError("Empty video input")

    # Check if it's a URL
    if is_url(video):
        return download_video(video)

    # Check if it's base64
    if is_base64_video(video):
        return decode_base64_video(video)

    raise ValueError(
        "Unsupported video input. Only http(s) URLs and data:video base64 payloads are allowed."
    )


def process_audio_input(audio: str | dict) -> str:
    """
    Process audio input in various formats and return local path.

    Supports:
    - Local file path
    - URL (http/https)
    - Base64 encoded string (data:audio/wav;base64,...)
    - OpenAI format dict: {"url": "..."} or {"audio_url": {"url": "..."}}
    """
    if isinstance(audio, dict):
        url = audio.get("url", audio.get("audio_url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        audio = url

    if not audio:
        raise ValueError("Empty audio input")

    if is_base64_audio(audio):
        return decode_base64_audio(audio)

    if is_url(audio):
        return download_audio(audio)

    if len(audio) < 4096 and Path(audio).exists():
        return audio

    raise ValueError(f"Cannot process audio: {audio[:50]}...")


def _video_has_audio_track(video_path: str) -> bool:
    """Return True if ffprobe finds an audio stream in the video."""
    import shutil
    import subprocess

    if not shutil.which("ffprobe"):
        return True  # assume yes; extraction will fail loudly if not
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-loglevel",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            timeout=30,
            text=True,
        )
        return bool(r.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return True


def _model_has_sound_encoder(model) -> bool:
    """Whether a loaded model exposes a usable sound encoder.

    Uses ``getattr(..., None) is not None`` rather than ``hasattr`` so model
    wrappers that declare ``sound_encoder`` in ``__init__`` but leave it as
    ``None`` until the first encoder pass are correctly treated as not yet
    enabled. A bare ``hasattr`` check would spuriously enable A/V fusion
    against a missing encoder and crash the processor downstream.
    """
    return getattr(model, "sound_encoder", None) is not None


def extract_audio_from_video(video_path: str) -> str | None:
    """Extract the audio track from a video file as 16 kHz mono WAV.

    Returns the path to the WAV (registered with the temp manager so it's
    cleaned up automatically), or None if the video has no audio or ffmpeg
    is unavailable.
    """
    import os
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        logger.warning(
            "ffmpeg not found; cannot fuse audio from video_url. "
            "Install ffmpeg to enable A/V fusion on omni models."
        )
        return None
    if not _video_has_audio_track(video_path):
        return None

    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="vllmmlx_va_")
    os.close(fd)
    try:
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                out_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=600,
        )
        if r.returncode != 0 or os.path.getsize(out_path) == 0:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            return None
        return _temp_manager.register(out_path)
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning(f"Audio extraction from video failed: {e}")
        try:
            os.unlink(out_path)
        except OSError:
            pass
        return None


# Cache for base64 images to avoid re-saving the same image
_base64_image_cache: dict[str, str] = {}  # hash -> temp file path


def save_base64_image(base64_string: str) -> str:
    """Save base64 image to temp file and return path. Caches identical images."""
    import hashlib

    # Hash the FULL base64 string — not just a prefix.
    # Using only the first 1000 chars caused cache collisions between
    # different images with identical JPEG headers (e.g. invoices from
    # the same PDF renderer), returning a previous request's image.
    image_hash = hashlib.sha256(base64_string.encode()).hexdigest()

    # Return cached path if available and file still exists
    if image_hash in _base64_image_cache:
        cached_path = _base64_image_cache[image_hash]
        if Path(cached_path).exists():
            return cached_path

    image_bytes = decode_base64_image(base64_string)

    # Detect format from magic bytes
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    elif image_bytes[:2] == b"\xff\xd8":
        ext = ".jpg"
    elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        ext = ".gif"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        ext = ".webp"
    else:
        ext = ".jpg"  # Default

    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(image_bytes)
    temp_file.close()

    path = _temp_manager.register(temp_file.name)
    _base64_image_cache[image_hash] = path
    return path


def process_image_input(image: str | dict) -> str:
    """
    Process image input in various formats and return local path.

    Supports:
    - URL (http/https)
    - Base64 encoded string
    - OpenAI format dict: {"url": "..."} or {"url": "data:image/...;base64,..."}
    """
    # Handle dict format (OpenAI style)
    if isinstance(image, dict):
        url = image.get("url", image.get("image_url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        image = url

    if not image:
        raise ValueError("Empty image input")

    # Check if it's base64 FIRST (before Path.exists() which fails on long strings)
    if is_base64_image(image):
        return save_base64_image(image)

    # Check if it's a URL
    if is_url(image):
        return download_image(image)

    raise ValueError(
        "Unsupported image input. Only http(s) URLs and data:image base64 payloads are allowed."
    )


def round_by_factor(x: int, factor: int) -> int:
    """Round to nearest multiple of factor."""
    return round(x / factor) * factor


def ceil_by_factor(x: float, factor: int) -> int:
    """Ceiling to next multiple of factor."""
    return math.ceil(x / factor) * factor


def floor_by_factor(x: float, factor: int) -> int:
    """Floor to previous multiple of factor."""
    return math.floor(x / factor) * factor


def smart_nframes(
    total_frames: int,
    video_fps: float,
    target_fps: float = DEFAULT_FPS,
    min_frames: int = MIN_FRAMES,
    max_frames: int = MAX_FRAMES,
) -> int:
    """
    Calculate optimal number of frames to extract from video.

    Uses smart sampling based on video length and target FPS.
    """
    # Calculate duration-based frame count
    duration = total_frames / video_fps if video_fps > 0 else 0
    nframes = duration * target_fps

    # Clamp to min/max
    nframes = max(min_frames, min(nframes, max_frames, total_frames))

    # Round to factor
    nframes = max(FRAME_FACTOR, floor_by_factor(nframes, FRAME_FACTOR))

    return int(nframes)


def extract_video_frames_smart(
    video_path: str,
    fps: float = DEFAULT_FPS,
    max_frames: int = MAX_FRAMES,
    resize: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """
    Extract frames from video with smart sampling.

    Args:
        video_path: Path to video file
        fps: Target frames per second (default: 2.0)
        max_frames: Maximum frames to extract
        resize: Optional (width, height) to resize frames

    Returns:
        List of frame arrays (RGB format)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for video processing")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Calculate number of frames to extract
    nframes = smart_nframes(
        total_frames=total_frames,
        video_fps=video_fps,
        target_fps=fps,
        max_frames=max_frames,
    )

    # Calculate frame indices (evenly spaced)
    indices = np.linspace(0, total_frames - 1, nframes).round().astype(int)

    logger.info(
        f"Video: {total_frames} total frames @ {video_fps:.1f} fps, "
        f"extracting {nframes} frames"
    )

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if specified
        if resize:
            frame = cv2.resize(frame, resize)

        frames.append(frame)

    cap.release()

    return frames


def save_frames_to_temp(frames: list[np.ndarray]) -> list[str]:
    """Save frame arrays to temporary files and return paths."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for frame processing")

    paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(temp_file.name, "JPEG", quality=85)
        paths.append(_temp_manager.register(temp_file.name))

    return paths


class MLXMultimodalLM:
    """
    Wrapper around mlx-vlm for multimodal inference.

    This class provides a unified interface for multimodal language models
    using Apple's MLX framework. Supports:
    - Image understanding (single and multi-image)
    - Video understanding (smart frame extraction)
    - Audio understanding (for supported models)
    - OpenAI-compatible API format

    Supported models include:
    - Qwen2-VL / Qwen2.5-VL / Qwen3-VL
    - LLaVA
    - Idefics3
    - PaliGemma
    - And more via mlx-vlm

    Example:
        >>> model = MLXMultimodalLM("mlx-community/Qwen2-VL-2B-Instruct-4bit")
        >>> model.load()
        >>> output = model.generate(
        ...     prompt="What's in this image?",
        ...     images=["photo.jpg"]
        ... )
        >>> print(output.text)
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        enable_cache: bool = True,
        cache_size: int = 50,
        max_kv_size: int = 0,
        draft_model: str | None = None,
        draft_kind: str | None = None,
        draft_block_size: int | None = None,
    ):
        """
        Initialize the MLX multimodal language model.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            enable_cache: Enable KV cache for repeated image/video+prompt (default: True)
            cache_size: Maximum cache entries (default: 50)
            max_kv_size: Maximum KV cache size per sequence (0 = unbounded)
            draft_model: Optional MLLM speculative draft/assistant model path.
            draft_kind: Optional mlx-vlm draft kind, for example "mtp".
            draft_block_size: Optional speculative block size passed to mlx-vlm.
        """
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.enable_cache = enable_cache
        self.max_kv_size = max_kv_size
        self.draft_model_path = draft_model
        self.draft_kind = draft_kind
        self.draft_block_size = draft_block_size

        self.model = None
        self.processor = None
        self.config = None
        self._draft_model = None
        self._loaded = False
        self._video_native = False
        self._video_native_with_audio = False

        # Initialize MLLM prefix cache manager (with vision embedding caching)
        self._cache_manager: MLLMPrefixCacheManager | None = None
        if enable_cache:
            self._cache_manager = MLLMPrefixCacheManager(max_entries=cache_size)

    def load(self) -> None:
        """Load the model and processor."""
        if self._loaded:
            return

        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            logger.info(f"Loading MLLM: {self.model_name}")

            self.model, self.processor = load(self.model_name)
            self.config = load_config(self.model_name)
            if self.draft_model_path:
                self._draft_model = self._load_draft_model()
                _install_draft_metrics_hooks(self._draft_model)

            self._loaded = True
            self._video_native = hasattr(
                self.model.config, "video_token_id"
            ) or hasattr(self.model.config, "video_token_index")
            # Omni models expose a sound_encoder; for these, a video_url
            # without a paired audio_url should auto-extract the video's
            # audio track so the model can fuse A/V in one forward pass.
            # Decoupled from _video_native because some omni models (e.g.
            # Nemotron-H Omni) don't expose video_token_id at config level
            # and run through the frames-as-images fallback path.
            self._video_native_with_audio = _model_has_sound_encoder(self.model)
            logger.info(f"MLLM loaded successfully: {self.model_name}")
            if self._video_native:
                logger.info("Native video pipeline enabled (temporal 3D conv + M-RoPE)")
            if self._video_native_with_audio:
                logger.info(
                    "Omni model detected: video_url will auto-extract audio for A/V fusion"
                )

        except ImportError:
            raise ImportError(
                "mlx-vlm is required for multimodal inference. "
                "Install with: pip install mlx-vlm"
            )
        except Exception as e:
            logger.error(f"Failed to load MLLM: {e}")
            raise

    def _load_draft_model(self):
        if self.draft_kind == "mtp":
            return load_gemma4_assistant_drafter(self.draft_model_path)

        from mlx_vlm.utils import load

        draft_model, _ = load(self.draft_model_path)
        return draft_model

    def _draft_generation_kwargs(self, call_kwargs: dict | None = None) -> dict:
        """Return mlx-vlm drafter kwargs when the request explicitly opts in.

        ``call_kwargs`` is the outbound mlx-vlm kwargs dict. This method removes
        vllm-mlx drafter control keys before the dict is forwarded so caller
        passthrough values cannot conflict with the configured server drafter.
        """
        draft_requested = False
        if call_kwargs is not None:
            draft_requested = bool(call_kwargs.pop("mllm_draft", False))
            for key in _DRAFT_KWARG_NAMES:
                call_kwargs.pop(key, None)
        if not draft_requested or self._draft_model is None:
            return {}
        # Tests may install the draft model after load(); the hook is idempotent.
        _install_draft_metrics_hooks(self._draft_model)
        kwargs = {"draft_model": self._draft_model}
        if self.draft_kind:
            kwargs["draft_kind"] = self.draft_kind
        if self.draft_block_size is not None:
            kwargs["draft_block_size"] = self.draft_block_size
        return kwargs

    def _reset_draft_metrics(self) -> int:
        if self._draft_model is None:
            return 0
        if hasattr(self._draft_model, "accept_lens"):
            self._draft_model.accept_lens = []
        if hasattr(self._draft_model, "_vllm_mlx_draft_counts"):
            self._draft_model._vllm_mlx_draft_counts = []
        return 0

    def _draft_metrics_since(self, start_accept_lens: int) -> dict[str, int]:
        if self._draft_model is None:
            return {"mtp_drafts": 0, "mtp_accepted": 0}
        accept_lens = list(getattr(self._draft_model, "accept_lens", []))
        if start_accept_lens > len(accept_lens):
            new_accept_lens = accept_lens
        else:
            new_accept_lens = accept_lens[start_accept_lens:]
        draft_counts = list(getattr(self._draft_model, "_vllm_mlx_draft_counts", []))
        if start_accept_lens > len(draft_counts):
            new_draft_counts = draft_counts
        else:
            new_draft_counts = draft_counts[start_accept_lens:]
        block_size = (
            int(self.draft_block_size)
            if self.draft_block_size is not None
            else int(
                getattr(getattr(self._draft_model, "config", None), "block_size", 0)
            )
        )
        drafted_per_round = max(block_size - 1, 0)
        mtp_drafts = (
            sum(max(int(value), 0) for value in new_draft_counts)
            if new_draft_counts
            else drafted_per_round * len(new_accept_lens)
        )
        return {
            "mtp_drafts": mtp_drafts,
            "mtp_accepted": sum(int(value) for value in new_accept_lens),
        }

    def get_language_model(self):
        """Extract the underlying language model for mlx_lm TextModel construction."""
        return self.model.language_model

    def get_tokenizer(self):
        """Get the text tokenizer (not the multimodal processor)."""
        return self.processor.tokenizer

    def _prepare_images(self, images: list) -> list[str]:
        """Process remote/base64 image inputs into local temp file paths."""
        processed = []
        for img in images:
            try:
                path = process_image_input(img)
                processed.append(path)
            except Exception as e:
                logger.warning(f"Failed to process image: {e}")
        return processed

    def _prepare_audio(self, audio_inputs: list) -> list[str]:
        """Process audio inputs and return local file paths."""
        processed = []
        for audio_input in audio_inputs:
            try:
                path = process_audio_input(audio_input)
                processed.append(path)
            except Exception as e:
                logger.warning(f"Failed to process audio: {e}")
        return processed

    def _prepare_video(
        self,
        video_input: str | dict,
        fps: float = DEFAULT_FPS,
        max_frames: int = MAX_FRAMES,
        resolved_path: str | None = None,
    ) -> list[str]:
        """
        Process video input and extract frames.

        Supports:
        - URLs (http/https) - will be downloaded
        - Base64 encoded videos (data:video/mp4;base64,...)
        - OpenAI format dicts: {"url": "..."} or {"video_url": {"url": "..."}}

        Args:
            video_input: Video in any supported format
            fps: Frames per second to extract
            max_frames: Maximum frames to extract
            resolved_path: Optional pre-resolved local path. Callers that
                already ran process_video_input (e.g. for parallel audio
                extraction) pass it here to avoid re-downloading / re-decoding.

        Returns:
            List of paths to extracted frame images
        """
        # Reuse caller's resolved path when supplied; otherwise resolve here
        # (downloads if URL, decodes if base64).
        video_path = resolved_path or process_video_input(video_input)

        # Extract frames
        frames = extract_video_frames_smart(
            video_path,
            fps=fps,
            max_frames=max_frames,
        )
        return save_frames_to_temp(frames)

    def _collect_video_inputs(self, messages: list[dict]) -> dict[int, list]:
        """Collect video inputs from messages, keyed by message index.

        Handles both 'video' and 'video_url' content types, including
        Pydantic model conversion.
        """
        video_inputs: dict[int, list] = {}
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for item in content:
                if hasattr(item, "model_dump"):
                    item = item.model_dump(exclude_none=True)
                elif hasattr(item, "dict"):
                    item = {k: v for k, v in item.dict().items() if v is not None}

                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")
                if item_type == "video":
                    video_inputs.setdefault(msg_idx, []).append(
                        item.get("video", item.get("url", ""))
                    )
                elif item_type == "video_url":
                    vid_url = item.get("video_url", {})
                    if isinstance(vid_url, str):
                        video_inputs.setdefault(msg_idx, []).append(vid_url)
                    elif isinstance(vid_url, dict):
                        url = vid_url.get("url", "")
                        if url:
                            video_inputs.setdefault(msg_idx, []).append(url)
        return video_inputs

    def _collect_audio_inputs(self, messages: list[dict]) -> dict[int, list]:
        """Collect audio inputs from messages, keyed by message index."""
        audio_inputs: dict[int, list] = {}
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for item in content:
                if hasattr(item, "model_dump"):
                    item = item.model_dump(exclude_none=True)
                elif hasattr(item, "dict"):
                    item = {k: v for k, v in item.dict().items() if v is not None}

                if not isinstance(item, dict):
                    continue

                item_type = item.get("type", "")
                if item_type == "audio":
                    audio_value = item.get("audio", item.get("url", ""))
                    if audio_value:
                        audio_inputs.setdefault(msg_idx, []).append(audio_value)
                elif item_type == "audio_url":
                    audio_url = item.get("audio_url", {})
                    if isinstance(audio_url, str):
                        audio_inputs.setdefault(msg_idx, []).append(audio_url)
                    elif isinstance(audio_url, dict):
                        url = audio_url.get("url", "")
                        if url:
                            audio_inputs.setdefault(msg_idx, []).append(url)
        return audio_inputs

    def _prepare_native_video_inputs(
        self,
        messages: list[dict],
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
        tools: list | None = None,
    ) -> tuple[str, dict]:
        """Preprocess messages into prompt + generation kwargs for native video.

        Mirrors the preprocessing in mlx_vlm.video_generate.main() so that
        upstream improvements are easy to adopt. Returns the formatted prompt
        text and a dict of kwargs ready to pass to video_generate.generate().

        Currently Qwen-family-specific (video_token_id / video_token_index).
        """
        import mlx.core as mx

        try:
            from mlx_vlm.video_generate import process_vision_info
        except ImportError:
            raise ImportError(
                "mlx_vlm.video_generate is required for native video support. "
                "Upgrade with: pip install --upgrade mlx-vlm"
            )

        # Translate OpenAI API messages into process_vision_info format
        native_messages = self._translate_messages_for_native_video(
            messages, video_fps, video_max_frames
        )

        # Use HF processor's chat template (handles timestamp interleaving)
        template_kwargs: dict = {}
        if tools:
            template_kwargs["tools"] = tools

        text = self.processor.apply_chat_template(
            native_messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

        # Extract vision inputs via mlx-vlm's process_vision_info
        image_inputs, video_inputs, fps_info = process_vision_info(
            native_messages, return_video_kwargs=True
        )

        # Collect audio paths emitted by the translation step
        # (explicit audio_url, or auto-extracted from video_url for omni
        # models).
        audio_inputs: list[str] = []
        for nmsg in native_messages:
            ncontent = nmsg.get("content", [])
            if not isinstance(ncontent, list):
                continue
            for nitem in ncontent:
                if isinstance(nitem, dict) and nitem.get("type") == "audio":
                    apath = nitem.get("audio")
                    if apath:
                        audio_inputs.append(apath)

        # Process through HF processor to get input_ids, pixel_values, grid_thw
        # and (for omni models) sound_clips / input_features.
        processor_kwargs: dict = {
            "text": [text],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if audio_inputs:
            processor_kwargs["audio"] = audio_inputs
        inputs = self.processor(**processor_kwargs)

        input_ids = mx.array(inputs["input_ids"])
        pixel_values = inputs.get(
            "pixel_values_videos", inputs.get("pixel_values", None)
        )
        if pixel_values is not None:
            pixel_values = mx.array(pixel_values)
        mask = mx.array(inputs["attention_mask"])

        gen_kwargs: dict = {}
        if inputs.get("video_grid_thw", None) is not None:
            gen_kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])
        if inputs.get("image_grid_thw", None) is not None:
            gen_kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

        # Forward audio embeddings/clips from the processor so the omni
        # model's sound encoder gets fed alongside the visual stream.
        for audio_key in (
            "sound_clips",
            "input_features",
            "feature_attention_mask",
            "audio_feature_lengths",
            "sound_feature_lengths",
            "sound_attention_mask",
        ):
            val = inputs.get(audio_key, None)
            if val is not None:
                gen_kwargs[audio_key] = val
        if audio_inputs:
            logger.info(
                f"Native video: forwarding audio ({len(audio_inputs)} clip(s)) "
                f"to omni model via "
                f"{[k for k in gen_kwargs if k in ('sound_clips', 'input_features')]}"
            )

        gen_kwargs["input_ids"] = input_ids
        gen_kwargs["pixel_values"] = pixel_values
        gen_kwargs["mask"] = mask

        grid_thw_info = gen_kwargs.get("video_grid_thw")
        logger.info(
            f"Native video: {input_ids.size} input tokens, "
            f"video_grid_thw={grid_thw_info.tolist() if grid_thw_info is not None else None}"
        )

        return text, gen_kwargs

    def _generate_native_video(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
        tools: list | None = None,
        **kwargs,
    ) -> MLLMOutput:
        """Generate using native video pipeline (Qwen-family models).

        Delegates preprocessing to _prepare_native_video_inputs and generation
        to mlx_vlm.video_generate.generate(), keeping our code aligned with
        upstream's video pipeline so improvements are easy to adopt.
        """
        try:
            from mlx_vlm.video_generate import generate
        except ImportError:
            raise ImportError(
                "mlx_vlm.video_generate is required for native video support. "
                "Upgrade with: pip install --upgrade mlx-vlm"
            )

        text, gen_kwargs = self._prepare_native_video_inputs(
            messages, video_fps, video_max_frames, tools
        )
        gen_kwargs["temperature"] = temperature

        result = generate(
            self.model,
            self.processor,
            prompt=text,
            max_tokens=max_tokens,
            verbose=False,
            **gen_kwargs,
        )

        if hasattr(result, "text"):
            return MLLMOutput(
                text=result.text,
                finish_reason="stop",
                prompt_tokens=getattr(result, "prompt_tokens", 0),
                completion_tokens=getattr(result, "generation_tokens", 0),
            )
        return MLLMOutput(text=str(result), finish_reason="stop")

    def _translate_messages_for_native_video(
        self,
        messages: list[dict],
        video_fps: float,
        video_max_frames: int,
    ) -> list[dict]:
        """Translate OpenAI API format messages to process_vision_info format.

        Converts video_url/video types and resolves remote/base64 inputs to local paths.
        Images are preserved as-is (process_vision_info handles them).
        """
        translated = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                translated.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                translated.append({"role": role, "content": str(content)})
                continue

            # Pre-pass: does this message have an explicit audio_url/audio
            # block? If so, we skip auto-extracting audio from a video_url to
            # honor the caller's explicit choice.
            has_explicit_audio = False
            for item in content:
                if hasattr(item, "model_dump"):
                    probe = item.model_dump(exclude_none=True)
                elif hasattr(item, "dict"):
                    probe = {k: v for k, v in item.dict().items() if v is not None}
                else:
                    probe = item
                if isinstance(probe, dict) and probe.get("type", "") in (
                    "audio",
                    "audio_url",
                ):
                    has_explicit_audio = True
                    break

            new_content = []
            for item in content:
                if hasattr(item, "model_dump"):
                    item = item.model_dump(exclude_none=True)
                elif hasattr(item, "dict"):
                    item = {k: v for k, v in item.dict().items() if v is not None}

                if not isinstance(item, dict):
                    new_content.append({"type": "text", "text": str(item)})
                    continue

                item_type = item.get("type", "")

                if item_type == "text":
                    new_content.append(item)

                elif item_type == "image_url":
                    img_url = item.get("image_url", {})
                    url = (
                        img_url.get("url", img_url)
                        if isinstance(img_url, dict)
                        else img_url
                    )
                    # Resolve to local path for process_vision_info
                    local_path = process_image_input(url)
                    new_content.append({"type": "image", "image": local_path})

                elif item_type == "image":
                    img = item.get("image", item.get("url", ""))
                    local_path = process_image_input(img)
                    new_content.append({"type": "image", "image": local_path})

                elif item_type in ("video", "video_url"):
                    # Extract video path/URL from various formats
                    if item_type == "video_url":
                        vid_url = item.get("video_url", {})
                        if isinstance(vid_url, str):
                            video_source = vid_url
                        elif isinstance(vid_url, dict):
                            video_source = vid_url.get("url", "")
                        else:
                            continue
                    else:
                        video_source = item.get("video", item.get("url", ""))

                    if not video_source:
                        continue

                    # Resolve to local path
                    video_path = process_video_input(video_source)
                    new_content.append(
                        {
                            "type": "video",
                            "video": video_path,
                            "fps": video_fps,
                            "max_frames": video_max_frames,
                        }
                    )
                    # For omni-capable models, pull the video's audio track
                    # alongside frames so the model can fuse A/V in one
                    # forward pass. We extract from the already-resolved local
                    # path (no raw user URL handed to ffmpeg → avoids URL-
                    # protocol SSRF via ffmpeg's network demuxers).
                    if not has_explicit_audio and getattr(
                        self, "_video_native_with_audio", False
                    ):
                        extracted = extract_audio_from_video(video_path)
                        if extracted is not None:
                            new_content.append({"type": "audio", "audio": extracted})

                elif item_type in ("audio", "audio_url"):
                    if item_type == "audio_url":
                        aud_url = item.get("audio_url", {})
                        if isinstance(aud_url, str):
                            audio_source = aud_url
                        elif isinstance(aud_url, dict):
                            audio_source = aud_url.get("url", "")
                        else:
                            continue
                    else:
                        audio_source = item.get("audio", item.get("url", ""))

                    if not audio_source:
                        continue

                    audio_path = process_audio_input(audio_source)
                    new_content.append({"type": "audio", "audio": audio_path})

                else:
                    new_content.append(item)

            translated.append({"role": role, "content": new_content})

        return translated

    def generate(
        self,
        prompt: str,
        images: list | None = None,
        videos: list | None = None,
        audio: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
        use_cache: bool = True,
        **kwargs,
    ) -> MLLMOutput:
        """
        Generate text from multimodal input.

        Args:
            prompt: Text prompt/question
            images: List of image URLs or base64 strings
            videos: List of video inputs (URLs, base64, or OpenAI format dicts)
            audio: List of audio file paths
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            video_fps: FPS for video frame extraction (default: 2.0)
            video_max_frames: Max frames to extract from video
            use_cache: Whether to use KV cache (default: True)
            **kwargs: Additional generation parameters

        Returns:
            MLLMOutput with generated text

        Example:
            # With local video
            output = model.generate("Describe this video", videos=["video.mp4"])

            # With video URL
            output = model.generate("What happens?", videos=["https://example.com/video.mp4"])

            # With base64 video
            output = model.generate("Describe", videos=["data:video/mp4;base64,AAAA..."])
        """
        if not self._loaded:
            self.load()

        from mlx_vlm import generate
        from mlx_vlm.models import cache as vlm_cache
        from mlx_vlm.prompt_utils import apply_chat_template

        images = images or []
        videos = videos or []
        audio = audio or []

        # Process all images (including frames from videos) and audio inputs
        all_images = []
        all_audio = []
        all_sources = []  # Track original sources for cache key

        # Process image inputs
        if images:
            all_images.extend(self._prepare_images(images))
            all_sources.extend(images)

        # Extract frames from videos
        for video_path in videos:
            frames = self._prepare_video(
                video_path,
                fps=video_fps,
                max_frames=video_max_frames,
            )
            all_images.extend(frames)
            # Include video params in cache key
            video_str = video_path if isinstance(video_path, str) else str(video_path)
            all_sources.append(
                f"video:{video_str}:fps{video_fps}:max{video_max_frames}"
            )
            logger.info(f"Added {len(frames)} frames from video: {video_path}")

        if audio:
            all_audio.extend(self._prepare_audio(audio))

        # Apply chat template if needed
        if (all_images or all_audio) and hasattr(self.processor, "apply_chat_template"):
            try:
                formatted_prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    prompt,
                    num_images=len(all_images),
                    num_audios=len(all_audio),
                )
            except Exception:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        # Check cache for existing KV state
        prompt_cache = None
        cache_hit = False

        if use_cache and all_audio:
            logger.info("MLLM cache disabled for audio inputs")
            use_cache = False

        if use_cache and self._cache_manager is not None and all_sources:
            prompt_cache, cache_hit = self._cache_manager.fetch_cache(
                all_sources, formatted_prompt
            )
            if cache_hit:
                logger.info(f"MLLM cache hit for {len(all_sources)} source(s)")

        # Create new cache if needed
        if prompt_cache is None and self.model is not None:
            try:
                prompt_cache = vlm_cache.make_prompt_cache(
                    self.model.language_model,
                    max_kv_size=self.max_kv_size or None,
                )
            except Exception:
                prompt_cache = None

        # Generate with cache
        draft_accept_start = self._reset_draft_metrics()
        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            all_images if all_images else None,
            audio=all_audio if all_audio else None,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            verbose=False,
            prompt_cache=prompt_cache,
            **self._draft_generation_kwargs(kwargs),
            **kwargs,
        )
        draft_metrics = self._draft_metrics_since(draft_accept_start)

        # Store cache for future reuse (only on miss)
        if use_cache and self._cache_manager and all_sources and not cache_hit:
            if prompt_cache is not None:
                try:
                    num_tokens = getattr(result, "prompt_tokens", 0)
                    self._cache_manager.store_cache(
                        all_sources, formatted_prompt, prompt_cache, num_tokens
                    )
                    logger.info(f"MLLM cache stored for {len(all_sources)} source(s)")
                except Exception as e:
                    logger.debug(f"Failed to store MLLM cache: {e}")

        # Handle GenerationResult object or plain string
        if hasattr(result, "text"):
            output_text = result.text
            prompt_tokens = getattr(result, "prompt_tokens", 0)
            generation_tokens = getattr(result, "generation_tokens", 0)
        else:
            output_text = str(result)
            prompt_tokens = 0
            generation_tokens = 0

        return MLLMOutput(
            text=output_text,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=generation_tokens,
            **draft_metrics,
        )

    def stream_generate(
        self,
        prompt: str,
        images: list | None = None,
        videos: list[str] | None = None,
        audio: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        video_fps: float = DEFAULT_FPS,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream text generation for multimodal input.

        Args:
            prompt: Text prompt
            images: List of image inputs
            videos: List of video paths
            audio: List of audio inputs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            video_fps: FPS for video frame extraction
            **kwargs: Additional parameters

        Yields:
            Generated text chunks
        """
        if not self._loaded:
            self.load()

        try:
            from mlx_vlm import stream_generate
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError:
            # Fallback to non-streaming
            output = self.generate(
                prompt=prompt,
                images=images,
                videos=videos,
                audio=audio,
                max_tokens=max_tokens,
                temperature=temperature,
                video_fps=video_fps,
                **kwargs,
            )
            yield output.text
            return

        images = images or []
        videos = videos or []
        audio = audio or []

        # Process images
        all_images = []
        all_audio = []
        if images:
            all_images.extend(self._prepare_images(images))
        for video_path in videos:
            frames = self._prepare_video(video_path, fps=video_fps)
            all_images.extend(frames)
        if audio:
            all_audio.extend(self._prepare_audio(audio))

        # Apply chat template
        if all_images or all_audio:
            try:
                formatted_prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    prompt,
                    num_images=len(all_images),
                    num_audios=len(all_audio),
                )
            except Exception:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        for chunk in stream_generate(
            self.model,
            self.processor,
            formatted_prompt,
            all_images if all_images else None,
            audio=all_audio if all_audio else None,
            max_tokens=max_tokens,
            temp=temperature,
            **self._draft_generation_kwargs(kwargs),
            **kwargs,
        ):
            yield chunk

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> MLLMOutput:
        """
        Chat with OpenAI-compatible message format.

        Supports multimodal content in messages:
        - {"type": "text", "text": "..."}
        - {"type": "image_url", "image_url": {"url": "..."}}
        - {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}

        Args:
            messages: List of chat messages (OpenAI format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            MLLMOutput with assistant's response
        """
        if not self._loaded:
            self.load()

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import get_chat_template

        # Extract text, images and audio from messages
        # Build chat_messages for multi-turn support WITH proper image/audio tokens per message
        all_image_urls = []  # Raw URLs/paths to process later
        chat_messages = []  # List of properly formatted messages for chat template

        logger.info(f"MLLM.chat() called with {len(messages)} messages")

        # Pop params early so they don't leak into mlx_vlm.generate()
        video_fps = kwargs.pop("video_fps", DEFAULT_FPS)
        video_max_frames = kwargs.pop("video_max_frames", MAX_FRAMES)
        tools = kwargs.pop("tools", None)
        use_cache = kwargs.pop("use_cache", True)
        enable_thinking = kwargs.pop("enable_thinking", True)

        # Collect video and audio inputs from messages
        _msg_video_inputs = self._collect_video_inputs(messages)
        _msg_audio_inputs = self._collect_audio_inputs(messages)

        # Use native video pipeline for supported models
        if self._video_native and _msg_video_inputs:
            return self._generate_native_video(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                video_fps=video_fps,
                video_max_frames=video_max_frames,
                tools=tools,
                **kwargs,
            )

        # Fallback: extract frames and treat as individual images
        _msg_video_frame_counts: dict[int, int] = {}
        _msg_extra_audio: dict[int, list[str]] = {}
        all_video_frames: list[str] = []
        all_audio_inputs: list[str] = []
        for msg_idx, vid_inputs in _msg_video_inputs.items():
            total_frames = 0
            has_explicit_audio = bool(_msg_audio_inputs.get(msg_idx))
            for vid_input in vid_inputs:
                # Resolve the video to a local path ONCE per input. Both
                # audio extraction (when this is an omni model with no
                # explicit audio block) and frame extraction need a local
                # file; resolving twice would re-download remote URLs and
                # re-decode base64. Resolving up front also keeps user-
                # supplied raw URLs out of ffmpeg's URL-protocol demuxers
                # (avoids SSRF via http://, rtsp://, etc.).
                try:
                    resolved_video_path = process_video_input(vid_input)
                except Exception as exc:
                    logger.warning(f"Could not resolve video: {exc}")
                    resolved_video_path = None

                if (
                    resolved_video_path
                    and self._video_native_with_audio
                    and not has_explicit_audio
                ):
                    extracted_audio = extract_audio_from_video(resolved_video_path)
                    if extracted_audio:
                        _msg_extra_audio.setdefault(msg_idx, []).append(extracted_audio)

                frames = self._prepare_video(
                    vid_input,
                    fps=video_fps,
                    max_frames=video_max_frames,
                    resolved_path=resolved_video_path,
                )
                all_video_frames.extend(frames)
                total_frames += len(frames)
                logger.info(f"Added {len(frames)} frames from video: {vid_input}")
            _msg_video_frame_counts[msg_idx] = total_frames

        # Merge auto-extracted audio into the per-message audio map so the
        # chat-template token-counting loop downstream sees the right count.
        for msg_idx, extra in _msg_extra_audio.items():
            _msg_audio_inputs.setdefault(msg_idx, []).extend(extra)

        for aud_inputs in _msg_audio_inputs.values():
            all_audio_inputs.extend(aud_inputs)

        chat_messages = _build_mllm_chat_messages(
            messages,
            all_image_urls=all_image_urls,
            video_frame_counts=_msg_video_frame_counts,
        )

        # Process images
        all_images = []
        if all_image_urls:
            all_images.extend(self._prepare_images(all_image_urls))
        # Append pre-processed video frames
        all_images.extend(all_video_frames)
        all_audio = self._prepare_audio(all_audio_inputs) if all_audio_inputs else []

        # Apply chat template directly - messages are already properly structured
        logger.info(
            f"Applying chat template with {len(chat_messages)} messages, {len(all_images)} images, {len(all_audio)} audios"
        )
        for i, cm in enumerate(chat_messages):
            content_preview = str(cm.get("content", ""))[:80]
            logger.info(
                f"  Chat msg {i}: role={cm['role']}, content={content_preview}..."
            )

        # Build template kwargs for tool definitions (tools already popped above)
        template_extra_kwargs = {}
        if tools:
            template_extra_kwargs["tools"] = tools

        try:
            formatted_prompt = get_chat_template(
                self.processor,
                chat_messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **template_extra_kwargs,
            )
        except Exception as e:
            logger.warning(
                f"Failed to apply chat template: {e}, using last user message"
            )
            # Fallback to last user message if template fails
            last_user_msg = ""
            for m in reversed(chat_messages):
                if m["role"] == "user":
                    content = m.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                last_user_msg = item.get("text", "")
                                break
                    else:
                        last_user_msg = content
                    break
            formatted_prompt = last_user_msg

        # Prefix caching with vision embedding support
        # Following LMCache approach: cache vision embeddings to skip encoder on hit
        import time

        from mlx_vlm.models import cache as vlm_cache

        cache_entry = None
        prefix_match_len = 0
        vision_embeddings = None
        cache_hit = False

        # Tokenize prompt for cache lookup
        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        token_ids = tokenizer.encode(formatted_prompt)

        # Check prefix cache
        if use_cache and all_audio:
            logger.info("Prefix cache disabled for audio inputs")
            use_cache = False

        if use_cache and self._cache_manager is not None and all_images:
            try:
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_images, formatted_prompt, token_ids
                )
                if cache_entry:
                    cache_hit = True
                    vision_embeddings = cache_entry.vision_embeddings
                    if vision_embeddings is not None:
                        logger.info(
                            "[PREFIX CACHE] Vision embeddings cached - would skip encoder!"
                        )
                    if prefix_match_len > 0:
                        logger.info(
                            f"[PREFIX CACHE] {prefix_match_len} prefix tokens match"
                        )
            except Exception as e:
                logger.warning(f"Cache fetch failed: {e}")

        # Generate - use KV cache if available from previous identical request
        start_time = time.time()

        # Create or reuse prompt cache for prefix caching speedup
        prompt_cache = None
        skip_prompt_processing = False

        if cache_hit and cache_entry and cache_entry.kv_cache:
            # NOTE: mlx-vlm's generate_step() has its own multimodal KV cache with prefix matching
            # (MULTIMODAL_KV_CACHE_ENABLED in mlx_vlm/utils.py). Let it handle caching.
            # We only use vllm-mlx's cache for text-only requests (no images).
            if all_images:
                # Let mlx-vlm's multimodal cache handle this - don't interfere
                logger.info(
                    "[PREFIX CACHE] Images present - delegating to mlx-vlm multimodal cache"
                )
                prompt_cache = None  # Fresh cache, mlx-vlm will handle prefix matching
                skip_prompt_processing = False
            else:
                # Text-only: can use skip_prompt_processing for maximum speedup
                logger.info(
                    "[PREFIX CACHE] Text-only cache hit - using skip_prompt_processing speedup"
                )
                cached_prompt_cache = cache_entry.kv_cache
                try:
                    import copy

                    prompt_cache = []
                    for layer_cache in cached_prompt_cache:
                        new_cache = copy.copy(layer_cache)
                        if hasattr(layer_cache, "state"):
                            state = layer_cache.state
                            if state is not None:
                                import mlx.core as mx

                                if len(state) >= 2 and state[0] is not None:
                                    new_cache.keys = mx.array(state[0])
                                    new_cache.values = mx.array(state[1])
                                    if len(state) >= 3:
                                        new_cache.offset = state[2]
                                    elif hasattr(layer_cache, "offset"):
                                        new_cache.offset = layer_cache.offset
                        prompt_cache.append(new_cache)
                    skip_prompt_processing = True
                    logger.info(
                        f"[PREFIX CACHE] Skipping {prefix_match_len} token forward pass"
                    )
                except Exception as e:
                    logger.warning(f"[PREFIX CACHE] Failed to copy cache: {e}")
                    prompt_cache = None
                    skip_prompt_processing = False

        if prompt_cache is None and self.model is not None:
            # Create fresh cache
            try:
                prompt_cache = vlm_cache.make_prompt_cache(
                    self.model.language_model,
                    max_kv_size=self.max_kv_size or None,
                )
            except Exception:
                prompt_cache = None

        draft_accept_start = self._reset_draft_metrics()
        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            all_images if all_images else None,
            audio=all_audio if all_audio else None,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
            prompt_cache=prompt_cache,
            skip_prompt_processing=skip_prompt_processing,
            **self._draft_generation_kwargs(kwargs),
            **kwargs,
        )
        draft_metrics = self._draft_metrics_since(draft_accept_start)

        # Store KV cache for future reuse (on cache miss)
        # IMPORTANT: We need to store only the prompt portion, not generated tokens
        if (
            use_cache
            and self._cache_manager is not None
            and all_images
            and not cache_hit
            and prompt_cache
        ):
            try:
                import copy

                import mlx.core as mx

                # Get prompt token count (before generation)
                prompt_tokens_count = getattr(result, "prompt_tokens", 0)

                # Deep copy the cache and trim to prompt tokens only
                cache_to_store = []
                for layer_cache in prompt_cache:
                    new_cache = copy.copy(layer_cache)
                    if hasattr(layer_cache, "state"):
                        state = layer_cache.state
                        if (
                            state is not None
                            and len(state) >= 2
                            and state[0] is not None
                        ):
                            # Copy arrays
                            keys = mx.array(state[0])
                            values = mx.array(state[1])
                            # Trim to prompt tokens only (not generated tokens)
                            if (
                                hasattr(layer_cache, "offset")
                                and layer_cache.offset > prompt_tokens_count
                            ):
                                # For caches with offset tracking, slice to prompt length
                                new_cache.keys = keys[:, :, :prompt_tokens_count, :]
                                new_cache.values = values[:, :, :prompt_tokens_count, :]
                                new_cache.offset = prompt_tokens_count
                            else:
                                new_cache.keys = keys
                                new_cache.values = values
                                if len(state) >= 3:
                                    new_cache.offset = state[2]
                                elif hasattr(layer_cache, "offset"):
                                    new_cache.offset = min(
                                        layer_cache.offset, prompt_tokens_count
                                    )
                    cache_to_store.append(new_cache)

                self._cache_manager.store(
                    images=all_images,
                    prompt=formatted_prompt,
                    vision_embeddings=None,
                    kv_cache=cache_to_store,
                    token_ids=token_ids,
                    num_image_tokens=256,
                    model_name=self.model_name,
                )
                logger.info(
                    f"[PREFIX CACHE] Stored KV cache for {len(all_images)} image(s) ({prompt_tokens_count} prompt tokens)"
                )
            except Exception as e:
                logger.warning(f"Failed to cache: {e}")

        # Handle GenerationResult object or plain string
        if hasattr(result, "text"):
            output_text = result.text
            prompt_tokens = getattr(result, "prompt_tokens", 0)
            generation_tokens = getattr(result, "generation_tokens", 0)
        else:
            output_text = str(result)
            prompt_tokens = 0
            generation_tokens = 0

        return MLLMOutput(
            text=output_text,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=generation_tokens,
            **draft_metrics,
        )

    def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[MLLMOutput]:
        """
        Stream chat with OpenAI-compatible message format.

        Supports multimodal content in messages:
        - {"type": "text", "text": "..."}
        - {"type": "image_url", "image_url": {"url": "..."}}
        - {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}

        Args:
            messages: List of chat messages (OpenAI format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Yields:
            MLLMOutput with incremental text chunks
        """
        if not self._loaded:
            self.load()

        try:
            from mlx_vlm import stream_generate
            from mlx_vlm.prompt_utils import get_chat_template
        except ImportError:
            # Fallback to non-streaming if stream_generate not available
            output = self.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            yield output
            return

        # Extract text and images from messages
        # Build chat_messages for multi-turn support WITH proper image tokens per message
        all_image_urls = []  # Raw URLs/paths to process later
        chat_messages = []  # List of properly formatted messages for chat template

        # Pop params early so they don't leak into mlx_vlm.generate()
        video_fps = kwargs.pop("video_fps", DEFAULT_FPS)
        video_max_frames = kwargs.pop("video_max_frames", MAX_FRAMES)
        tools = kwargs.pop("tools", None)
        use_cache = kwargs.pop("use_cache", True)
        enable_thinking = kwargs.pop("enable_thinking", True)

        # Collect video and audio inputs from messages
        _msg_video_inputs = self._collect_video_inputs(messages)
        _msg_audio_inputs = self._collect_audio_inputs(messages)

        # Use native video pipeline for supported models.
        # NOTE: Native video yields a single chunk (not incremental streaming)
        # because mlx_vlm.video_generate has no streaming API. The event loop
        # is NOT blocked at the server level — SimpleEngine wraps this in
        # asyncio.to_thread(). True token-level streaming requires upstream
        # mlx-vlm support for video stream_generate.
        if self._video_native and _msg_video_inputs:
            output = self._generate_native_video(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                video_fps=video_fps,
                video_max_frames=video_max_frames,
                tools=tools,
                **kwargs,
            )
            yield output
            return

        # Fallback: frames as images
        _msg_video_frame_counts: dict[int, int] = {}
        _msg_extra_audio: dict[int, list[str]] = {}
        all_video_frames: list[str] = []
        all_audio_inputs: list[str] = []
        for msg_idx, vid_inputs in _msg_video_inputs.items():
            total_frames = 0
            has_explicit_audio = bool(_msg_audio_inputs.get(msg_idx))
            for vid_input in vid_inputs:
                # Resolve once; reused for audio extraction and frame prep.
                # See the matching block in chat() for rationale.
                try:
                    resolved_video_path = process_video_input(vid_input)
                except Exception as exc:
                    logger.warning(f"Could not resolve video: {exc}")
                    resolved_video_path = None

                if (
                    resolved_video_path
                    and self._video_native_with_audio
                    and not has_explicit_audio
                ):
                    extracted_audio = extract_audio_from_video(resolved_video_path)
                    if extracted_audio:
                        _msg_extra_audio.setdefault(msg_idx, []).append(extracted_audio)

                frames = self._prepare_video(
                    vid_input,
                    fps=video_fps,
                    max_frames=video_max_frames,
                    resolved_path=resolved_video_path,
                )
                all_video_frames.extend(frames)
                total_frames += len(frames)
                logger.info(f"Added {len(frames)} frames from video: {vid_input}")
            _msg_video_frame_counts[msg_idx] = total_frames

        for msg_idx, extra in _msg_extra_audio.items():
            _msg_audio_inputs.setdefault(msg_idx, []).extend(extra)

        for aud_inputs in _msg_audio_inputs.values():
            all_audio_inputs.extend(aud_inputs)

        chat_messages = _build_mllm_chat_messages(
            messages,
            all_image_urls=all_image_urls,
            video_frame_counts=_msg_video_frame_counts,
        )

        all_images = []
        if all_image_urls:
            all_images.extend(self._prepare_images(all_image_urls))
        all_images.extend(all_video_frames)
        all_audio = self._prepare_audio(all_audio_inputs) if all_audio_inputs else []

        # Build template kwargs for tool definitions (tools already popped above)
        template_extra_kwargs = {}
        if tools:
            template_extra_kwargs["tools"] = tools

        try:
            formatted_prompt = get_chat_template(
                self.processor,
                chat_messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **template_extra_kwargs,
            )
        except Exception as e:
            logger.warning(
                f"Failed to apply chat template: {e}, using last user message"
            )
            # Fallback to last user message if template fails
            last_user_msg = ""
            for m in reversed(chat_messages):
                if m["role"] == "user":
                    content = m.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                last_user_msg = item.get("text", "")
                                break
                    else:
                        last_user_msg = content
                    break
            formatted_prompt = last_user_msg

        # Check cache for existing KV state (uses images as cache key)
        from mlx_vlm.models import cache as vlm_cache

        prompt_cache = None
        cache_hit = False

        if use_cache and all_audio:
            logger.info("Stream chat cache disabled for audio inputs")
            use_cache = False

        if use_cache and self._cache_manager is not None and all_images:
            prompt_cache, cache_hit = self._cache_manager.fetch_cache(
                all_images, formatted_prompt
            )
            if cache_hit:
                logger.debug(f"Stream chat cache hit for {len(all_images)} image(s)")

        # Create new cache if needed
        if prompt_cache is None and self.model is not None:
            try:
                prompt_cache = vlm_cache.make_prompt_cache(
                    self.model.language_model,
                    max_kv_size=self.max_kv_size or None,
                )
            except Exception:
                prompt_cache = None

        # Stream generate tokens with cache
        accumulated_text = ""
        token_count = 0
        draft_accept_start = self._reset_draft_metrics()

        for chunk in stream_generate(
            self.model,
            self.processor,
            formatted_prompt,
            all_images if all_images else None,
            audio=all_audio if all_audio else None,
            max_tokens=max_tokens,
            temp=temperature,
            prompt_cache=prompt_cache,
            **self._draft_generation_kwargs(kwargs),
            **kwargs,
        ):
            token_count += 1
            # chunk is a GenerationResult with .text attribute containing the new token
            new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            accumulated_text += new_text

            yield MLLMOutput(
                text=new_text,  # Just the new token for streaming
                finish_reason=None,
                prompt_tokens=getattr(chunk, "prompt_tokens", 0),
                completion_tokens=token_count,
            )

        # Final yield with finish_reason
        yield MLLMOutput(
            text="",
            finish_reason="stop",
            prompt_tokens=getattr(chunk, "prompt_tokens", 0) if "chunk" in dir() else 0,
            completion_tokens=token_count,
            **self._draft_metrics_since(draft_accept_start),
        )

    def describe_image(
        self,
        image: str,
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """
        Convenience method to describe an image.

        Args:
            image: Image path, URL, or base64 string
            prompt: Description prompt
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Image description text
        """
        output = self.generate(
            prompt=prompt,
            images=[image],
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def answer_about_image(
        self,
        image: str,
        question: str,
        max_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Answer a question about an image.

        Args:
            image: Image path, URL, or base64 string
            question: Question about the image
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Answer text
        """
        output = self.generate(
            prompt=question,
            images=[image],
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def describe_video(
        self,
        video: str | dict,
        prompt: str = "Describe what happens in this video.",
        fps: float = 2.0,
        max_frames: int = 32,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """
        Describe a video using frame extraction.

        Args:
            video: Video file path, URL, base64, or OpenAI format dict
            prompt: Description prompt
            fps: Frames per second to extract
            max_frames: Maximum frames to extract
            max_tokens: Maximum tokens to generate

        Returns:
            Video description text

        Example:
            # URL
            model.describe_video("https://example.com/video.mp4")

            # OpenAI format
            model.describe_video({"url": "https://example.com/video.mp4"})
        """
        output = self.generate(
            prompt=prompt,
            videos=[video],
            video_fps=fps,
            video_max_frames=max_frames,
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def get_cache_stats(self) -> dict:
        """
        Get MLLM cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, hit_rate, tokens_saved, etc.)
        """
        if self._cache_manager is None:
            return {"enabled": False}

        stats = self._cache_manager.get_stats()
        stats["enabled"] = True
        stats["cache_entries"] = len(self._cache_manager)
        stats["max_entries"] = self._cache_manager.max_size
        return stats

    def clear_cache(self) -> None:
        """Clear the MLLM KV cache."""
        if self._cache_manager is not None:
            self._cache_manager.clear()
            logger.info("MLLM cache cleared")

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "type": "multimodal-language-model",
            "supports_video": True,
            "supports_streaming": True,
            "cache_enabled": self.enable_cache,
        }

        if self.config:
            info["model_type"] = getattr(self.config, "model_type", "unknown")

        if self._cache_manager is not None:
            info["cache_stats"] = self._cache_manager.get_stats()

        return info

    @staticmethod
    def list_supported_model_families() -> dict[str, str]:
        """
        List supported model families and their patterns.

        Any model on HuggingFace containing these patterns in the name
        is likely compatible with mlx-vlm.
        """
        return {
            "Qwen-VL": "Qwen VL models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, etc.)",
            "LLaVA": "LLaVA vision-language models",
            "Idefics": "Idefics vision-language models",
            "PaliGemma": "PaliGemma multimodal models",
            "Pixtral": "Mistral's Pixtral vision models",
            "Molmo": "Allen AI's Molmo models",
            "Phi-3-Vision": "Microsoft's Phi-3 Vision models",
            "CogVLM": "Tsinghua's CogVLM models",
            "InternVL": "InternVL models",
            "MiniCPM-V": "OpenBMB's MiniCPM-V models",
            "Florence": "Microsoft Florence vision models",
            "DeepSeek-VL": "DeepSeek's vision-language models (DeepSeek-VL, DeepSeek-VL2)",
        }

    @staticmethod
    def is_mllm_model(model_name: str) -> bool:
        """Check if a model name indicates an MLLM model."""
        mllm_patterns = [
            "-VL-",
            "-VL/",
            "VL-",
            "llava",
            "LLaVA",
            "idefics",
            "Idefics",
            "paligemma",
            "PaliGemma",
            "gemma-3",
            "gemma3",  # Gemma 3 (multimodal)
            "medgemma",
            "MedGemma",  # MedGemma (medical multimodal)
            "pixtral",
            "Pixtral",
            "molmo",
            "Molmo",
            "phi3-vision",
            "phi-3-vision",
            "cogvlm",
            "CogVLM",
            "internvl",
            "InternVL",
            "minicpm-v",
            "MiniCPM-V",
            "florence",
            "Florence",
            "deepseek-vl",
            "DeepSeek-VL",
        ]
        model_lower = model_name.lower()
        return any(pattern.lower() in model_lower for pattern in mllm_patterns)

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXMultimodalLM model={self.model_name} status={status}>"


# Backwards compatibility aliases
MLXVisionLanguageModel = MLXMultimodalLM
VLMOutput = MLLMOutput
is_vlm_model = MLXMultimodalLM.is_mllm_model
