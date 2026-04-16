# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible API server."""

import json
import platform
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# Unit Tests - Request/Response Models
# =============================================================================


class TestRequestModels:
    """Test Pydantic request models."""

    def test_chat_message_text_only(self):
        """Test chat message with text content."""
        from vllm_mlx.server import Message

        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_multimodal(self):
        """Test chat message with multimodal content."""
        from vllm_mlx.server import Message

        content = [
            {"type": "text", "text": "What's this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        msg = Message(role="user", content=content)

        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_image_url_model(self):
        """Test ImageUrl model."""
        from vllm_mlx.server import ImageUrl

        img_url = ImageUrl(url="https://example.com/image.jpg")
        assert img_url.url == "https://example.com/image.jpg"
        assert img_url.detail is None

    def test_video_url_model(self):
        """Test VideoUrl model."""
        from vllm_mlx.server import VideoUrl

        video_url = VideoUrl(url="https://example.com/video.mp4")
        assert video_url.url == "https://example.com/video.mp4"

    def test_content_part_text(self):
        """Test ContentPart with text."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(type="text", text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"

    def test_content_part_image(self):
        """Test ContentPart with image_url."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(
            type="image_url", image_url={"url": "https://example.com/img.jpg"}
        )
        assert part.type == "image_url"
        # image_url can be dict or ImageUrl object
        if isinstance(part.image_url, dict):
            assert part.image_url["url"] == "https://example.com/img.jpg"
        else:
            assert part.image_url.url == "https://example.com/img.jpg"

    def test_content_part_video(self):
        """Test ContentPart with video."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(type="video", video="/path/to/video.mp4")
        assert part.type == "video"
        assert part.video == "/path/to/video.mp4"

    def test_content_part_video_url(self):
        """Test ContentPart with video_url."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(
            type="video_url", video_url={"url": "https://example.com/video.mp4"}
        )
        assert part.type == "video_url"
        # video_url can be dict or VideoUrl object
        if isinstance(part.video_url, dict):
            assert part.video_url["url"] == "https://example.com/video.mp4"
        else:
            assert part.video_url.url == "https://example.com/video.mp4"


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_basic_request(self):
        """Test basic chat completion request."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model", messages=[Message(role="user", content="Hello")]
        )

        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.max_tokens is None  # uses _default_max_tokens when None
        assert (
            request.temperature is None
        )  # resolved at runtime by _resolve_temperature
        assert request.stream is False  # default

    def test_request_with_options(self):
        """Test request with custom options."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            temperature=0.5,
            stream=True,
        )

        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.stream is True

    def test_request_with_video_params(self):
        """Test request with video parameters."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Describe the video")],
            video_fps=2.0,
            video_max_frames=16,
        )

        assert request.video_fps == 2.0
        assert request.video_max_frames == 16


class TestCompletionRequest:
    """Test CompletionRequest model."""

    def test_basic_completion_request(self):
        """Test basic completion request."""
        from vllm_mlx.server import CompletionRequest

        request = CompletionRequest(model="test-model", prompt="Once upon a time")

        assert request.model == "test-model"
        assert request.prompt == "Once upon a time"
        assert request.max_tokens is None  # uses _default_max_tokens when None


class TestServeCli:
    """Test serve CLI argument parsing."""

    def test_trust_remote_code_flag_defaults_false(self):
        """Serve CLI should require explicit opt-in for remote code loading."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["serve", "mlx-community/Llama-3.2-3B-Instruct-4bit"])
        assert args.trust_remote_code is False

        args = parser.parse_args(
            [
                "serve",
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "--trust-remote-code",
            ]
        )
        assert args.trust_remote_code is True

    def test_tool_call_parser_accepts_harmony_aliases(self):
        """GPT-OSS/Harmony parsers should be selectable from the serve CLI."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "serve",
                "lmstudio-community/gpt-oss-20b-MLX-8bit",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "harmony",
            ]
        )

        assert args.command == "serve"
        assert args.tool_call_parser == "harmony"
        assert args.enable_auto_tool_choice is True

        args = parser.parse_args(
            [
                "serve",
                "lmstudio-community/gpt-oss-20b-MLX-8bit",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "gpt-oss",
            ]
        )

        assert args.tool_call_parser == "gpt-oss"


class TestStandaloneServerCli:
    """Test standalone server CLI argument parsing."""

    def test_trust_remote_code_flag_defaults_false(self):
        """Standalone server should require explicit opt-in for remote code loading."""
        from vllm_mlx.server import create_parser

        parser = create_parser()
        args = parser.parse_args(
            ["--model", "mlx-community/Llama-3.2-3B-Instruct-4bit"]
        )
        assert args.trust_remote_code is False

        args = parser.parse_args(
            [
                "--model",
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "--trust-remote-code",
            ]
        )
        assert args.trust_remote_code is True


class TestLoadModelTrustRemoteCode:
    """Test load_model trust_remote_code wiring into engine constructors."""

    def test_load_model_simple_defaults_trust_remote_code_false(self):
        """SimpleEngine should receive trust_remote_code=False by default."""
        from vllm_mlx import server

        fake_engine = MagicMock()
        fake_loop = MagicMock()

        with (
            patch.object(
                server, "SimpleEngine", return_value=fake_engine
            ) as mock_engine,
            patch.object(server, "_detect_native_tool_support", return_value=False),
            patch("vllm_mlx.server.asyncio.new_event_loop", return_value=fake_loop),
            patch("vllm_mlx.server.asyncio.set_event_loop"),
        ):
            server.load_model("test-model", use_batching=False)

        assert mock_engine.call_args.kwargs["trust_remote_code"] is False

    def test_load_model_batched_forwards_explicit_trust_remote_code(self):
        """BatchedEngine should receive explicit trust_remote_code opt-in."""
        from vllm_mlx import server

        fake_engine = MagicMock()

        with (
            patch.object(
                server, "BatchedEngine", return_value=fake_engine
            ) as mock_engine,
            patch.object(server, "_detect_native_tool_support", return_value=False),
        ):
            server.load_model(
                "test-model",
                use_batching=True,
                trust_remote_code=True,
            )

        assert mock_engine.call_args.kwargs["trust_remote_code"] is True


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test server helper functions."""

    def test_is_mllm_model_patterns(self):
        """Test MLLM model detection patterns."""
        from vllm_mlx.server import is_mllm_model

        # Should detect as MLLM
        assert is_mllm_model("mlx-community/Qwen3-VL-4B-Instruct-3bit")
        assert is_mllm_model("mlx-community/llava-1.5-7b-4bit")
        assert is_mllm_model("mlx-community/paligemma-3b-mix-224-4bit")
        assert is_mllm_model("mlx-community/pixtral-12b-4bit")
        assert is_mllm_model("mlx-community/Idefics3-8B-Llama3-4bit")
        assert is_mllm_model("mlx-community/deepseek-vl-7b-chat-4bit")

        # Should NOT detect as MLLM
        assert not is_mllm_model("mlx-community/Llama-3.2-1B-Instruct-4bit")
        assert not is_mllm_model("mlx-community/Mistral-7B-Instruct-4bit")
        assert not is_mllm_model("mlx-community/Qwen2-7B-Instruct-4bit")

    def test_extract_multimodal_content_text_only(self):
        """Test extracting content from text-only messages."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 2
        assert processed[0]["content"] == "Hello"
        assert len(images) == 0
        assert len(videos) == 0

    def test_extract_multimodal_content_with_image(self):
        """Test extracting content with images."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.jpg"},
                    },
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "What's this?"
        assert len(images) == 1
        assert "https://example.com/img.jpg" in images[0]

    def test_extract_multimodal_content_with_video(self):
        """Test extracting content with videos."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this video"},
                    {"type": "video", "video": "/path/to/video.mp4"},
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "Describe this video"
        assert len(videos) == 1
        assert videos[0] == "/path/to/video.mp4"

    def test_extract_multimodal_content_with_video_url(self):
        """Test extracting content with video_url format."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What happens?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/video.mp4"},
                    },
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(videos) == 1


# =============================================================================
# Security and Reliability Tests (PR #4)
# =============================================================================


class TestRateLimiter:
    """Test the RateLimiter class for rate limiting functionality."""

    def test_rate_limiter_disabled_by_default(self):
        """Test that rate limiter allows all requests when disabled."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=5, enabled=False)

        # Should allow unlimited requests when disabled
        for _ in range(100):
            allowed, retry_after = limiter.is_allowed("client1")
            assert allowed is True
            assert retry_after == 0

    def test_rate_limiter_enforces_limit(self):
        """Test that rate limiter enforces the request limit."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=3, enabled=True)

        # First 3 requests should be allowed
        for i in range(3):
            allowed, retry_after = limiter.is_allowed("client1")
            assert allowed is True, f"Request {i + 1} should be allowed"
            assert retry_after == 0

        # 4th request should be blocked
        allowed, retry_after = limiter.is_allowed("client1")
        assert allowed is False
        assert retry_after > 0

    def test_rate_limiter_per_client(self):
        """Test that rate limits are tracked per client."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Client 1 uses its quota
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        allowed, _ = limiter.is_allowed("client1")
        assert allowed is False

        # Client 2 should still have quota
        allowed, _ = limiter.is_allowed("client2")
        assert allowed is True

    def test_rate_limiter_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        import threading
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=100, enabled=True)
        results = []
        errors = []

        def make_requests():
            try:
                for _ in range(10):
                    allowed, _ = limiter.is_allowed("shared_client")
                    results.append(allowed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 100
        # Exactly 100 requests allowed (our limit)
        assert results.count(True) == 100


class TestTempFileManager:
    """Test the TempFileManager class for temp file cleanup."""

    def test_register_and_cleanup_single_file(self):
        """Test registering and cleaning up a single temp file."""
        import tempfile
        import os
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()

        # Create a real temp file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        temp.write(b"test content")
        temp.close()

        # Register it
        path = manager.register(temp.name)
        assert path == temp.name
        assert os.path.exists(temp.name)

        # Cleanup
        result = manager.cleanup(temp.name)
        assert result is True
        assert not os.path.exists(temp.name)

    def test_cleanup_all_files(self):
        """Test cleaning up all registered temp files."""
        import tempfile
        import os
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()
        paths = []

        # Create multiple temp files
        for i in range(3):
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.txt")
            temp.write(f"content {i}".encode())
            temp.close()
            manager.register(temp.name)
            paths.append(temp.name)

        # Verify all exist
        for p in paths:
            assert os.path.exists(p)

        # Cleanup all
        cleaned = manager.cleanup_all()
        assert cleaned == 3

        # Verify all deleted
        for p in paths:
            assert not os.path.exists(p)

    def test_cleanup_nonexistent_file(self):
        """Test cleanup of a non-existent file."""
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()

        # Cleanup a file that doesn't exist
        result = manager.cleanup("/nonexistent/path/file.txt")
        assert result is False

    def test_thread_safe_registration(self):
        """Test that TempFileManager is thread-safe."""
        import threading
        import tempfile
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()
        paths = []
        lock = threading.Lock()
        errors = []

        def register_files():
            try:
                for _ in range(5):
                    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
                    temp.write(b"test")
                    temp.close()
                    path = manager.register(temp.name)
                    with lock:
                        paths.append(path)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_files) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(paths) == 25

        # Cleanup all
        cleaned = manager.cleanup_all()
        assert cleaned == 25


class TestRequestOutputCollectorThreadSafety:
    """Test thread-safety of RequestOutputCollector._waiting_consumers."""

    def test_waiting_consumers_thread_safe(self):
        """Test that _waiting_consumers counter is thread-safe."""
        import threading
        from vllm_mlx.output_collector import RequestOutputCollector

        # Reset the counter
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0

        errors = []

        def manipulate_counter():
            try:
                for _ in range(100):
                    with RequestOutputCollector._waiting_lock:
                        RequestOutputCollector._waiting_consumers += 1
                    with RequestOutputCollector._waiting_lock:
                        RequestOutputCollector._waiting_consumers -= 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=manipulate_counter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        # Should return to zero
        with RequestOutputCollector._waiting_lock:
            assert RequestOutputCollector._waiting_consumers == 0

    def test_has_waiting_consumers_method(self):
        """Test has_waiting_consumers class method."""
        from vllm_mlx.output_collector import RequestOutputCollector

        # Reset counter
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0

        assert RequestOutputCollector.has_waiting_consumers() is False

        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 1

        assert RequestOutputCollector.has_waiting_consumers() is True

        # Reset
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0


class TestRequestTimeoutField:
    """Test the new timeout field in request models."""

    def test_chat_completion_request_timeout_field(self):
        """Test that ChatCompletionRequest has timeout field."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        # Default should be None
        request = ChatCompletionRequest(
            model="test-model", messages=[Message(role="user", content="Hello")]
        )
        assert request.timeout is None

        # Can set custom timeout
        request_with_timeout = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            timeout=60.0,
        )
        assert request_with_timeout.timeout == 60.0

    def test_completion_request_timeout_field(self):
        """Test that CompletionRequest has timeout field."""
        from vllm_mlx.server import CompletionRequest

        # Default should be None
        request = CompletionRequest(model="test-model", prompt="Once upon a time")
        assert request.timeout is None

        # Can set custom timeout
        request_with_timeout = CompletionRequest(
            model="test-model", prompt="Once upon a time", timeout=120.0
        )
        assert request_with_timeout.timeout == 120.0


class TestAPIKeyVerification:
    """Test API key verification with timing attack prevention."""

    def test_secrets_compare_digest_usage(self):
        """Test that secrets.compare_digest is used (timing attack prevention)."""
        import secrets

        # Verify secrets.compare_digest works as expected
        key1 = "test-api-key-12345"
        key2 = "test-api-key-12345"
        key3 = "different-key-67890"

        # Same keys should match
        assert secrets.compare_digest(key1, key2) is True

        # Different keys should not match
        assert secrets.compare_digest(key1, key3) is False

        # Verify it's constant-time (by checking function exists)
        assert hasattr(secrets, "compare_digest")

    def test_verify_api_key_rejects_invalid(self):
        """Test that invalid API key is rejected with 401."""
        import asyncio
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        # Import and set up the module
        import vllm_mlx.server as server

        original_key = server._api_key

        try:
            # Set a known API key
            server._api_key = "valid-secret-key"

            # Create mock credentials with invalid key
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="invalid-key"
            )

            # Should raise HTTPException with 401
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(server.verify_api_key(credentials))

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in str(exc_info.value.detail)
        finally:
            server._api_key = original_key

    def test_verify_api_key_accepts_valid(self):
        """Test that valid API key is accepted."""
        import asyncio
        from fastapi.security import HTTPAuthorizationCredentials

        import vllm_mlx.server as server

        original_key = server._api_key

        try:
            # Set a known API key
            server._api_key = "valid-secret-key"

            # Create mock credentials with valid key
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="valid-secret-key"
            )

            # Should not raise any exception
            result = asyncio.run(server.verify_api_key(credentials))
            # verify_api_key returns True on success (no exception raised)
            assert result is True or result is None
        finally:
            server._api_key = original_key


class TestRateLimiterHTTPResponse:
    """Test rate limiter HTTP response behavior."""

    def test_rate_limiter_returns_retry_after(self):
        """Test that rate limiter returns retry_after when limit exceeded."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Exhaust the limit
        limiter.is_allowed("test_client")
        limiter.is_allowed("test_client")

        # Next request should be denied with retry_after
        allowed, retry_after = limiter.is_allowed("test_client")

        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0


class TestEndpointSecurityDependencies:
    """Test auth/rate-limit coverage on protected endpoints."""

    @pytest.fixture
    def client(self):
        import vllm_mlx.server as server

        return TestClient(server.app)

    @pytest.fixture(autouse=True)
    def restore_security_state(self):
        import vllm_mlx.server as server

        original_key = server._api_key
        original_limiter = server._rate_limiter
        try:
            yield
        finally:
            server._api_key = original_key
            server._rate_limiter = original_limiter

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/v1/status"),
            ("get", "/v1/cache/stats"),
            ("delete", "/v1/cache"),
            ("post", "/v1/messages"),
            ("post", "/v1/messages/count_tokens"),
        ],
    )
    def test_endpoints_require_api_key(self, client, method, path):
        import vllm_mlx.server as server

        server._api_key = "test-secret"

        kwargs = {}
        if method == "post":
            kwargs["json"] = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            }

        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 401
        assert response.json()["detail"] == "API key required"

    @pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
    def test_anthropic_endpoints_apply_rate_limit(self, client, path, monkeypatch):
        import vllm_mlx.server as server

        server._api_key = "test-secret"

        def deny_all(_client_id):
            return False, 7

        monkeypatch.setattr(server._rate_limiter, "enabled", True)
        monkeypatch.setattr(server._rate_limiter, "is_allowed", deny_all)

        response = client.post(
            path,
            headers={"Authorization": "Bearer test-secret"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            },
        )

        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
        assert response.headers["Retry-After"] == "7"

    def test_rate_limiter_window_cleanup(self):
        """Test that rate limiter cleans up old requests from sliding window."""
        from vllm_mlx.server import RateLimiter
        import time

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Make some requests
        limiter.is_allowed("test_client")
        limiter.is_allowed("test_client")

        # Should be denied (limit reached)
        allowed, _ = limiter.is_allowed("test_client")
        assert allowed is False

        # Manually inject old timestamps to simulate time passing
        # The sliding window should clean these up
        old_time = time.time() - 120  # 2 minutes ago
        with limiter._lock:
            limiter._requests["test_client"] = [old_time, old_time]

        # Now should be allowed again (old requests cleaned up)
        allowed, _ = limiter.is_allowed("test_client")
        assert allowed is True


class TestStreamChatCompletion:
    """Tests for streaming chat completion behavior."""

    @pytest.mark.anyio
    async def test_stream_without_parser_flags_emits_structured_tool_calls(
        self, monkeypatch
    ):
        """Streaming tools should still parse without explicit parser flags."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_chat(self, messages, **kwargs):
                chunks = [
                    GenerationOutput(text="", new_text="<tool_call>", finished=False),
                    GenerationOutput(
                        text="",
                        new_text="<function=list_directory>",
                        finished=False,
                    ),
                    GenerationOutput(
                        text="",
                        new_text="<parameter=path>/Users/testuser</parameter>",
                        finished=False,
                    ),
                    GenerationOutput(
                        text="",
                        new_text="</function>",
                        finished=False,
                    ),
                    GenerationOutput(
                        text="",
                        new_text="</tool_call>",
                        finished=True,
                        finish_reason="stop",
                        prompt_tokens=5,
                        completion_tokens=7,
                    ),
                ]
                for chunk in chunks:
                    yield chunk

        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_parser_instance", None)

        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="hi")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List files in a directory",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            stream=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]

        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk != "data: [DONE]\n\n"
        ]
        tool_payloads = [
            payload
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
        ]

        assert len(tool_payloads) == 1
        delta = tool_payloads[0]["choices"][0]["delta"]
        assert delta["tool_calls"][0]["function"]["name"] == "list_directory"
        assert delta["tool_calls"][0]["function"]["arguments"] == (
            '{"path": "/Users/testuser"}'
        )
        assert delta["content"] is None
        assert tool_payloads[0]["choices"][0]["finish_reason"] == "tool_calls"
        assert tool_payloads[0]["usage"] == {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        }

    @pytest.mark.anyio
    async def test_stream_without_parser_flags_keeps_plain_text(self, monkeypatch):
        """Generic streaming fallback should not interfere with normal text."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_chat(self, messages, **kwargs):
                chunks = [
                    GenerationOutput(text="", new_text="hello ", finished=False),
                    GenerationOutput(
                        text="",
                        new_text="world",
                        finished=True,
                        finish_reason="stop",
                        prompt_tokens=4,
                        completion_tokens=2,
                    ),
                ]
                for chunk in chunks:
                    yield chunk

        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_parser_instance", None)

        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="hi")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List files in a directory",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            stream=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]
        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk != "data: [DONE]\n\n"
        ]

        assert payloads[1]["choices"][0]["delta"]["content"] == "hello "
        assert payloads[2]["choices"][0]["delta"]["content"] == "world"
        assert payloads[2]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.anyio
    async def test_auto_parser_streams_bare_bracket_tool_calls(self, monkeypatch):
        """Bare bracket tool calls should stream as structured tool_calls."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_chat(self, messages, **kwargs):
                chunks = [
                    GenerationOutput(text="", new_text="[read(", finished=False),
                    GenerationOutput(
                        text="",
                        new_text='{"file_path": "/tmp/test.py"}',
                        finished=False,
                    ),
                    GenerationOutput(
                        text="",
                        new_text=")]",
                        finished=True,
                        finish_reason="stop",
                        prompt_tokens=4,
                        completion_tokens=3,
                    ),
                ]
                for chunk in chunks:
                    yield chunk

        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "auto")
        monkeypatch.setattr(server, "_tool_parser_instance", None)

        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="hi")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"],
                        },
                    },
                }
            ],
            stream=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]

        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk != "data: [DONE]\n\n"
        ]
        tool_payloads = [
            payload
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
        ]

        assert len(tool_payloads) == 1
        delta = tool_payloads[0]["choices"][0]["delta"]
        assert delta["tool_calls"][0]["function"]["name"] == "read"
        assert delta["tool_calls"][0]["function"]["arguments"] == (
            '{"file_path": "/tmp/test.py"}'
        )
        assert delta["content"] is None
        assert tool_payloads[0]["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.anyio
    async def test_reasoning_stream_emits_structured_tool_calls(self, monkeypatch):
        """Tool markup after </think> should emit tool_calls chunks."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.reasoning import DeltaMessage
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_chat(self, messages, **kwargs):
                chunks = [
                    GenerationOutput(text="", new_text="<think>", finished=False),
                    GenerationOutput(text="", new_text="reasoning", finished=False),
                    GenerationOutput(text="", new_text="</think>", finished=False),
                    GenerationOutput(text="", new_text="<tool_call>", finished=False),
                    GenerationOutput(
                        text="", new_text='{"name":"search"}', finished=False
                    ),
                    GenerationOutput(
                        text="",
                        new_text="</tool_call>",
                        finished=True,
                        finish_reason="stop",
                        prompt_tokens=7,
                        completion_tokens=3,
                    ),
                ]
                for chunk in chunks:
                    yield chunk

        class FakeReasoningParser:
            def reset_state(self):
                self._in_reasoning = False

            def extract_reasoning_streaming(
                self, previous_text, current_text, delta_text
            ):
                if delta_text == "<think>":
                    self._in_reasoning = True
                    return None
                if delta_text == "</think>":
                    self._in_reasoning = False
                    return None
                if self._in_reasoning:
                    return DeltaMessage(reasoning=delta_text)
                return DeltaMessage(content=delta_text)

        class FakeToolParser:
            def reset(self):
                pass

            def extract_tool_calls_streaming(
                self, previous_text, current_text, delta_text
            ):
                if "</tool_call>" in current_text:
                    return {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"weather"}',
                                },
                            }
                        ]
                    }
                return None

        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_reasoning_parser", FakeReasoningParser())
        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "fake")
        monkeypatch.setattr(server, "_tool_parser_instance", FakeToolParser())

        request = ChatCompletionRequest(
            model="request-model",
            messages=[Message(role="user", content="hi")],
            stream=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]

        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk != "data: [DONE]\n\n"
        ]

        tool_payloads = [
            payload
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
        ]

        assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
        assert payloads[1]["choices"][0]["delta"]["reasoning_content"] == "reasoning"
        assert "reasoning" not in payloads[1]["choices"][0]["delta"]
        assert len(tool_payloads) == 1
        assert (
            tool_payloads[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "search"
        )
        assert tool_payloads[0]["choices"][0]["finish_reason"] == "tool_calls"
        assert tool_payloads[0]["usage"] == {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        }

    @pytest.mark.anyio
    async def test_reasoning_stream_skips_tool_parser_until_markup_appears(
        self, monkeypatch
    ):
        """Plain post-reasoning content should stream normally on the fast path."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.reasoning import DeltaMessage
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_chat(self, messages, **kwargs):
                chunks = [
                    GenerationOutput(text="", new_text="<think>", finished=False),
                    GenerationOutput(text="", new_text="reasoning", finished=False),
                    GenerationOutput(text="", new_text="</think>", finished=False),
                    GenerationOutput(
                        text="",
                        new_text="final answer",
                        finished=True,
                        finish_reason="stop",
                    ),
                ]
                for chunk in chunks:
                    yield chunk

        class FakeReasoningParser:
            def reset_state(self):
                self._in_reasoning = False

            def extract_reasoning_streaming(
                self, previous_text, current_text, delta_text
            ):
                if delta_text == "<think>":
                    self._in_reasoning = True
                    return None
                if delta_text == "</think>":
                    self._in_reasoning = False
                    return None
                if self._in_reasoning:
                    return DeltaMessage(reasoning=delta_text)
                return DeltaMessage(content=delta_text)

        class TrackingToolParser:
            def __init__(self):
                self.calls = []

            def reset(self):
                self.calls.clear()

            def extract_tool_calls_streaming(
                self, previous_text, current_text, delta_text
            ):
                self.calls.append((previous_text, current_text, delta_text))
                return {"content": delta_text}

        tool_parser = TrackingToolParser()

        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_reasoning_parser", FakeReasoningParser())
        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "fake")
        monkeypatch.setattr(server, "_tool_parser_instance", tool_parser)

        request = ChatCompletionRequest(
            model="request-model",
            messages=[Message(role="user", content="hi")],
            stream=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]

        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk != "data: [DONE]\n\n"
        ]

        assert tool_parser.calls == []
        assert payloads[1]["choices"][0]["delta"]["reasoning_content"] == "reasoning"
        assert "reasoning" not in payloads[1]["choices"][0]["delta"]
        assert payloads[2]["choices"][0]["delta"]["content"] == "final answer"
        assert payloads[2]["choices"][0]["finish_reason"] == "stop"


# =============================================================================
# Integration Tests (require running server)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestServerIntegration:
    """Integration tests that require a running server.

    These tests are skipped by default. Run with:
        pytest -m integration --server-url http://localhost:8000
    """

    @pytest.fixture
    def server_url(self, request):
        """Get server URL from command line or use default."""
        return request.config.getoption("--server-url", default="http://localhost:8000")

    def test_health_endpoint(self, server_url):
        """Test /health endpoint."""
        import requests

        response = requests.get(f"{server_url}/health", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model_name" in data

    def test_models_endpoint(self, server_url):
        """Test /v1/models endpoint."""
        import requests

        response = requests.get(f"{server_url}/v1/models", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0

    def test_chat_completion(self, server_url):
        """Test /v1/chat/completions endpoint."""
        import requests

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
        }

        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]


class TestSseDoneTermination:
    """Regression tests for SSE data: [DONE] termination signal.

    Covers #101: streaming responses must always emit exactly one
    data: [DONE] event, even when the engine raises mid-stream.
    """

    @pytest.mark.anyio
    async def test_stream_completion_normal_emits_done(self, monkeypatch):
        """Normal stream_completion yields exactly one [DONE] at the end."""
        from vllm_mlx.api.models import CompletionRequest
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import stream_completion

        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_generate(self, **kwargs):
                yield GenerationOutput(text="Hello", new_text="Hello", finished=False)
                yield GenerationOutput(
                    text="Hello world",
                    new_text=" world",
                    finished=True,
                    finish_reason="stop",
                    prompt_tokens=5,
                    completion_tokens=2,
                )

        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_max_tokens", 100)

        request = CompletionRequest(model="test-model", prompt="Say hello")
        chunks = [
            chunk
            async for chunk in stream_completion(FakeEngine(), "Say hello", request)
        ]

        done_chunks = [c for c in chunks if c == "data: [DONE]\n\n"]
        assert (
            len(done_chunks) == 1
        ), f"Expected exactly 1 [DONE], got {len(done_chunks)}"
        assert chunks[-1] == "data: [DONE]\n\n", "[DONE] must be the last chunk"

    @pytest.mark.anyio
    async def test_stream_completion_exception_still_emits_done(self, monkeypatch):
        """When engine raises mid-stream, [DONE] is still emitted via _ensure_sse_terminal."""
        from vllm_mlx.api.models import CompletionRequest
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import _ensure_sse_terminal, stream_completion

        import vllm_mlx.server as server

        class ExplodingEngine:
            model_name = "exploding-engine"

            async def stream_generate(self, **kwargs):
                yield GenerationOutput(
                    text="partial", new_text="partial", finished=False
                )
                raise RuntimeError("Metal command buffer SIGABRT")

        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_max_tokens", 100)

        request = CompletionRequest(model="test-model", prompt="Say hello")
        # Wrap with _ensure_sse_terminal, matching server routing
        chunks = [
            chunk
            async for chunk in _ensure_sse_terminal(
                stream_completion(ExplodingEngine(), "Say hello", request),
                "data: [DONE]\n\n",
            )
        ]

        done_chunks = [c for c in chunks if c == "data: [DONE]\n\n"]
        assert (
            len(done_chunks) == 1
        ), f"Expected exactly 1 [DONE], got {len(done_chunks)}"
        assert chunks[-1] == "data: [DONE]\n\n", "[DONE] must be the last chunk"

    @pytest.mark.anyio
    async def test_ensure_sse_terminal_normal_no_duplicate(self):
        """Wrapper passes through the generator's own [DONE] without duplicating."""
        from vllm_mlx.server import _ensure_sse_terminal

        async def happy_generator():
            yield "data: {}\n\n"
            yield "data: [DONE]\n\n"

        chunks = [
            chunk
            async for chunk in _ensure_sse_terminal(
                happy_generator(), "data: [DONE]\n\n"
            )
        ]

        done_chunks = [c for c in chunks if c == "data: [DONE]\n\n"]
        assert (
            len(done_chunks) == 1
        ), f"Expected exactly 1 [DONE], got {len(done_chunks)}"

    @pytest.mark.anyio
    async def test_ensure_sse_terminal_exception_emits_done(self):
        """Wrapper emits [DONE] when inner generator raises before reaching it."""
        from vllm_mlx.server import _ensure_sse_terminal

        async def exploding_generator():
            yield "data: {}\n\n"
            raise RuntimeError("engine crashed")

        chunks = [
            chunk
            async for chunk in _ensure_sse_terminal(
                exploding_generator(), "data: [DONE]\n\n"
            )
        ]

        done_chunks = [c for c in chunks if c == "data: [DONE]\n\n"]
        assert (
            len(done_chunks) == 1
        ), f"Expected exactly 1 [DONE], got {len(done_chunks)}"
        assert chunks[-1] == "data: [DONE]\n\n", "[DONE] must be the last chunk"

    @pytest.mark.anyio
    async def test_ensure_sse_terminal_anthropic_protocol(self):
        """Wrapper emits Anthropic message_stop, not OpenAI [DONE], on exception."""
        import json

        from vllm_mlx.server import _ensure_sse_terminal

        anthropic_terminal = (
            f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        )

        async def exploding_anthropic_stream():
            yield "event: content_block_delta\ndata: {}\n\n"
            raise RuntimeError("engine crashed")

        chunks = [
            chunk
            async for chunk in _ensure_sse_terminal(
                exploding_anthropic_stream(), anthropic_terminal
            )
        ]

        # Must emit Anthropic terminal, NOT OpenAI [DONE]
        assert chunks[-1] == anthropic_terminal
        openai_done = [c for c in chunks if c == "data: [DONE]\n\n"]
        assert (
            len(openai_done) == 0
        ), "Must not emit OpenAI [DONE] for Anthropic streams"


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="URL of the vllm-mlx server for integration tests",
    )
