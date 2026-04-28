# SPDX-License-Identifier: Apache-2.0
"""Server integration tests for lifecycle / residency behavior."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from types import SimpleNamespace

import pytest


async def _wait_for_resident_state(manager, model_key: str, state: str) -> None:
    """Poll until a resident reaches the expected state."""
    while manager.get_status(model_key)["state"] != state:
        await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def restore_server_globals():
    """Restore mutated server module globals between lifecycle tests."""
    import vllm_mlx.server as srv

    sentinel = object()
    global_names = (
        "_engine",
        "_model_name",
        "_model_path",
        "_default_model_key",
        "_default_max_tokens",
        "_default_timeout",
        "_default_temperature",
        "_default_top_p",
        "_force_mllm_model",
        "_auto_unload_idle_seconds",
        "_lazy_load_model",
        "_residency_manager",
        "_lifecycle_task",
        "_lifespan_active",
        "_mcp_manager",
        "_mcp_executor",
        "_embedding_engine",
        "_embedding_model_locked",
        "_api_key",
        "_auth_warning_logged",
        "_rate_limiter",
        "_reasoning_parser",
        "_enable_auto_tool_choice",
        "_tool_call_parser",
        "_tool_parser_instance",
        "_idle_unload_enabled",
    )
    snapshot = {name: getattr(srv, name, sentinel) for name in global_names}

    yield

    leaked_task = getattr(srv, "_lifecycle_task", None)
    original_task = snapshot["_lifecycle_task"]
    if (
        leaked_task is not sentinel
        and leaked_task is not None
        and leaked_task is not original_task
        and not leaked_task.done()
    ):
        leaked_task.cancel()

    for name, value in snapshot.items():
        if value is sentinel:
            if hasattr(srv, name):
                delattr(srv, name)
        else:
            setattr(srv, name, value)

    # _idle_unload_enabled is a lazily-created asyncio.Event bound to whatever
    # event loop was running when _get_idle_unload_event() was first called.
    # Reset to None so the next test gets a fresh Event on its own loop.
    srv._idle_unload_enabled = None


class TestLifecycleStatusEndpoints:
    """Lock in residency metadata surfaced by server status endpoints."""

    @pytest.mark.anyio
    async def test_status_reports_unloaded_resident_metadata(self, monkeypatch):
        """Status should surface residency details even when model is unloaded."""
        import vllm_mlx.server as srv

        fake_manager = SimpleNamespace(
            get_status=lambda model_key: {
                "model_key": model_key,
                "state": "unloaded",
                "active_requests": 0,
                "last_used_at": 1710200000.0,
                "loaded_at": None,
                "auto_unload_idle_seconds": 300,
            }
        )

        monkeypatch.setattr(srv, "_engine", None)
        monkeypatch.setattr(
            srv, "_model_name", "mlx-community/Qwen3-0.6B-8bit", raising=False
        )
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_residency_manager", fake_manager, raising=False)

        payload = await srv.status()

        assert payload["status"] == "not_loaded"
        assert payload["model"] == "mlx-community/Qwen3-0.6B-8bit"
        assert payload["residency"]["model_key"] == "default"
        assert payload["residency"]["state"] == "unloaded"
        assert payload["residency"]["active_requests"] == 0
        assert payload["residency"]["last_used_at"] == 1710200000.0
        assert payload["residency"]["loaded_at"] is None
        assert payload["residency"]["auto_unload_idle_seconds"] == 300
        assert payload["requests"] == []

    @pytest.mark.anyio
    async def test_health_exposes_residency_state_for_unloaded_model(self, monkeypatch):
        """Health should report lifecycle state, not only a loaded bool."""
        import vllm_mlx.server as srv

        fake_manager = SimpleNamespace(
            get_status=lambda model_key: {
                "model_key": model_key,
                "state": "unloaded",
                "active_requests": 0,
                "last_used_at": 1710200000.0,
                "loaded_at": None,
                "auto_unload_idle_seconds": 120,
            }
        )

        monkeypatch.setattr(srv, "_engine", None)
        monkeypatch.setattr(
            srv,
            "_model_name",
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            raising=False,
        )
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_residency_manager", fake_manager, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None)

        payload = await srv.health()

        assert payload["status"] == "healthy"
        assert payload["model_loaded"] is False
        assert payload["model_name"] == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert payload["residency_state"] == "unloaded"
        assert payload["active_requests"] == 0
        assert payload["last_used_at"] == 1710200000.0
        assert payload["loaded_at"] is None
        assert payload["auto_unload_idle_seconds"] == 120

    @pytest.mark.anyio
    async def test_failed_resident_surfaces_as_unhealthy_and_failed(self, monkeypatch):
        """Public status should not leak backend model identity or raw errors."""
        import vllm_mlx.server as srv

        fake_manager = SimpleNamespace(
            get_status=lambda model_key: {
                "model_key": model_key,
                "model_name": "/tmp/private-local-model",
                "state": "failed",
                "active_requests": 0,
                "last_used_at": 1710200000.0,
                "loaded_at": None,
                "last_error": "reload boom",
                "auto_unload_idle_seconds": 120,
            }
        )

        monkeypatch.setattr(srv, "_engine", None)
        monkeypatch.setattr(
            srv,
            "_model_name",
            "friendly-model",
            raising=False,
        )
        monkeypatch.setattr(
            srv,
            "_model_path",
            "/tmp/private-local-model",
            raising=False,
        )
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_residency_manager", fake_manager, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None)

        health_payload = await srv.health()
        status_payload = await srv.status()

        assert health_payload["status"] == "unhealthy"
        assert health_payload["model_loaded"] is False
        assert health_payload["residency_state"] == "failed"
        # /health surfaces a sanitized error category for failed residents
        assert health_payload["last_error"] == "model_load_failed"
        assert status_payload["status"] == "not_loaded"
        assert status_payload["model"] == "friendly-model"
        assert status_payload["residency"]["state"] == "failed"
        assert status_payload["residency"]["model_name"] == "friendly-model"
        # /v1/status surfaces a generic error indicator, not raw exception text
        assert status_payload["residency"]["last_error"] == "model_load_failed"
        assert status_payload["requests"] == []

    @pytest.mark.anyio
    async def test_health_preserves_mllm_type_when_resident_is_unloaded(
        self, monkeypatch
    ):
        """Unloaded multimodal residents should still report model_type=mllm."""
        import vllm_mlx.server as srv

        fake_manager = SimpleNamespace(
            get_status=lambda model_key: {
                "model_key": model_key,
                "state": "unloaded",
                "active_requests": 0,
                "last_used_at": 1710200000.0,
                "loaded_at": None,
                "auto_unload_idle_seconds": 120,
            }
        )

        monkeypatch.setattr(srv, "_engine", None)
        monkeypatch.setattr(
            srv,
            "_model_name",
            "mlx-community/gemma-3-4b-it-4bit",
            raising=False,
        )
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_residency_manager", fake_manager, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None)

        payload = await srv.health()

        assert payload["model_type"] == "mllm"

    @pytest.mark.anyio
    async def test_health_uses_model_path_for_unloaded_served_alias_mllm(
        self, monkeypatch
    ):
        """Served aliases should not hide unloaded multimodal model type."""
        import vllm_mlx.server as srv

        fake_manager = SimpleNamespace(
            get_status=lambda model_key: {
                "model_key": model_key,
                "model_name": "mlx-community/gemma-3-4b-it-4bit",
                "state": "unloaded",
                "active_requests": 0,
                "last_used_at": 1710200000.0,
                "loaded_at": None,
                "auto_unload_idle_seconds": 120,
            }
        )

        monkeypatch.setattr(srv, "_engine", None)
        monkeypatch.setattr(srv, "_model_name", "prod-chat", raising=False)
        monkeypatch.setattr(
            srv,
            "_model_path",
            "mlx-community/gemma-3-4b-it-4bit",
            raising=False,
        )
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_residency_manager", fake_manager, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None)

        payload = await srv.health()

        assert payload["model_type"] == "mllm"

    @pytest.mark.anyio
    async def test_health_uses_force_mllm_for_unloaded_local_model(self, monkeypatch):
        """force_mllm should survive the unloaded-resident health fallback."""
        import vllm_mlx.server as srv

        monkeypatch.setattr(srv, "_mcp_manager", None)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        srv.load_model(
            "/tmp/local-model",
            force_mllm=True,
            auto_unload_idle_seconds=60,
        )
        monkeypatch.setattr(srv, "_engine", None)

        payload = await srv.health()

        assert payload["model_type"] == "mllm"


class TestCompletionStreamingRelease:
    """Verify the completion endpoint releases residency on all paths."""

    @pytest.mark.anyio
    async def test_completion_nonstreaming_error_releases_active_request(
        self, monkeypatch
    ):
        """Non-streaming completion errors must still release the active request."""
        import vllm_mlx.server as srv

        releases = {"count": 0}
        acquires = {"count": 0}

        class FakeEngine:
            preserve_native_tool_format = False
            is_mllm = False

            async def start(self):
                pass

            async def stop(self):
                pass

            async def generate(self, **kwargs):
                raise RuntimeError("generation failed")

        async def fake_acquire(
            raw_request, *, total_timeout=None, deadline=None, count_activity=True
        ):
            acquires["count"] += 1
            return FakeEngine()

        async def fake_release(*, count_activity=True):
            releases["count"] += 1

        monkeypatch.setattr(srv, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(srv, "_release_default_engine", fake_release)
        monkeypatch.setattr(srv, "_model_name", "test-model")

        class FakeRequest:
            async def is_disconnected(self):
                return False

        request = SimpleNamespace(
            model="test-model",
            prompt="hello",
            stream=False,
            max_tokens=10,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
            specprefill=None,
            specprefill_keep_pct=None,
            stop=None,
            timeout=60.0,
        )

        with pytest.raises(RuntimeError, match="generation failed"):
            await srv.create_completion(request, FakeRequest())

        assert acquires["count"] == 1
        assert (
            releases["count"] == 1
        ), "Non-streaming completion must release residency on generation errors"

    @pytest.mark.anyio
    async def test_completion_streaming_release_matches_chat_pattern(self, monkeypatch):
        """Streaming completion should use try/finally like chat completion does."""
        import vllm_mlx.server as srv

        # The chat completion endpoint uses a release_on_exit flag with try/finally.
        # The completion endpoint should follow the same pattern for consistency
        # and safety. This test verifies that the streaming path eventually
        # calls release via the cleanup callback.
        releases = {"count": 0}

        class FakeEngine:
            preserve_native_tool_format = False
            is_mllm = False

            async def start(self):
                pass

            async def stop(self):
                pass

            async def stream_generate(self, **kwargs):
                yield SimpleNamespace(
                    text="done",
                    new_text="done",
                    finish_reason="stop",
                    completion_tokens=1,
                    prompt_tokens=1,
                    finished=True,
                )

        async def fake_acquire(
            raw_request, *, total_timeout=None, deadline=None, count_activity=True
        ):
            return FakeEngine()

        async def fake_release(*, count_activity=True):
            releases["count"] += 1

        monkeypatch.setattr(srv, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(srv, "_release_default_engine", fake_release)
        monkeypatch.setattr(srv, "_model_name", "test-model")

        class FakeRequest:
            async def is_disconnected(self):
                return False

        request = SimpleNamespace(
            model="test-model",
            prompt="hello",
            stream=True,
            max_tokens=10,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
            specprefill=None,
            specprefill_keep_pct=None,
            stop=None,
            timeout=None,
        )

        response = await srv.create_completion(request, FakeRequest())

        # Iterate to completion
        async for _ in response.body_iterator:
            pass

        assert (
            releases["count"] == 1
        ), "Streaming completion must release residency via cleanup callback"


class TestStatusEndpointEngineRace:
    """Verify status/health endpoints handle engine being None."""

    @pytest.mark.anyio
    async def test_status_endpoint_returns_not_loaded_when_engine_is_none(
        self, monkeypatch
    ):
        """/v1/status should not 500 if engine is unloaded between check and use."""
        import vllm_mlx.server as srv

        call_count = {"n": 0}

        class DisappearingEngine:
            """Engine that disappears after the null check."""

            def get_stats(self):
                call_count["n"] += 1
                return {
                    "running": True,
                    "uptime_seconds": 10,
                    "steps_executed": 0,
                    "num_running": 0,
                    "num_waiting": 0,
                    "num_requests_processed": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "metal_active_memory_gb": 0,
                    "metal_peak_memory_gb": 0,
                    "metal_cache_memory_gb": 0,
                    "requests": [],
                }

        # Set engine to a real object, then unload it mid-call by patching
        engine = DisappearingEngine()
        monkeypatch.setattr(srv, "_engine", engine)
        monkeypatch.setattr(srv, "_model_name", "test")
        monkeypatch.setattr(srv, "_residency_manager", None)
        monkeypatch.setattr(srv, "_default_model_key", None)

        # Normal case: should work
        result = await srv.status()
        assert result["status"] == "running"

        # Now simulate the race: engine becomes None after the check
        monkeypatch.setattr(srv, "_engine", None)
        result = await srv.status()
        assert result["status"] == "not_loaded"

    @pytest.mark.anyio
    async def test_health_endpoint_handles_engine_none(self, monkeypatch):
        """/health should not 500 when engine is None."""
        import vllm_mlx.server as srv

        monkeypatch.setattr(srv, "_engine", None)
        monkeypatch.setattr(srv, "_model_name", "test")
        monkeypatch.setattr(srv, "_model_path", None)
        monkeypatch.setattr(srv, "_force_mllm_model", False)
        monkeypatch.setattr(srv, "_mcp_manager", None)
        monkeypatch.setattr(srv, "_residency_manager", None)
        monkeypatch.setattr(srv, "_default_model_key", None)

        result = await srv.health()
        assert result["status"] == "healthy"
        assert result["model_loaded"] is False


class TestToolParserUsesLocalEngine:
    """Tool parser should initialize from the request-local engine."""

    @pytest.mark.anyio
    async def test_chat_completion_initializes_parser_from_acquired_engine(
        self, monkeypatch
    ):
        """Chat completion should seed parser state from the acquired engine."""
        from vllm_mlx.engine.base import GenerationOutput
        import vllm_mlx.server as srv

        parser_tokenizers = []

        class FakeParser:
            def __init__(self, tokenizer=None):
                parser_tokenizers.append(tokenizer)

            def reset(self):
                return None

            def extract_tool_calls(self, output_text, request_dict=None):
                return SimpleNamespace(
                    tools_called=False,
                    tool_calls=[],
                    content=output_text,
                )

        class FakeEngine:
            preserve_native_tool_format = False
            is_mllm = False

            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            async def chat(self, **kwargs):
                return GenerationOutput(
                    text="hello",
                    completion_tokens=1,
                    prompt_tokens=1,
                )

        local_engine = FakeEngine("tok-local")

        async def fake_acquire(
            raw_request, *, total_timeout=None, deadline=None, count_activity=True
        ):
            return local_engine

        async def fake_release(*, count_activity=True):
            return None

        monkeypatch.setattr(srv, "_validate_model_name", lambda _m: None)
        monkeypatch.setattr(srv, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(srv, "_release_default_engine", fake_release)
        monkeypatch.setattr(srv, "_model_name", "served-model")
        monkeypatch.setattr(srv, "_default_max_tokens", 32)
        monkeypatch.setattr(srv, "_engine", None)
        monkeypatch.setattr(srv, "_reasoning_parser", None)
        monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(srv, "_tool_call_parser", "fake")
        monkeypatch.setattr(srv, "_tool_parser_instance", None)
        monkeypatch.setattr(
            srv.ToolParserManager,
            "get_tool_parser",
            lambda name: FakeParser,
        )

        class FakeRawRequest:
            async def is_disconnected(self):
                return False

        request = srv.ChatCompletionRequest(
            model="user-sent-model-name",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            tool_choice="auto",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        await srv.create_chat_completion(request, FakeRawRequest())

        assert parser_tokenizers == ["tok-local"], (
            "Parser init should use the request-local engine acquired for "
            "this request, not the stale global _engine"
        )


class TestLifecycleFailureHandling:
    """Regression coverage for lifecycle failure paths."""

    @pytest.mark.anyio
    async def test_anthropic_validation_error_does_not_acquire_resident(
        self, monkeypatch
    ):
        """Malformed Anthropic payloads should not touch residency at all."""
        from pydantic import ValidationError

        import vllm_mlx.server as srv

        calls = {"acquires": 0, "releases": 0}

        class FakeRequest:
            async def json(self):
                return {}

        class FakeEngine:
            preserve_native_tool_format = False

        async def fake_acquire(
            raw_request, *, total_timeout=None, deadline=None, count_activity=True
        ):
            calls["acquires"] += 1
            return FakeEngine()

        async def fake_release(*, count_activity=True):
            calls["releases"] += 1

        monkeypatch.setattr(srv, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(srv, "_release_default_engine", fake_release)

        with pytest.raises(ValidationError):
            await srv.create_anthropic_message(FakeRequest())

        assert calls["acquires"] == 0
        assert calls["releases"] == 0

    @pytest.mark.anyio
    async def test_chat_completion_prep_error_releases_resident(self, monkeypatch):
        """Prep failures after acquire should still release chat residency."""
        import vllm_mlx.server as srv

        calls = {"acquires": 0, "releases": 0}

        class FakeEngine:
            is_mllm = False
            preserve_native_tool_format = False

        async def fake_acquire(
            raw_request, *, total_timeout=None, deadline=None, count_activity=True
        ):
            calls["acquires"] += 1
            return FakeEngine()

        async def fake_release(*, count_activity=True):
            calls["releases"] += 1

        def fake_extract(messages, preserve_native_format):
            return ([{"role": "user", "content": "hi"}], [], [], [])

        def fake_convert_tools(_tools):
            raise RuntimeError("boom")

        monkeypatch.setattr(srv, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(srv, "_release_default_engine", fake_release)
        monkeypatch.setattr(srv, "extract_multimodal_content", fake_extract)
        monkeypatch.setattr(srv, "convert_tools_for_template", fake_convert_tools)

        request = SimpleNamespace(
            stream=False,
            messages=[SimpleNamespace(role="user", content="hi")],
            model="mlx-community/Qwen3-0.6B-8bit",
            max_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
            response_format=None,
            tools=[{"type": "function"}],
            tool_choice=None,
            enable_thinking=None,
            video_fps=None,
            video_max_frames=None,
            specprefill=None,
            specprefill_keep_pct=None,
            chat_template_kwargs=None,
            stop=None,
            timeout=None,
        )

        with pytest.raises(RuntimeError, match="boom"):
            await srv.create_chat_completion(request, SimpleNamespace())

        assert calls["acquires"] == 1
        assert calls["releases"] == 1

    @pytest.mark.anyio
    async def test_request_acquire_helper_disconnect_covers_final_lease(
        self, monkeypatch
    ):
        """Disconnects should abort even if residency is stalled in final acquire()."""
        import vllm_mlx.server as srv

        acquire_cancelled = asyncio.Event()
        lease_gate = asyncio.Event()

        class FakeEngine:
            preserve_native_tool_format = False

        class FakeRequest:
            async def is_disconnected(self):
                return True

        async def fake_acquire(model_key):
            try:
                await lease_gate.wait()
            except asyncio.CancelledError:
                acquire_cancelled.set()
                raise
            return FakeEngine()

        fake_manager = SimpleNamespace(acquire=fake_acquire)

        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", fake_manager, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)

        total_timeout, deadline = srv._start_request_budget(60.0)
        result = await srv._acquire_default_engine_for_request(
            FakeRequest(),
            total_timeout=total_timeout,
            deadline=deadline,
        )

        assert result is None
        await asyncio.wait_for(acquire_cancelled.wait(), timeout=1.0)

    @pytest.mark.anyio
    async def test_wait_with_disconnect_reports_total_request_timeout(self):
        """Timeout details should reflect the configured request budget, not the sub-step."""
        from fastapi import HTTPException

        import vllm_mlx.server as srv

        class FakeRawRequest:
            async def json(self):
                return {
                    "model": "mlx-community/Qwen3-0.6B-8bit",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "max_tokens": 16,
                }

            async def is_disconnected(self):
                return False

        with pytest.raises(HTTPException, match="10.0 seconds"):
            await srv._wait_with_disconnect(
                asyncio.sleep(0.05),
                FakeRawRequest(),
                timeout=0.01,
                timeout_detail_seconds=10.0,
                poll_interval=0.001,
            )

    @pytest.mark.anyio
    async def test_lifespan_startup_failure_cleans_up_loaded_resident_and_loop(
        self, monkeypatch
    ):
        """Startup failures before yield should not leak lifecycle tasks or loaded residents."""
        import vllm_mlx.server as srv

        stopped = {"count": 0}

        class FakeEngine:
            preserve_native_tool_format = False

            async def start(self):
                return None

            async def stop(self):
                stopped["count"] += 1

        class FakeRawRequest:
            async def json(self):
                return {
                    "model": "mlx-community/Qwen3-0.6B-8bit",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "max_tokens": 16,
                }

            async def is_disconnected(self):
                return False

        async def fake_engine_factory(spec):
            return FakeEngine()

        async def fake_init_mcp(config_path):
            raise RuntimeError("mcp boom")

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "init_mcp", fake_init_mcp)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)
        monkeypatch.setenv("VLLM_MLX_MCP_CONFIG", "/tmp/fake-mcp.json")

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=False,
        )

        lifespan = srv.lifespan(srv.app)
        try:
            with pytest.raises(RuntimeError, match="mcp boom"):
                await lifespan.__anext__()

            status = srv._get_lifecycle_status()
            assert srv._lifecycle_task is None
            assert srv._engine is None
            assert status is not None
            assert status["state"] == "unloaded"
            assert stopped["count"] == 1
        finally:
            if srv._lifecycle_task is not None:
                srv._lifecycle_task.cancel()
                with suppress(asyncio.CancelledError):
                    await srv._lifecycle_task
                srv._lifecycle_task = None
            if srv._residency_manager is not None:
                with suppress(Exception):
                    await srv._residency_manager.shutdown()
                srv._sync_engine_from_residency()
            with suppress(Exception):
                await lifespan.aclose()

    @pytest.mark.anyio
    async def test_lifespan_startup_failure_preserves_original_exception(
        self, monkeypatch, caplog
    ):
        """Cleanup failures should not replace the original startup exception."""
        import vllm_mlx.server as srv

        class FakeEngine:
            preserve_native_tool_format = False

            async def start(self):
                return None

            async def stop(self):
                raise RuntimeError("stop boom")

        async def fake_engine_factory(spec):
            return FakeEngine()

        async def fake_init_mcp(config_path):
            raise RuntimeError("mcp boom")

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "init_mcp", fake_init_mcp)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)
        monkeypatch.setenv("VLLM_MLX_MCP_CONFIG", "/tmp/fake-mcp.json")

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=False,
        )

        lifespan = srv.lifespan(srv.app)
        try:
            caplog.clear()
            with pytest.raises(RuntimeError, match="mcp boom") as excinfo:
                await lifespan.__anext__()
            assert excinfo.value.__cause__ is None
            assert (
                "Lifecycle cleanup failed while preserving the original exception"
                in caplog.text
            )
            assert "stop boom" in caplog.text
        finally:
            if srv._lifecycle_task is not None:
                srv._lifecycle_task.cancel()
                with suppress(asyncio.CancelledError):
                    await srv._lifecycle_task
                srv._lifecycle_task = None
            if srv._residency_manager is not None:
                with suppress(Exception):
                    await srv._residency_manager.shutdown()
                srv._sync_engine_from_residency()
            with suppress(Exception):
                await lifespan.aclose()

    @pytest.mark.anyio
    async def test_lifespan_startup_failure_keeps_live_runtime_guarded_if_cleanup_fails(
        self, monkeypatch
    ):
        """Startup failures should not orphan a live runtime when cleanup also fails."""
        import vllm_mlx.server as srv

        class FakeEngine:
            preserve_native_tool_format = False

            async def start(self):
                return None

            async def stop(self):
                raise RuntimeError("stop boom")

        async def fake_engine_factory(spec):
            return FakeEngine()

        async def fake_init_mcp(config_path):
            raise RuntimeError("mcp boom")

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "init_mcp", fake_init_mcp)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)
        monkeypatch.setenv("VLLM_MLX_MCP_CONFIG", "/tmp/fake-mcp.json")

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=False,
        )

        lifespan = srv.lifespan(srv.app)
        try:
            with pytest.raises(RuntimeError, match="mcp boom"):
                await lifespan.__anext__()

            status = srv._get_lifecycle_status()
            assert srv._engine is not None
            assert status is not None
            assert status["state"] == "loaded"
            with pytest.raises(RuntimeError, match="existing residency manager"):
                srv.load_model(
                    "mlx-community/Qwen3-0.6B-8bit",
                    auto_unload_idle_seconds=60.0,
                )
        finally:
            if srv._lifecycle_task is not None:
                srv._lifecycle_task.cancel()
                with suppress(asyncio.CancelledError):
                    await srv._lifecycle_task
                srv._lifecycle_task = None
            if srv._residency_manager is not None:
                with suppress(Exception):
                    await srv._residency_manager.shutdown()
                srv._sync_engine_from_residency()
            with suppress(Exception):
                await lifespan.aclose()

    @pytest.mark.anyio
    async def test_eager_residency_registers_unloaded_resident_before_lifespan_startup(
        self, monkeypatch
    ):
        """Pre-lifespan lifecycle setup should stay unloaded until startup runs."""
        from fastapi import HTTPException
        import vllm_mlx.server as srv

        create_calls = {"count": 0}

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                return None

        async def fake_engine_factory(spec):
            create_calls["count"] += 1
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        try:
            srv.load_model(
                "mlx-community/Qwen3-0.6B-8bit",
                auto_unload_idle_seconds=60.0,
                lazy_load_model=False,
            )

            lifecycle = srv._get_lifecycle_status()
            health_payload = await srv.health()
            status_payload = await srv.status()

            assert create_calls["count"] == 0
            assert srv._engine is None
            assert srv._residency_manager is not None
            assert lifecycle is not None
            assert lifecycle["state"] == "unloaded"
            assert lifecycle["active_requests"] == 0
            assert health_payload["model_loaded"] is False
            assert health_payload["residency_state"] == "unloaded"
            assert status_payload["status"] == "not_loaded"
            assert status_payload["residency"]["state"] == "unloaded"

            with pytest.raises(HTTPException, match="Model not loaded"):
                srv.get_engine()
        finally:
            if srv._residency_manager is not None:
                with suppress(Exception):
                    await srv._residency_manager.shutdown()
                srv._sync_engine_from_residency()

    @pytest.mark.anyio
    async def test_load_model_preserves_scheduler_config_when_enabling_residency(
        self, monkeypatch
    ):
        """Residency load should preserve batching config through engine factory."""
        import vllm_mlx.server as srv

        scheduler_config = SimpleNamespace(
            enable_mtp=True,
            mtp_num_draft_tokens=4,
            mtp_optimistic=True,
        )
        captured = {}

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                return None

        async def fake_engine_factory(spec):
            captured["spec"] = spec
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        try:
            srv.load_model(
                "mlx-community/Qwen3-0.6B-8bit",
                use_batching=True,
                scheduler_config=scheduler_config,
                stream_interval=7,
                auto_unload_idle_seconds=60.0,
                lazy_load_model=True,
            )

            manager = srv._residency_manager
            assert manager is not None

            resident = manager._residents["default"]
            assert resident.spec.use_batching is True
            assert resident.spec.scheduler_config is scheduler_config
            assert resident.spec.stream_interval == 7
            assert manager.auto_unload_idle_seconds == 60.0

            engine = await srv._acquire_default_engine()

            assert isinstance(engine, FakeEngine)
            assert captured["spec"].use_batching is True
            assert captured["spec"].scheduler_config is scheduler_config
            assert captured["spec"].stream_interval == 7
            assert srv._engine is engine

            await srv._release_default_engine()
        finally:
            if srv._residency_manager is not None:
                with suppress(Exception):
                    await srv._residency_manager.shutdown()
                srv._sync_engine_from_residency()

    @pytest.mark.anyio
    async def test_eager_residency_stays_loaded_until_startup_ready(self, monkeypatch):
        """Eager residency should not auto-unload before startup reaches readiness."""
        import vllm_mlx.server as srv

        now = {"value": 1000.0}
        stopped = {"count": 0}

        class FakeEngine:
            preserve_native_tool_format = False

            async def start(self):
                return None

            async def stop(self):
                stopped["count"] += 1

        async def fake_engine_factory(spec):
            return FakeEngine()

        async def fake_init_mcp(config_path):
            now["value"] += 120.0
            await asyncio.sleep(0.05)

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "init_mcp", fake_init_mcp)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)
        monkeypatch.setenv("VLLM_MLX_MCP_CONFIG", "/tmp/fake-mcp.json")

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=0.02,
            lazy_load_model=False,
        )
        assert srv._residency_manager is not None
        srv._residency_manager._time_fn = lambda: now["value"]

        lifespan = srv.lifespan(srv.app)
        try:
            await lifespan.__anext__()

            status = srv._get_lifecycle_status()
            assert srv._engine is not None
            assert status is not None
            assert status["state"] == "loaded"
            assert stopped["count"] == 0
        finally:
            if srv._lifecycle_task is not None:
                srv._lifecycle_task.cancel()
                with suppress(asyncio.CancelledError):
                    await srv._lifecycle_task
                srv._lifecycle_task = None
            if srv._residency_manager is not None:
                with suppress(Exception):
                    await srv._residency_manager.shutdown()
                srv._sync_engine_from_residency()
            with suppress(Exception):
                await lifespan.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("engine_kind", ["simple", "batched-llm", "batched-mllm"])
    async def test_eager_engine_start_cancellation_cleans_prepared_state(
        self, monkeypatch, engine_kind
    ):
        """Cancelling eager engine startup should not leave prepared model state behind."""
        import threading

        if engine_kind == "simple":
            import vllm_mlx.engine.simple as engine_mod

            engine = engine_mod.SimpleEngine("fake-model")
        else:
            import vllm_mlx.engine.batched as engine_mod

            engine = engine_mod.BatchedEngine("fake-model")
            if engine_kind == "batched-llm":

                async def unexpected_start_llm():
                    raise AssertionError("_start_llm should not run after cancellation")

                monkeypatch.setattr(engine, "_is_mllm", False, raising=False)
                monkeypatch.setattr(
                    engine, "_start_llm", unexpected_start_llm, raising=False
                )
            else:

                async def unexpected_start_mllm():
                    raise AssertionError(
                        "_start_mllm should not run after cancellation"
                    )

                monkeypatch.setattr(engine, "_is_mllm", True, raising=False)
                monkeypatch.setattr(
                    engine, "_start_mllm", unexpected_start_mllm, raising=False
                )

        prepare_entered = threading.Event()
        allow_prepare_finish = threading.Event()
        prepare_finished = threading.Event()

        def fake_prepare():
            prepare_entered.set()
            allow_prepare_finish.wait()
            engine._model = object()
            if engine_kind == "batched-mllm":
                engine._processor = object()
                engine._mllm_instance = object()
            elif hasattr(engine, "_tokenizer"):
                engine._tokenizer = object()
            prepare_finished.set()

        monkeypatch.setattr(engine, "prepare_for_start", fake_prepare)

        start_task = asyncio.create_task(engine.start())
        try:
            assert await asyncio.wait_for(
                asyncio.to_thread(prepare_entered.wait, 1.0),
                timeout=1.0,
            )

            start_task.cancel()
            allow_prepare_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await start_task

            assert await asyncio.wait_for(
                asyncio.to_thread(prepare_finished.wait, 1.0),
                timeout=1.0,
            )
            assert engine._loaded is False
            assert engine._model is None
            if engine_kind == "batched-mllm":
                assert engine._processor is None
                assert engine._mllm_instance is None
            elif hasattr(engine, "_tokenizer"):
                assert engine._tokenizer is None
        finally:
            allow_prepare_finish.set()
            if not start_task.done():
                start_task.cancel()
                with suppress(asyncio.CancelledError):
                    await start_task
            with suppress(Exception):
                await engine.stop()

    @pytest.mark.anyio
    async def test_simple_start_cancellation_preserves_cancelled_error_when_stop_fails(
        self, monkeypatch, caplog
    ):
        """Simple eager startup should still surface cancellation if stop() fails."""
        import threading

        import vllm_mlx.engine.simple as engine_mod

        engine = engine_mod.SimpleEngine("fake-model")
        prepare_entered = threading.Event()
        allow_prepare_finish = threading.Event()
        stop_calls = {"count": 0}

        def fake_prepare():
            prepare_entered.set()
            allow_prepare_finish.wait()
            engine._model = object()

        async def failing_stop():
            stop_calls["count"] += 1
            raise RuntimeError("stop boom")

        monkeypatch.setattr(engine, "prepare_for_start", fake_prepare)
        monkeypatch.setattr(engine, "stop", failing_stop)

        start_task = asyncio.create_task(engine.start())
        try:
            assert await asyncio.wait_for(
                asyncio.to_thread(prepare_entered.wait, 1.0),
                timeout=1.0,
            )

            caplog.set_level("ERROR")
            caplog.clear()
            start_task.cancel()
            allow_prepare_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await start_task

            assert stop_calls["count"] == 1
            assert any(
                record.levelname == "ERROR"
                and "startup cleanup failed while preserving cancellation"
                in record.getMessage()
                for record in caplog.records
            )
        finally:
            allow_prepare_finish.set()
            if not start_task.done():
                start_task.cancel()
                with suppress(asyncio.CancelledError):
                    await start_task

    @pytest.mark.anyio
    async def test_run_blocking_startup_work_waits_for_thread_under_repeated_cancel(
        self,
    ):
        """Repeated cancellation should not return before blocking startup work finishes."""
        import threading

        from vllm_mlx.engine.base import run_blocking_startup_work

        entered = threading.Event()
        allow_finish = threading.Event()
        finished = threading.Event()

        def blocking_work():
            entered.set()
            allow_finish.wait()
            # Keep a deterministic post-release window so pre-fix behavior
            # can still return cancellation before thread completion.
            time.sleep(0.05)
            finished.set()

        task = asyncio.create_task(run_blocking_startup_work(blocking_work))
        try:
            assert await asyncio.wait_for(
                asyncio.to_thread(entered.wait, 1.0),
                timeout=1.0,
            )

            task.cancel()
            await asyncio.sleep(0)
            task.cancel()

            allow_finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert finished.is_set() is True
        finally:
            allow_finish.set()
            assert await asyncio.wait_for(
                asyncio.to_thread(finished.wait, 1.0),
                timeout=1.0,
            )

    @pytest.mark.anyio
    async def test_blocking_cache_io_waits_for_thread_under_repeated_cancel(
        self,
    ):
        """Repeated cancellation should not return before cache I/O thread finishes."""
        import threading

        import vllm_mlx.server as srv

        entered = threading.Event()
        allow_finish = threading.Event()
        finished = threading.Event()

        class FakeEngine:
            pass

        def blocking_io(_engine):
            entered.set()
            allow_finish.wait()
            # Keep a deterministic post-release window so pre-fix behavior
            # can still return cancellation before thread completion.
            time.sleep(0.05)
            finished.set()

        task = asyncio.create_task(
            srv._run_blocking_engine_cache_io(blocking_io, FakeEngine())
        )
        try:
            assert await asyncio.wait_for(
                asyncio.to_thread(entered.wait, 1.0),
                timeout=1.0,
            )

            task.cancel()
            await asyncio.sleep(0)
            task.cancel()

            allow_finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert finished.is_set() is True
        finally:
            allow_finish.set()
            assert await asyncio.wait_for(
                asyncio.to_thread(finished.wait, 1.0),
                timeout=1.0,
            )

    @pytest.mark.anyio
    async def test_prepare_engine_start_waits_for_thread_under_repeated_cancel(self):
        """Repeated cancellation of a residency cold load must not return before
        prepare_for_start() finishes, otherwise the thread can keep mutating
        model state past request/shutdown boundaries."""
        import threading

        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        entered = threading.Event()
        allow_finish = threading.Event()
        finished = threading.Event()

        class SlowPrepEngine:
            async def start(self):
                pass

            async def stop(self):
                pass

            def prepare_for_start(self):
                entered.set()
                allow_finish.wait()
                time.sleep(0.05)
                finished.set()

        async def factory(spec):
            return SlowPrepEngine()

        manager = ResidencyManager(factory, auto_unload_idle_seconds=0)
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        load_task = asyncio.create_task(manager.ensure_loaded("m"))
        try:
            # Wait for prepare_for_start to enter
            assert await asyncio.wait_for(
                asyncio.to_thread(entered.wait, 2.0),
                timeout=2.0,
            )

            # Double-cancel the load task
            load_task.cancel()
            await asyncio.sleep(0)
            load_task.cancel()

            # Let the blocking work complete
            allow_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await load_task

            # The critical assertion: prepare_for_start must have fully
            # completed before the CancelledError was raised.
            assert finished.is_set() is True, (
                "prepare_for_start() was still running after load task returned "
                "CancelledError — repeated cancellation escaped the drain loop"
            )
        finally:
            allow_finish.set()
            assert await asyncio.wait_for(
                asyncio.to_thread(finished.wait, 2.0),
                timeout=2.0,
            )
            # Clean up any leftover resident state
            with suppress(Exception):
                await manager.shutdown()

    @pytest.mark.anyio
    async def test_run_blocking_startup_work_does_not_livelock_on_cancelled_inner_task(
        self,
    ):
        """If the inner to_thread task ends up cancelled, the drain loop must
        exit instead of spinning forever on CancelledError."""
        from vllm_mlx.engine.base import run_blocking_startup_work

        def work_that_will_be_cancelled():
            raise asyncio.CancelledError()

        task = asyncio.create_task(
            run_blocking_startup_work(work_that_will_be_cancelled)
        )
        await asyncio.sleep(0)
        task.cancel()

        # Must complete promptly — a livelock would hang here
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.anyio
    async def test_blocking_cache_io_does_not_livelock_on_cancelled_inner_task(self):
        """If the inner to_thread task ends cancelled, the drain loop must exit."""
        import vllm_mlx.server as srv

        class FakeEngine:
            pass

        def io_that_will_be_cancelled(_engine):
            raise asyncio.CancelledError()

        task = asyncio.create_task(
            srv._run_blocking_engine_cache_io(io_that_will_be_cancelled, FakeEngine())
        )
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.anyio
    async def test_prepare_engine_start_does_not_livelock_on_cancelled_inner_task(self):
        """If prepare_for_start's to_thread task ends cancelled, the residency
        drain loop must exit instead of spinning."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        class CancellingPrepEngine:
            async def start(self):
                pass

            async def stop(self):
                pass

            def prepare_for_start(self):
                raise asyncio.CancelledError()

        async def factory(spec):
            return CancellingPrepEngine()

        manager = ResidencyManager(factory, auto_unload_idle_seconds=0)
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        load_task = asyncio.create_task(manager.ensure_loaded("m"))
        await asyncio.sleep(0)
        load_task.cancel()

        # Must complete promptly — a livelock would cause wait_for to raise
        # TimeoutError, which we want to surface as a hard failure.
        try:
            await asyncio.wait_for(load_task, timeout=2.0)
        except asyncio.CancelledError:
            pass  # expected: the load was cancelled
        except asyncio.TimeoutError:
            pytest.fail("Drain loop livelocked — load_task did not complete within 2s")

        # Clean up
        with suppress(Exception):
            await manager.shutdown()

    @pytest.mark.anyio
    @pytest.mark.parametrize("start_phase", ["llm", "mllm"])
    async def test_batched_start_phase_cancellation_preserves_cancelled_error_when_stop_fails(
        self, monkeypatch, caplog, start_phase
    ):
        """Batched eager startup should keep cancellation primary if teardown fails."""
        import vllm_mlx.engine.batched as engine_mod

        engine = engine_mod.BatchedEngine("fake-model")
        start_entered = asyncio.Event()
        stop_calls = {"count": 0}

        async def failing_stop():
            stop_calls["count"] += 1
            raise RuntimeError("stop boom")

        async def cancellable_start_phase():
            start_entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise

        monkeypatch.setattr(engine, "stop", failing_stop)

        if start_phase == "llm":
            monkeypatch.setattr(engine, "_is_mllm", False, raising=False)
            monkeypatch.setattr(engine, "_model", object(), raising=False)
            monkeypatch.setattr(engine, "_tokenizer", object(), raising=False)
            monkeypatch.setattr(
                engine, "_start_llm", cancellable_start_phase, raising=False
            )
        else:
            monkeypatch.setattr(engine, "_is_mllm", True, raising=False)
            monkeypatch.setattr(engine, "_model", object(), raising=False)
            monkeypatch.setattr(engine, "_processor", object(), raising=False)
            monkeypatch.setattr(engine, "_mllm_instance", object(), raising=False)
            monkeypatch.setattr(
                engine,
                "_start_mllm",
                cancellable_start_phase,
                raising=False,
            )

        start_task = asyncio.create_task(engine.start())
        try:
            await asyncio.wait_for(start_entered.wait(), timeout=1.0)

            caplog.set_level("ERROR")
            caplog.clear()
            start_task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await start_task

            assert stop_calls["count"] == 1
            assert any(
                record.levelname == "ERROR"
                and "startup cleanup failed while preserving cancellation"
                in record.getMessage()
                for record in caplog.records
            )
        finally:
            if not start_task.done():
                start_task.cancel()
                with suppress(asyncio.CancelledError):
                    await start_task

    @pytest.mark.anyio
    async def test_cleanup_failure_does_not_orphan_live_eager_engine(self, monkeypatch):
        """A failed eager-engine stop should keep the live engine guarded against replacement."""
        import vllm_mlx.server as srv

        class LiveEngine:
            _loaded = True

            async def stop(self):
                raise RuntimeError("stop boom")

        live_engine = LiveEngine()

        monkeypatch.setattr(srv, "_engine", live_engine, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        lifespan = srv.lifespan(srv.app)
        try:
            await lifespan.__anext__()

            with pytest.raises(RuntimeError, match="stop boom"):
                await lifespan.__anext__()

            assert srv._engine is live_engine
            with pytest.raises(RuntimeError, match="existing engine"):
                srv.load_model("new-model")
        finally:
            monkeypatch.setattr(srv, "_engine", None, raising=False)
            monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
            with suppress(Exception):
                await lifespan.aclose()

    @pytest.mark.anyio
    async def test_cleanup_failure_before_residency_shutdown_keeps_live_manager_guarded(
        self, monkeypatch
    ):
        """A pre-shutdown cleanup failure should not orphan a live residency manager."""
        import vllm_mlx.server as srv

        calls = {"shutdown": 0}

        class LiveEngine:
            preserve_native_tool_format = False

        live_engine = LiveEngine()

        async def ensure_loaded(model_key):
            return live_engine

        async def shutdown():
            calls["shutdown"] += 1
            raise AssertionError("shutdown should not be reached")

        class FakeMCPManager:
            async def stop(self):
                raise RuntimeError("mcp stop boom")

        live_manager = SimpleNamespace(
            ensure_loaded=ensure_loaded,
            get_engine=lambda model_key: live_engine,
            get_status=lambda model_key: {
                "state": "loaded",
                "active_requests": 0,
            },
            shutdown=shutdown,
        )

        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", live_manager, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", FakeMCPManager(), raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)
        monkeypatch.setattr(srv, "_lazy_load_model", False, raising=False)

        lifespan = srv.lifespan(srv.app)
        try:
            await lifespan.__anext__()

            with pytest.raises(RuntimeError, match="mcp stop boom"):
                await lifespan.__anext__()

            assert calls["shutdown"] == 0
            assert srv._residency_manager is live_manager
            with pytest.raises(RuntimeError, match="existing residency manager"):
                srv.load_model(
                    "mlx-community/Qwen3-0.6B-8bit",
                    auto_unload_idle_seconds=60,
                )
        finally:
            monkeypatch.setattr(srv, "_engine", None, raising=False)
            monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
            monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
            with suppress(Exception):
                await lifespan.aclose()

    @pytest.mark.anyio
    async def test_idle_unload_loop_survives_one_unload_failure(self, monkeypatch):
        """One unload failure should not kill the background lifecycle loop."""
        import vllm_mlx.server as srv

        retried = asyncio.Event()
        calls = {"count": 0}

        class FakeManager:
            async def unload_if_idle(self, model_key):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise RuntimeError("boom")
                retried.set()
                return False

        monkeypatch.setattr(srv, "_residency_manager", FakeManager(), raising=False)
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_auto_unload_idle_seconds", 1.0, raising=False)
        monkeypatch.setattr(srv, "_sync_engine_from_residency", lambda: None)

        task = asyncio.create_task(srv._lifecycle_loop())
        try:
            await asyncio.wait_for(retried.wait(), timeout=1.5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert calls["count"] >= 2

    @pytest.mark.anyio
    async def test_idle_unload_loop_honors_subsecond_timeout_granularity(
        self, monkeypatch
    ):
        """Sub-second unload timeouts should not be rounded up to one-second polling."""
        import vllm_mlx.server as srv

        sleeps = []

        class FakeManager:
            async def unload_if_idle(self, model_key):
                return False

        async def fake_sleep(delay):
            sleeps.append(delay)
            raise asyncio.CancelledError()

        monkeypatch.setattr(srv, "_residency_manager", FakeManager(), raising=False)
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_auto_unload_idle_seconds", 0.5, raising=False)
        monkeypatch.setattr(srv, "_sync_engine_from_residency", lambda: None)
        monkeypatch.setattr(srv.asyncio, "sleep", fake_sleep)

        with pytest.raises(asyncio.CancelledError):
            await srv._lifecycle_loop()

        assert sleeps == [0.25]

    @pytest.mark.anyio
    async def test_cache_restore_hook_does_not_block_event_loop(self, monkeypatch):
        """Cold-load cache restore should not freeze unrelated loop work."""
        import threading

        import vllm_mlx.server as srv

        callback_fired = threading.Event()
        callback_seen_during_hook = {"value": False}

        class FakeEngine:
            preserve_native_tool_format = False

            async def start(self):
                return None

            async def stop(self):
                return None

            def load_cache_from_disk(self, path):
                time.sleep(0.2)
                callback_seen_during_hook["value"] = callback_fired.is_set()
                return 1

        async def fake_engine_factory(spec):
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=True,
        )

        asyncio.get_running_loop().call_later(0.01, callback_fired.set)
        acquire_task = asyncio.create_task(srv._acquire_default_engine())

        await acquire_task
        assert callback_seen_during_hook["value"] is True
        await srv._release_default_engine()

    @pytest.mark.anyio
    async def test_cache_persist_hook_does_not_block_event_loop(self, monkeypatch):
        """Idle-unload cache persistence should not freeze unrelated loop work."""
        import threading

        import vllm_mlx.server as srv

        now = {"value": 1000.0}
        callback_fired = threading.Event()
        callback_seen_during_hook = {"value": False}

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                return None

            def load_cache_from_disk(self, path):
                return 0

            def save_cache_to_disk(self, path):
                time.sleep(0.2)
                callback_seen_during_hook["value"] = callback_fired.is_set()
                return True

        async def fake_engine_factory(spec):
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=True,
        )

        manager = srv._residency_manager
        assert manager is not None
        monkeypatch.setattr(manager, "_time_fn", lambda: now["value"])

        await srv._acquire_default_engine()
        await srv._release_default_engine()

        now["value"] += 120.0
        asyncio.get_running_loop().call_later(0.01, callback_fired.set)
        unload_task = asyncio.create_task(manager.unload_if_idle("default"))

        unloaded = await unload_task
        assert unloaded is True
        assert callback_seen_during_hook["value"] is True

    @pytest.mark.anyio
    async def test_idle_unload_persists_and_restores_prefix_cache(self, monkeypatch):
        """Server-managed idle unload should save and reload prefix cache state."""
        import vllm_mlx.server as srv

        load_calls = {"count": 0}
        save_calls = {"count": 0}
        now = {"value": 1000.0}

        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                return None

            def load_cache_from_disk(self, path):
                return 1

            def save_cache_to_disk(self, path):
                return True

        async def fake_engine_factory(spec):
            return FakeEngine()

        def fake_load_prefix_cache_from_disk(engine=None):
            load_calls["count"] += 1

        def fake_save_prefix_cache_to_disk(engine=None):
            save_calls["count"] += 1

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(
            srv, "_load_prefix_cache_from_disk", fake_load_prefix_cache_from_disk
        )
        monkeypatch.setattr(
            srv, "_save_prefix_cache_to_disk", fake_save_prefix_cache_to_disk
        )

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60,
        )

        manager = srv._residency_manager
        assert manager is not None
        monkeypatch.setattr(manager, "_time_fn", lambda: now["value"])

        await manager.acquire("default")
        await manager.release("default")
        now["value"] += 120.0
        await manager.unload_if_idle("default")
        await manager.acquire("default")

        assert load_calls["count"] == 2
        assert save_calls["count"] == 1

        await manager.release("default")

    @pytest.mark.anyio
    async def test_lifespan_does_not_double_apply_cache_hooks_in_lifecycle_mode(
        self, monkeypatch
    ):
        """Lifecycle startup/shutdown should not duplicate cache load/save hooks."""
        import vllm_mlx.server as srv

        load_calls = {"count": 0}
        save_calls = {"count": 0}

        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                return None

            def load_cache_from_disk(self, path):
                load_calls["count"] += 1
                return 1

            def save_cache_to_disk(self, path):
                save_calls["count"] += 1
                return True

        async def fake_engine_factory(spec):
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_mcp_manager", None)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60,
        )

        lifespan = srv.lifespan(srv.app)
        await lifespan.__anext__()
        assert load_calls["count"] == 1

        with pytest.raises(StopAsyncIteration):
            await lifespan.__anext__()

        assert save_calls["count"] == 1

    def test_residency_reload_invalidates_cached_tool_parser(self, monkeypatch):
        """Tool parser instances should be rebuilt after an unload/reload engine swap."""
        import vllm_mlx.server as srv

        parser_tokenizers = []

        class FakeParser:
            def __init__(self, tokenizer=None):
                parser_tokenizers.append(tokenizer)

            @classmethod
            def supports_native_format(cls):
                return False

            def reset(self):
                return None

            def extract_tool_calls(self, output_text, request_dict=None):
                return SimpleNamespace(
                    tools_called=False,
                    tool_calls=[],
                    content=output_text,
                )

        class FakeEngine:
            def __init__(self, tokenizer):
                self._tokenizer_value = tokenizer
                self.preserve_native_tool_format = False

            @property
            def tokenizer(self):
                return self._tokenizer_value

        engine_state = {"engine": FakeEngine("tok-1")}
        fake_manager = SimpleNamespace(
            get_engine=lambda model_key: engine_state["engine"]
        )

        monkeypatch.setattr(srv, "_enable_auto_tool_choice", True, raising=False)
        monkeypatch.setattr(srv, "_tool_call_parser", "fake", raising=False)
        monkeypatch.setattr(srv, "_tool_parser_instance", None, raising=False)
        monkeypatch.setattr(srv, "_engine", engine_state["engine"], raising=False)
        monkeypatch.setattr(srv, "_residency_manager", fake_manager, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(
            srv.ToolParserManager,
            "get_tool_parser",
            lambda name: FakeParser,
        )

        srv._parse_tool_calls_with_parser("<tool_call>")
        assert parser_tokenizers == ["tok-1"]

        engine_state["engine"] = None
        srv._sync_engine_from_residency()
        engine_state["engine"] = FakeEngine("tok-2")
        srv._sync_engine_from_residency()

        srv._parse_tool_calls_with_parser("<tool_call>")
        assert parser_tokenizers == ["tok-1", "tok-2"]

    @pytest.mark.anyio
    async def test_acquire_path_rebuilds_tool_parser_after_resident_swap(
        self, monkeypatch
    ):
        """The first request path should rebuild parser state when acquire returns a new engine."""
        import vllm_mlx.server as srv

        parser_tokenizers = []

        class FakeParser:
            def __init__(self, tokenizer=None):
                parser_tokenizers.append(tokenizer)

            @classmethod
            def supports_native_format(cls):
                return False

            def reset(self):
                return None

            def extract_tool_calls(self, output_text, request_dict=None):
                return SimpleNamespace(
                    tools_called=False,
                    tool_calls=[],
                    content=output_text,
                )

        class FakeEngine:
            def __init__(self, tokenizer):
                self._tokenizer_value = tokenizer
                self.preserve_native_tool_format = False

            @property
            def tokenizer(self):
                return self._tokenizer_value

        current = {"engine": FakeEngine("tok-1")}

        async def fake_acquire(model_key):
            return current["engine"]

        async def fake_release(model_key):
            return None

        fake_manager = SimpleNamespace(
            acquire=fake_acquire,
            release=fake_release,
            get_engine=lambda model_key: current["engine"],
        )

        monkeypatch.setattr(srv, "_enable_auto_tool_choice", True, raising=False)
        monkeypatch.setattr(srv, "_tool_call_parser", "fake", raising=False)
        monkeypatch.setattr(srv, "_tool_parser_instance", None, raising=False)
        monkeypatch.setattr(srv, "_engine", current["engine"], raising=False)
        monkeypatch.setattr(srv, "_residency_manager", fake_manager, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(
            srv.ToolParserManager,
            "get_tool_parser",
            lambda name: FakeParser,
        )

        srv._parse_tool_calls_with_parser("<tool_call>")
        assert parser_tokenizers == ["tok-1"]

        current["engine"] = FakeEngine("tok-2")
        await srv._acquire_default_engine()
        srv._parse_tool_calls_with_parser("<tool_call>")

        assert parser_tokenizers == ["tok-1", "tok-2"]

        await srv._release_default_engine()

    @pytest.mark.anyio
    async def test_shutdown_clears_stopped_eager_engine_for_inprocess_reload(
        self, monkeypatch
    ):
        """A stopped eager engine should not block the next in-process load_model()."""
        import vllm_mlx.engine.simple as simple_mod
        import vllm_mlx.server as srv

        class OldEngine:
            _loaded = True

            def __init__(self):
                self.stopped = False

            async def stop(self):
                self.stopped = True

        old_engine = OldEngine()

        class NewEngine:
            def __init__(self):
                self.is_mllm = False
                self.preserve_native_tool_format = False

            async def start(self):
                return None

        new_engine = NewEngine()

        monkeypatch.setattr(srv, "_engine", old_engine, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(
            simple_mod,
            "SimpleEngine",
            lambda model_name, **kwargs: new_engine,
        )

        lifespan = srv.lifespan(srv.app)
        await lifespan.__anext__()

        with pytest.raises(StopAsyncIteration):
            await lifespan.__anext__()

        assert old_engine.stopped is True

        await asyncio.to_thread(
            srv.load_model,
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=0.0,
        )

        assert srv._engine is new_engine

    def test_load_model_closes_temporary_loop_after_eager_simple_start(
        self, monkeypatch
    ):
        """In-process eager simple startup should close its temporary event loop."""
        import vllm_mlx.engine.simple as simple_mod
        import vllm_mlx.server as srv

        created_loop = {"loop": None}
        observed_loop = {"loop": None}
        real_new_event_loop = asyncio.new_event_loop
        previous_loop = real_new_event_loop()

        class FakeEngine:
            is_mllm = False
            preserve_native_tool_format = False

            async def start(self):
                await asyncio.to_thread(lambda: None)

            async def stop(self):
                return None

        def tracked_new_event_loop():
            loop = real_new_event_loop()
            created_loop["loop"] = loop
            return loop

        monkeypatch.setattr(asyncio, "new_event_loop", tracked_new_event_loop)
        monkeypatch.setattr(
            simple_mod,
            "SimpleEngine",
            lambda model_name, **kwargs: FakeEngine(),
        )
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        try:
            asyncio.set_event_loop(previous_loop)
            srv.load_model(
                "mlx-community/Qwen3-0.6B-8bit",
                auto_unload_idle_seconds=0.0,
            )

            assert created_loop["loop"] is not None
            assert created_loop["loop"].is_closed() is True
            try:
                observed_loop["loop"] = asyncio.get_event_loop()
            except RuntimeError:
                observed_loop["loop"] = None
            if observed_loop["loop"] is not None:
                assert observed_loop["loop"] is not created_loop["loop"]
                assert observed_loop["loop"].is_closed() is False
        finally:
            asyncio.set_event_loop(None)
            if (
                observed_loop["loop"] is not None
                and observed_loop["loop"] is not previous_loop
                and observed_loop["loop"] is not created_loop["loop"]
                and not observed_loop["loop"].is_closed()
            ):
                observed_loop["loop"].close()
            if not previous_loop.is_closed():
                previous_loop.close()
            if (
                created_loop["loop"] is not None
                and not created_loop["loop"].is_closed()
            ):
                created_loop["loop"].close()
            monkeypatch.setattr(srv, "_engine", None, raising=False)

    @pytest.mark.anyio
    async def test_load_model_rejects_reconfiguration_after_lifespan_start(
        self, monkeypatch
    ):
        """Post-start reconfiguration should require a server restart and be a no-op."""
        import vllm_mlx.server as srv

        async def fake_engine_factory(spec):
            raise AssertionError("lazy startup should not load during lifespan entry")

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=0.0,
            lazy_load_model=True,
        )

        lifespan = srv.lifespan(srv.app)
        await lifespan.__anext__()
        original_engine = srv._engine
        original_manager = srv._residency_manager
        original_model_name = srv._model_name
        original_default_model_key = srv._default_model_key
        original_auto_unload = srv._auto_unload_idle_seconds
        original_lazy_load = srv._lazy_load_model
        original_parser = srv._tool_parser_instance

        try:
            with pytest.raises(RuntimeError, match="restart the server"):
                srv.load_model(
                    "mlx-community/Llama-3.2-3B-Instruct-4bit",
                    auto_unload_idle_seconds=0.0,
                    lazy_load_model=True,
                )

            assert srv._engine is original_engine
            assert srv._residency_manager is original_manager
            assert srv._model_name == original_model_name
            assert srv._default_model_key == original_default_model_key
            assert srv._auto_unload_idle_seconds == original_auto_unload
            assert srv._lazy_load_model is original_lazy_load
            assert srv._tool_parser_instance is original_parser
        finally:
            with pytest.raises(StopAsyncIteration):
                await lifespan.__anext__()

    @pytest.mark.anyio
    async def test_lazy_cold_acquire_does_not_block_event_loop(self, monkeypatch):
        """Cold resident startup should not freeze unrelated event-loop work."""
        import vllm_mlx.server as srv

        class FakeEngine:
            preserve_native_tool_format = False

            def prepare_for_start(self):
                time.sleep(0.2)

            async def start(self):
                return None

            async def stop(self):
                return None

        async def fake_engine_factory(spec):
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=True,
        )

        acquire_task = asyncio.create_task(srv._acquire_default_engine())

        async def heartbeat():
            await asyncio.sleep(0)
            return "heartbeat"

        await asyncio.sleep(0)
        heartbeat_task = asyncio.create_task(heartbeat())
        assert await heartbeat_task == "heartbeat"
        assert not acquire_task.done()

        await acquire_task

        await srv._release_default_engine()

    @pytest.mark.anyio
    async def test_completion_timeout_covers_cold_resident_acquire(self, monkeypatch):
        """Request timeout should include lazy-load engine acquisition."""
        from fastapi import HTTPException

        import vllm_mlx.server as srv

        generate_calls = {"count": 0}
        load_gate = asyncio.Event()

        class FakeEngine:
            preserve_native_tool_format = False

            async def start(self):
                await load_gate.wait()

            async def stop(self):
                return None

            async def generate(self, **kwargs):
                generate_calls["count"] += 1
                return SimpleNamespace(
                    text="done",
                    finish_reason="stop",
                    completion_tokens=1,
                    prompt_tokens=1,
                )

        class FakeRawRequest:
            async def json(self):
                return {
                    "model": "mlx-community/Qwen3-0.6B-8bit",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "max_tokens": 16,
                }

            async def is_disconnected(self):
                return False

        async def fake_engine_factory(spec):
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=True,
        )

        request = SimpleNamespace(
            model="mlx-community/Qwen3-0.6B-8bit",
            prompt="hi",
            stream=False,
            max_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
            specprefill=None,
            specprefill_keep_pct=None,
            stop=None,
            timeout=0.01,
        )

        request_task = asyncio.create_task(
            srv.create_completion(request, FakeRawRequest())
        )
        try:
            done, _ = await asyncio.wait({request_task}, timeout=0.2)
            assert request_task in done

            with pytest.raises(HTTPException, match="Request timed out"):
                await request_task

            assert generate_calls["count"] == 0

            load_gate.set()
            engine = await asyncio.wait_for(srv._acquire_default_engine(), timeout=1.0)
            await srv._release_default_engine()

            status = srv._get_lifecycle_status()
            assert engine is not None
            assert status is not None
            assert status["state"] == "loaded"
            assert status["active_requests"] == 0
        finally:
            load_gate.set()
            if not request_task.done():
                with suppress(Exception):
                    await request_task

    @pytest.mark.anyio
    async def test_chat_timeout_covers_cold_resident_acquire(self, monkeypatch):
        """Chat timeout should include lazy-load engine acquisition."""
        from fastapi import HTTPException

        import vllm_mlx.server as srv

        chat_calls = {"count": 0}
        load_gate = asyncio.Event()

        class FakeEngine:
            is_mllm = False
            preserve_native_tool_format = False

            async def start(self):
                await load_gate.wait()

            async def stop(self):
                return None

            async def chat(self, **kwargs):
                chat_calls["count"] += 1
                return SimpleNamespace(
                    text="done",
                    finish_reason="stop",
                    completion_tokens=1,
                    prompt_tokens=1,
                )

        class FakeRawRequest:
            async def json(self):
                return {
                    "model": "mlx-community/Qwen3-0.6B-8bit",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "max_tokens": 16,
                }

            async def is_disconnected(self):
                return False

        def fake_extract(messages, preserve_native_format):
            return ([{"role": "user", "content": "hi"}], [], [], [])

        async def fake_engine_factory(spec):
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "extract_multimodal_content", fake_extract)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=True,
        )

        request = SimpleNamespace(
            model="mlx-community/Qwen3-0.6B-8bit",
            messages=[SimpleNamespace(role="user", content="hi")],
            stream=False,
            max_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
            response_format=None,
            tools=None,
            tool_choice=None,
            enable_thinking=None,
            video_fps=None,
            video_max_frames=None,
            specprefill=None,
            specprefill_keep_pct=None,
            chat_template_kwargs=None,
            timeout=0.01,
        )

        request_task = asyncio.create_task(
            srv.create_chat_completion(request, FakeRawRequest())
        )
        try:
            done, _ = await asyncio.wait({request_task}, timeout=0.2)
            assert request_task in done

            with pytest.raises(HTTPException, match="Request timed out"):
                await request_task

            assert chat_calls["count"] == 0

            load_gate.set()
            engine = await asyncio.wait_for(srv._acquire_default_engine(), timeout=1.0)
            await srv._release_default_engine()

            status = srv._get_lifecycle_status()
            assert engine is not None
            assert status is not None
            assert status["state"] == "loaded"
            assert status["active_requests"] == 0
        finally:
            load_gate.set()
            if not request_task.done():
                with suppress(Exception):
                    await request_task

    @pytest.mark.anyio
    async def test_completion_disconnect_covers_cold_resident_acquire(
        self, monkeypatch
    ):
        """Disconnect handling should abort a cold resident acquire before generation."""
        from fastapi.responses import Response

        import vllm_mlx.server as srv

        generate_calls = {"count": 0}
        load_gate = asyncio.Event()
        disconnect_polled = asyncio.Event()

        class FakeEngine:
            preserve_native_tool_format = False

            async def start(self):
                await load_gate.wait()

            async def stop(self):
                return None

            async def generate(self, **kwargs):
                generate_calls["count"] += 1
                return SimpleNamespace(
                    text="done",
                    finish_reason="stop",
                    completion_tokens=1,
                    prompt_tokens=1,
                )

        class FakeRequest:
            async def is_disconnected(self):
                disconnect_polled.set()
                return True

        async def fake_engine_factory(spec):
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=True,
        )

        request = SimpleNamespace(
            model="mlx-community/Qwen3-0.6B-8bit",
            prompt="hi",
            stream=False,
            max_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
            specprefill=None,
            specprefill_keep_pct=None,
            stop=None,
            timeout=60.0,
        )

        request_task = asyncio.create_task(
            srv.create_completion(request, FakeRequest())
        )
        try:
            # Leave generous slack over the production 0.5s poll interval so
            # this stays a behavior test rather than a scheduler-jitter race.
            await asyncio.wait_for(disconnect_polled.wait(), timeout=2.0)

            done, _ = await asyncio.wait({request_task}, timeout=1.0)
            assert request_task in done

            response = await request_task

            assert isinstance(response, Response)
            assert response.status_code == 499
            assert generate_calls["count"] == 0

            load_gate.set()
            engine = await asyncio.wait_for(srv._acquire_default_engine(), timeout=1.0)
            await srv._release_default_engine()

            status = srv._get_lifecycle_status()
            assert engine is not None
            assert status is not None
            assert status["state"] == "loaded"
            assert status["active_requests"] == 0
        finally:
            load_gate.set()
            if not request_task.done():
                with suppress(Exception):
                    await request_task

    @pytest.mark.anyio
    async def test_count_tokens_disconnect_covers_cold_resident_acquire(
        self, monkeypatch
    ):
        """Token counting should unwind a solo cold load after a client disconnect."""
        from fastapi.responses import Response

        import vllm_mlx.server as srv

        created = 0
        encode_calls = {"count": 0}
        disconnect_polled = asyncio.Event()
        first_load_gate = asyncio.Event()
        first_start_cancelled = asyncio.Event()
        stopped_generations: list[int] = []

        class FakeTokenizer:
            def encode(self, text):
                encode_calls["count"] += 1
                return list(range(len(text)))

        class FakeEngine:
            def __init__(self, generation):
                self.generation = generation

            preserve_native_tool_format = False

            async def start(self):
                if self.generation != 1:
                    return None
                try:
                    await first_load_gate.wait()
                except asyncio.CancelledError:
                    first_start_cancelled.set()
                    raise

            async def stop(self):
                stopped_generations.append(self.generation)
                return None

            @property
            def tokenizer(self):
                return FakeTokenizer()

        class FakeRequest:
            async def json(self):
                return {
                    "system": "sys",
                    "messages": [{"role": "user", "content": "hi"}],
                }

            async def is_disconnected(self):
                disconnect_polled.set()
                return True

        async def fake_engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine(created)

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=True,
        )

        request_task = asyncio.create_task(srv.count_anthropic_tokens(FakeRequest()))
        try:
            await asyncio.wait_for(disconnect_polled.wait(), timeout=2.0)

            done, _ = await asyncio.wait({request_task}, timeout=1.0)
            assert request_task in done

            response = await request_task

            assert isinstance(response, Response)
            assert response.status_code == 499
            assert encode_calls["count"] == 0

            await asyncio.wait_for(first_start_cancelled.wait(), timeout=1.0)

            status = srv._get_lifecycle_status()
            assert status is not None
            assert status["state"] == "unloaded"
            assert status["active_requests"] == 0
            assert stopped_generations == [1]

            engine = await asyncio.wait_for(srv._acquire_default_engine(), timeout=1.0)
            await srv._release_default_engine()

            assert engine is not None
            assert engine.generation == 2
            assert created == 2
            status = srv._get_lifecycle_status()
            assert status["state"] == "loaded"
            assert status["active_requests"] == 0
        finally:
            first_load_gate.set()
            if not request_task.done():
                with suppress(Exception):
                    await request_task

    @pytest.mark.anyio
    async def test_count_tokens_does_not_refresh_idle_unload_activity(
        self, monkeypatch
    ):
        """Budgeting-only traffic should not keep the resident hot indefinitely."""
        import vllm_mlx.server as srv

        now = {"value": 1000.0}

        class FakeTokenizer:
            def encode(self, text):
                return list(range(len(text)))

        class FakeEngine:
            preserve_native_tool_format = False

            async def start(self):
                return None

            async def stop(self):
                return None

            @property
            def tokenizer(self):
                return FakeTokenizer()

        class FakeRequest:
            async def json(self):
                return {
                    "system": "sys",
                    "messages": [{"role": "user", "content": "hi"}],
                }

            async def is_disconnected(self):
                return False

        async def fake_engine_factory(spec):
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=True,
        )

        manager = srv._residency_manager
        assert manager is not None
        monkeypatch.setattr(manager, "_time_fn", lambda: now["value"])

        await srv._acquire_default_engine()
        await srv._release_default_engine()

        now["value"] = 1059.0
        response = await srv.count_anthropic_tokens(FakeRequest())
        assert response == {"input_tokens": 5}

        now["value"] = 1061.0
        unloaded = await manager.unload_if_idle("default")
        status = srv._get_lifecycle_status()

        assert unloaded is True
        assert status is not None
        assert status["state"] == "unloaded"

    @pytest.mark.anyio
    async def test_count_tokens_validates_model_before_resident_acquire(
        self, monkeypatch
    ):
        """count_tokens should reject wrong model names before cold resident acquire."""
        from fastapi import HTTPException

        import vllm_mlx.server as srv

        calls = {"acquire": 0}

        class FakeRequest:
            async def json(self):
                return {
                    "model": "wrong-model",
                    "system": "sys",
                    "messages": [{"role": "user", "content": "hi"}],
                }

            async def is_disconnected(self):
                return False

        async def fake_acquire_default_engine_for_request(*args, **kwargs):
            calls["acquire"] += 1
            raise AssertionError("acquire should not run for wrong-model count_tokens")

        monkeypatch.setattr(
            srv,
            "_acquire_default_engine_for_request",
            fake_acquire_default_engine_for_request,
        )
        monkeypatch.setattr(
            srv,
            "_model_name",
            "mlx-community/Qwen3-0.6B-8bit",
            raising=False,
        )

        with pytest.raises(HTTPException, match="does not exist"):
            await srv.count_anthropic_tokens(FakeRequest())

        assert calls["acquire"] == 0

    @pytest.mark.anyio
    async def test_anthropic_messages_refresh_idle_unload_activity(self, monkeypatch):
        """Successful Anthropic messages requests should count as residency activity."""
        import vllm_mlx.server as srv

        now = {"value": 1000.0}

        class FakeOutput:
            text = "hello"
            completion_tokens = 5
            prompt_tokens = 3
            finish_reason = "stop"

        class FakeEngine:
            preserve_native_tool_format = False

            async def start(self):
                return None

            async def stop(self):
                return None

            async def chat(self, messages, **kwargs):
                return FakeOutput()

        class FakeRawRequest:
            async def json(self):
                return {
                    "model": "mlx-community/Qwen3-0.6B-8bit",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "max_tokens": 16,
                }

            async def is_disconnected(self):
                return False

        async def fake_engine_factory(spec):
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)
        monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)
        monkeypatch.setattr(
            srv, "_parse_tool_calls_with_parser", lambda text, request: (text, None)
        )

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=True,
        )

        manager = srv._residency_manager
        assert manager is not None
        monkeypatch.setattr(manager, "_time_fn", lambda: now["value"])

        await srv._acquire_default_engine()
        await srv._release_default_engine()
        status = manager.get_status("default")
        assert status["last_used_at"] == 1000.0

        now["value"] = 1059.0
        response = await srv.create_anthropic_message(FakeRawRequest())
        assert response.status_code == 200

        status = manager.get_status("default")
        assert status["last_used_at"] == 1059.0

        now["value"] = 1061.0
        unloaded = await manager.unload_if_idle("default")
        status = srv._get_lifecycle_status()

        assert unloaded is False
        assert status is not None
        assert status["state"] == "loaded"

    @pytest.mark.anyio
    async def test_wait_with_disconnect_treats_raced_task_cancellation_as_disconnect(
        self,
    ):
        """A raced cancelled task should not leak CancelledError past disconnect handling."""
        import vllm_mlx.server as srv

        task_ref = {"task": None}

        class FakeRequest:
            async def is_disconnected(self):
                task = task_ref["task"]
                assert task is not None
                task.cancel()
                await asyncio.sleep(0)
                return True

        async def cancellable_work():
            await asyncio.sleep(3600)

        task = asyncio.create_task(cancellable_work())
        task_ref["task"] = task

        result = await srv._wait_with_disconnect(
            task,
            FakeRequest(),
            timeout=1.0,
            poll_interval=0.001,
        )

        assert result is None
        assert task.cancelled() is True

    @pytest.mark.anyio
    async def test_lazy_load_model_starts_unloaded_and_reports_unloaded_status(
        self, monkeypatch
    ):
        """Lazy lifecycle mode should register an unloaded resident before first request."""
        import vllm_mlx.server as srv

        create_calls = {"count": 0}

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                return None

        async def fake_engine_factory(spec):
            create_calls["count"] += 1
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=0.0,
            lazy_load_model=True,
        )

        lifespan = srv.lifespan(srv.app)
        await lifespan.__anext__()

        assert create_calls["count"] == 0
        assert srv._engine is None

        health_payload = await srv.health()
        status_payload = await srv.status()

        assert health_payload["model_loaded"] is False
        assert health_payload["residency_state"] == "unloaded"
        assert status_payload["status"] == "not_loaded"
        assert status_payload["residency"]["state"] == "unloaded"

        with pytest.raises(StopAsyncIteration):
            await lifespan.__anext__()

    @pytest.mark.anyio
    async def test_lazy_load_first_acquire_triggers_initial_engine_load(
        self, monkeypatch
    ):
        """The first real acquire after lazy startup should create and start the engine."""
        import vllm_mlx.server as srv

        create_calls = {"count": 0}
        start_calls = {"count": 0}

        class FakeEngine:
            preserve_native_tool_format = False
            is_mllm = False

            async def start(self):
                start_calls["count"] += 1

            async def stop(self):
                return None

            def get_stats(self):
                return {}

        async def fake_engine_factory(spec):
            create_calls["count"] += 1
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=0.0,
            lazy_load_model=True,
        )

        lifespan = srv.lifespan(srv.app)
        await lifespan.__anext__()

        assert create_calls["count"] == 0
        assert start_calls["count"] == 0

        engine = await srv._acquire_default_engine()
        status_payload = await srv.status()

        assert create_calls["count"] == 1
        assert start_calls["count"] == 1
        assert engine is srv._engine
        assert status_payload["residency"]["state"] == "loaded"
        assert status_payload["status"] == "stopped"

        await srv._release_default_engine()

        with pytest.raises(StopAsyncIteration):
            await lifespan.__anext__()

    @pytest.mark.anyio
    async def test_lazy_load_with_idle_unload_starts_unloaded_and_reports_unloaded_status(
        self, monkeypatch
    ):
        """Lazy startup should stay cold even when idle auto-unload is also enabled."""
        import vllm_mlx.server as srv

        create_calls = {"count": 0}

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                return None

        async def fake_engine_factory(spec):
            create_calls["count"] += 1
            return FakeEngine()

        monkeypatch.setattr(srv, "_engine_factory", fake_engine_factory)
        monkeypatch.setattr(srv, "_engine", None, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None, raising=False)
        monkeypatch.setattr(srv, "_lifecycle_task", None, raising=False)

        srv.load_model(
            "mlx-community/Qwen3-0.6B-8bit",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=True,
        )

        lifespan = srv.lifespan(srv.app)
        await lifespan.__anext__()

        assert create_calls["count"] == 0
        assert srv._engine is None

        health_payload = await srv.health()
        status_payload = await srv.status()

        assert health_payload["model_loaded"] is False
        assert health_payload["residency_state"] == "unloaded"
        assert health_payload["auto_unload_idle_seconds"] == 60.0
        assert status_payload["status"] == "not_loaded"
        assert status_payload["residency"]["state"] == "unloaded"
        assert status_payload["residency"]["auto_unload_idle_seconds"] == 60.0

        with pytest.raises(StopAsyncIteration):
            await lifespan.__anext__()

    def test_load_model_rejects_replacing_live_residency_manager(self, monkeypatch):
        """load_model() should not overwrite a live residency manager in-process."""
        import vllm_mlx.server as srv

        live_engine = object()
        live_manager = SimpleNamespace(
            get_engine=lambda model_key: live_engine,
            get_status=lambda model_key: {
                "state": "loaded",
                "active_requests": 0,
            },
        )

        monkeypatch.setattr(srv, "_residency_manager", live_manager, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)

        with pytest.raises(RuntimeError, match="existing residency manager"):
            srv.load_model(
                "mlx-community/Qwen3-0.6B-8bit",
                auto_unload_idle_seconds=60,
            )

        assert srv._residency_manager is live_manager

    def test_load_model_rejection_leaves_server_globals_unchanged(self, monkeypatch):
        """Rejected live-manager replacement should behave like a no-op."""
        import vllm_mlx.server as srv

        live_engine = object()
        live_manager = SimpleNamespace(
            get_engine=lambda model_key: live_engine,
            get_status=lambda model_key: {
                "state": "loaded",
                "active_requests": 0,
            },
        )

        monkeypatch.setattr(srv, "_residency_manager", live_manager, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_model_name", "old-model", raising=False)
        monkeypatch.setattr(srv, "_default_max_tokens", 128, raising=False)
        monkeypatch.setattr(srv, "_force_mllm_model", False, raising=False)
        monkeypatch.setattr(srv, "_auto_unload_idle_seconds", 60.0, raising=False)
        monkeypatch.setattr(srv, "_tool_parser_instance", object(), raising=False)

        original_parser = srv._tool_parser_instance

        with pytest.raises(RuntimeError, match="existing residency manager"):
            srv.load_model(
                "new-model",
                max_tokens=999,
                force_mllm=True,
                auto_unload_idle_seconds=120,
            )

        assert srv._residency_manager is live_manager
        assert srv._model_name == "old-model"
        assert srv._default_max_tokens == 128
        assert srv._force_mllm_model is False
        assert srv._auto_unload_idle_seconds == 60.0
        assert srv._tool_parser_instance is original_parser

    def test_load_model_rejects_live_legacy_engine_when_enabling_lifecycle(
        self, monkeypatch
    ):
        """Enabling lifecycle should reject while a live eager engine is present."""
        import vllm_mlx.server as srv

        class LiveEngine:
            def __init__(self):
                self.stopped = False

            async def stop(self):
                self.stopped = True

        live_engine = LiveEngine()

        monkeypatch.setattr(srv, "_engine", live_engine, raising=False)
        monkeypatch.setattr(srv, "_residency_manager", None, raising=False)
        monkeypatch.setattr(srv, "_default_model_key", None, raising=False)
        monkeypatch.setattr(srv, "_model_name", "old-model", raising=False)
        monkeypatch.setattr(srv, "_default_max_tokens", 128, raising=False)
        monkeypatch.setattr(srv, "_force_mllm_model", False, raising=False)
        monkeypatch.setattr(srv, "_auto_unload_idle_seconds", 0.0, raising=False)
        monkeypatch.setattr(srv, "_tool_parser_instance", object(), raising=False)

        original_parser = srv._tool_parser_instance

        with pytest.raises(RuntimeError, match="existing engine"):
            srv.load_model(
                "new-model",
                max_tokens=999,
                force_mllm=True,
                auto_unload_idle_seconds=120,
            )
        assert live_engine.stopped is False
        assert srv._engine is live_engine
        assert srv._residency_manager is None
        assert srv._default_model_key is None
        assert srv._model_name == "old-model"
        assert srv._default_max_tokens == 128
        assert srv._force_mllm_model is False
        assert srv._auto_unload_idle_seconds == 0.0
        assert srv._tool_parser_instance is original_parser


class TestLifecycleLoopIdleEvent:
    """Verify that the lifecycle loop uses an asyncio.Event for gating."""

    @pytest.mark.anyio
    async def test_lifecycle_loop_blocks_when_event_cleared(self):
        """The loop should block on the Event, not busy-poll."""
        import vllm_mlx.server as srv

        iterations = 0

        class FakeManager:
            def __init__(self):
                self.auto_unload_idle_seconds = 10

            def get_engine(self, key):
                return None

            def get_status(self, key):
                return {"state": "unloaded", "active_requests": 0}

            async def unload_if_idle(self, key):
                nonlocal iterations
                iterations += 1
                return False

        monkeypatch_attrs = {
            "_residency_manager": FakeManager(),
            "_default_model_key": "default",
            "_auto_unload_idle_seconds": 10.0,
        }
        originals = {}
        for k, v in monkeypatch_attrs.items():
            originals[k] = getattr(srv, k, None)
            setattr(srv, k, v)

        # Clear the event so the loop should block
        idle_event = srv._get_idle_unload_event()
        idle_event.clear()

        task = asyncio.create_task(srv._lifecycle_loop())
        try:
            # Give it time — if it were polling at 0.1s it would iterate many times
            await asyncio.sleep(0.3)
            assert (
                iterations == 0
            ), f"Loop iterated {iterations} times while event was cleared"

            # Now set the event and let it run one iteration
            idle_event.set()
            await asyncio.sleep(0.1)
            assert iterations >= 1
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            for k, v in originals.items():
                setattr(srv, k, v)
            idle_event.set()


class TestPublicLifecycleStatusSanitization:
    """Verify _public_lifecycle_status sanitization behavior."""

    def test_none_error_stays_none(self):
        import vllm_mlx.server as srv

        result = srv._public_lifecycle_status(
            {
                "state": "loaded",
                "last_error": None,
            }
        )
        assert result["last_error"] is None

    def test_raw_error_replaced_with_category(self):
        import vllm_mlx.server as srv

        result = srv._public_lifecycle_status(
            {
                "state": "failed",
                "last_error": "OSError: /tmp/model not found",
            }
        )
        assert result["last_error"] == "model_load_failed"

    def test_returns_none_for_none_input(self):
        import vllm_mlx.server as srv

        assert srv._public_lifecycle_status(None) is None

    @pytest.mark.anyio
    async def test_health_surfaces_sanitized_error_on_failed_resident(
        self, monkeypatch
    ):
        """Failed resident should show last_error='model_load_failed' on /health."""
        import vllm_mlx.server as srv

        fake_manager = SimpleNamespace(
            get_status=lambda model_key: {
                "model_key": model_key,
                "state": "failed",
                "active_requests": 0,
                "last_used_at": None,
                "loaded_at": None,
                "last_error": "RuntimeError: out of memory",
                "auto_unload_idle_seconds": 60,
            }
        )
        monkeypatch.setattr(srv, "_engine", None)
        monkeypatch.setattr(srv, "_model_name", "test-model", raising=False)
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_residency_manager", fake_manager, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None)

        payload = await srv.health()

        assert payload["status"] == "unhealthy"
        assert payload["residency_state"] == "failed"
        assert payload["last_error"] == "model_load_failed"

    @pytest.mark.anyio
    async def test_health_omits_last_error_for_healthy_resident(self, monkeypatch):
        """Healthy/loaded resident should not include last_error in health."""
        import vllm_mlx.server as srv

        fake_manager = SimpleNamespace(
            get_status=lambda model_key: {
                "model_key": model_key,
                "state": "loaded",
                "active_requests": 1,
                "last_used_at": 1710200000.0,
                "loaded_at": 1710199000.0,
                "last_error": None,
                "auto_unload_idle_seconds": 0,
            }
        )

        class FakeEngine:
            is_mllm = False

            def get_stats(self):
                return {"engine_type": "simple"}

        monkeypatch.setattr(srv, "_engine", FakeEngine())
        monkeypatch.setattr(srv, "_model_name", "test-model", raising=False)
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_residency_manager", fake_manager, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None)

        payload = await srv.health()

        assert payload["status"] == "healthy"
        assert "last_error" not in payload

    @pytest.mark.anyio
    async def test_health_and_status_agree_on_empty_string_error(self, monkeypatch):
        """An empty-string last_error should be treated the same by both
        /health and /v1/status — both use ``is not None`` for consistency."""
        import vllm_mlx.server as srv

        fake_manager = SimpleNamespace(
            get_status=lambda model_key: {
                "model_key": model_key,
                "model_name": "test-model",
                "state": "failed",
                "active_requests": 0,
                "last_used_at": None,
                "loaded_at": None,
                "last_error": "",
                "auto_unload_idle_seconds": 60,
            }
        )
        monkeypatch.setattr(srv, "_engine", None)
        monkeypatch.setattr(srv, "_model_name", "test-model", raising=False)
        monkeypatch.setattr(srv, "_default_model_key", "default", raising=False)
        monkeypatch.setattr(srv, "_residency_manager", fake_manager, raising=False)
        monkeypatch.setattr(srv, "_mcp_manager", None)

        health_payload = await srv.health()
        status_payload = await srv.status()

        # Both should report the same sanitized error category
        assert health_payload["last_error"] == "model_load_failed"
        assert status_payload["residency"]["last_error"] == "model_load_failed"

    def test_public_lifecycle_status_empty_string_error(self):
        """_public_lifecycle_status should treat '' the same as a real error."""
        import vllm_mlx.server as srv

        result = srv._public_lifecycle_status(
            {
                "state": "failed",
                "last_error": "",
            }
        )
        assert result["last_error"] == "model_load_failed"


class TestResponseModelFieldUsesServedName:
    """Verify response .model echoes _model_name (the served name), not
    whatever the client sent in request.model.

    Each test monkeypatches _validate_model_name to a no-op so the request
    can carry a distinct model string, proving the response field is sourced
    from the server-side served name rather than the request echo-back.
    """

    @pytest.mark.anyio
    async def test_completion_response_uses_served_model_name(self, monkeypatch):
        import vllm_mlx.server as srv
        from vllm_mlx.engine.base import GenerationOutput

        class FakeEngine:
            preserve_native_tool_format = False
            is_mllm = False

            async def generate(self, **kwargs):
                return GenerationOutput(
                    text="hello",
                    completion_tokens=1,
                    prompt_tokens=1,
                )

        served_name = "my-custom-served-name"

        async def fake_acquire(
            raw_request, *, total_timeout=None, deadline=None, count_activity=True
        ):
            return FakeEngine()

        async def fake_release(*, count_activity=True):
            pass

        monkeypatch.setattr(srv, "_validate_model_name", lambda _m: None)
        monkeypatch.setattr(srv, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(srv, "_release_default_engine", fake_release)
        monkeypatch.setattr(srv, "_model_name", served_name)
        monkeypatch.setattr(srv, "_default_max_tokens", 32)

        class FakeRawRequest:
            async def is_disconnected(self):
                return False

        request = srv.CompletionRequest(
            model="user-sent-model-name",
            prompt="hi",
            stream=False,
        )

        response = await srv.create_completion(request, FakeRawRequest())
        assert response.model == served_name
        assert response.model != "user-sent-model-name"

    @pytest.mark.anyio
    async def test_chat_completion_response_uses_served_model_name(self, monkeypatch):
        import vllm_mlx.server as srv
        from vllm_mlx.engine.base import GenerationOutput

        class FakeEngine:
            preserve_native_tool_format = False
            is_mllm = False
            tokenizer = None

            async def chat(self, **kwargs):
                return GenerationOutput(
                    text="hello",
                    completion_tokens=1,
                    prompt_tokens=1,
                )

        served_name = "my-custom-served-name"

        async def fake_acquire(
            raw_request, *, total_timeout=None, deadline=None, count_activity=True
        ):
            return FakeEngine()

        async def fake_release(*, count_activity=True):
            pass

        monkeypatch.setattr(srv, "_validate_model_name", lambda _m: None)
        monkeypatch.setattr(srv, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(srv, "_release_default_engine", fake_release)
        monkeypatch.setattr(srv, "_model_name", served_name)
        monkeypatch.setattr(srv, "_default_max_tokens", 32)
        monkeypatch.setattr(srv, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(srv, "_reasoning_parser", None)

        class FakeRawRequest:
            async def is_disconnected(self):
                return False

        request = srv.ChatCompletionRequest(
            model="user-sent-model-name",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )

        response = await srv.create_chat_completion(request, FakeRawRequest())
        assert response.model == served_name
        assert response.model != "user-sent-model-name"

    @pytest.mark.anyio
    async def test_anthropic_response_uses_served_model_name(self, monkeypatch):
        import json

        import vllm_mlx.server as srv
        from vllm_mlx.engine.base import GenerationOutput

        class FakeEngine:
            preserve_native_tool_format = False
            is_mllm = False
            tokenizer = None

            async def chat(self, **kwargs):
                return GenerationOutput(
                    text="hello",
                    completion_tokens=1,
                    prompt_tokens=1,
                )

        served_name = "my-custom-served-name"

        async def fake_acquire(
            raw_request, *, total_timeout=None, deadline=None, count_activity=True
        ):
            return FakeEngine()

        async def fake_release(*, count_activity=True):
            pass

        monkeypatch.setattr(srv, "_validate_model_name", lambda _m: None)
        monkeypatch.setattr(srv, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(srv, "_release_default_engine", fake_release)
        monkeypatch.setattr(srv, "_model_name", served_name)
        monkeypatch.setattr(srv, "_default_max_tokens", 32)
        monkeypatch.setattr(srv, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(srv, "_reasoning_parser", None)

        class FakeRawRequest:
            async def json(self):
                return {
                    "model": "user-sent-model-name",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "stream": False,
                }

            async def is_disconnected(self):
                return False

        response = await srv.create_anthropic_message(FakeRawRequest())
        body = json.loads(response.body.decode())
        assert body["model"] == served_name
        assert body["model"] != "user-sent-model-name"
