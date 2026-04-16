# SPDX-License-Identifier: Apache-2.0
"""Fail-first tests for server-side model lifecycle / residency behavior."""

from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
import time
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
    snapshot = {
        name: getattr(srv, name, sentinel)
        for name in global_names
    }

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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_failed_resident_surfaces_as_unhealthy_and_failed(
        self, monkeypatch
    ):
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

class TestLifecycleCli:
    """Lock in the first lifecycle CLI surface area."""

    def test_main_parses_auto_unload_idle_seconds_flag(self, monkeypatch):
        """The top-level CLI should accept an idle-unload timeout knob."""
        import vllm_mlx.cli as cli

        captured = {}

        def fake_serve_command(args):
            captured["auto_unload_idle_seconds"] = args.auto_unload_idle_seconds

        monkeypatch.setattr(cli, "serve_command", fake_serve_command)
        monkeypatch.setattr(
            cli.sys,
            "argv",
            [
                "vllm-mlx",
                "serve",
                "mlx-community/Qwen3-0.6B-8bit",
                "--auto-unload-idle-seconds",
                "300",
            ],
        )

        cli.main()

        assert captured["auto_unload_idle_seconds"] == 300

    def test_main_parses_lazy_load_model_flag(self, monkeypatch):
        """The top-level CLI should accept lazy lifecycle startup."""
        import vllm_mlx.cli as cli

        captured = {}

        def fake_serve_command(args):
            captured["lazy_load_model"] = args.lazy_load_model

        monkeypatch.setattr(cli, "serve_command", fake_serve_command)
        monkeypatch.setattr(
            cli.sys,
            "argv",
            [
                "vllm-mlx",
                "serve",
                "mlx-community/Qwen3-0.6B-8bit",
                "--lazy-load-model",
            ],
        )

        cli.main()

        assert captured["lazy_load_model"] is True

    def test_main_defaults_lazy_load_model_to_false(self, monkeypatch):
        """Serve startup should stay eager unless the user explicitly opts in."""
        import vllm_mlx.cli as cli

        captured = {}

        def fake_serve_command(args):
            captured["lazy_load_model"] = args.lazy_load_model

        monkeypatch.setattr(cli, "serve_command", fake_serve_command)
        monkeypatch.setattr(
            cli.sys,
            "argv",
            [
                "vllm-mlx",
                "serve",
                "mlx-community/Qwen3-0.6B-8bit",
            ],
        )

        cli.main()

        assert captured["lazy_load_model"] is False

    def test_serve_command_wires_auto_unload_idle_seconds_into_load_model(
        self, monkeypatch
    ):
        """serve_command should pass lifecycle config into model loading."""
        import uvicorn

        import vllm_mlx.cli as cli
        import vllm_mlx.server as srv

        captured = {}

        def fake_load_model(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(srv, "load_model", fake_load_model)
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)

        args = SimpleNamespace(
            model="mlx-community/Qwen3-0.6B-8bit",
            host="127.0.0.1",
            port=8000,
            max_num_seqs=256,
            prefill_batch_size=8,
            completion_batch_size=32,
            enable_prefix_cache=True,
            disable_prefix_cache=False,
            prefix_cache_size=100,
            cache_memory_mb=None,
            cache_memory_percent=0.20,
            no_memory_aware_cache=False,
            kv_cache_quantization=False,
            kv_cache_quantization_bits=8,
            kv_cache_quantization_group_size=64,
            kv_cache_min_quantize_tokens=256,
            stream_interval=7,
            max_tokens=32768,
            continuous_batching=False,
            use_paged_cache=False,
            paged_cache_block_size=64,
            max_cache_blocks=1000,
            chunked_prefill_tokens=0,
            enable_mtp=False,
            mtp_num_draft_tokens=1,
            mtp_optimistic=False,
            prefill_step_size=2048,
            specprefill=False,
            specprefill_threshold=8192,
            specprefill_keep_pct=0.3,
            specprefill_draft_model=None,
            mcp_config=None,
            api_key=None,
            rate_limit=0,
            timeout=300.0,
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            reasoning_parser=None,
            mllm=False,
            default_temperature=None,
            default_top_p=None,
            served_model_name=None,
            embedding_model=None,
            gpu_memory_utilization=0.90,
            enable_metrics=False,
            download_timeout=120,
            download_retries=3,
            mllm_prefill_step_size=None,
            lazy_load_model=True,
            auto_unload_idle_seconds=300,
        )

        cli.serve_command(args)

        assert captured["kwargs"]["stream_interval"] == 1
        assert captured["kwargs"]["auto_unload_idle_seconds"] == 300
        assert captured["kwargs"]["lazy_load_model"] is True

    def test_serve_command_wires_lazy_load_model_without_idle_unload(
        self, monkeypatch
    ):
        """Lazy startup should be forwarded even when idle auto-unload is disabled."""
        import uvicorn

        import vllm_mlx.cli as cli
        import vllm_mlx.server as srv

        captured = {}

        def fake_load_model(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(srv, "load_model", fake_load_model)
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)

        args = SimpleNamespace(
            model="mlx-community/Qwen3-0.6B-8bit",
            host="127.0.0.1",
            port=8000,
            max_num_seqs=256,
            prefill_batch_size=8,
            completion_batch_size=32,
            enable_prefix_cache=True,
            disable_prefix_cache=False,
            prefix_cache_size=100,
            cache_memory_mb=None,
            cache_memory_percent=0.20,
            no_memory_aware_cache=False,
            kv_cache_quantization=False,
            kv_cache_quantization_bits=8,
            kv_cache_quantization_group_size=64,
            kv_cache_min_quantize_tokens=256,
            stream_interval=7,
            max_tokens=32768,
            continuous_batching=False,
            use_paged_cache=False,
            paged_cache_block_size=64,
            max_cache_blocks=1000,
            chunked_prefill_tokens=0,
            enable_mtp=False,
            mtp_num_draft_tokens=1,
            mtp_optimistic=False,
            prefill_step_size=2048,
            specprefill=False,
            specprefill_threshold=8192,
            specprefill_keep_pct=0.3,
            specprefill_draft_model=None,
            mcp_config=None,
            api_key=None,
            rate_limit=0,
            timeout=300.0,
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            reasoning_parser=None,
            mllm=False,
            default_temperature=None,
            default_top_p=None,
            served_model_name=None,
            embedding_model=None,
            gpu_memory_utilization=0.90,
            enable_metrics=False,
            download_timeout=120,
            download_retries=3,
            mllm_prefill_step_size=None,
            lazy_load_model=True,
            auto_unload_idle_seconds=0.0,
        )

        cli.serve_command(args)

        assert captured["kwargs"]["stream_interval"] == 1
        assert captured["kwargs"]["auto_unload_idle_seconds"] == 0.0
        assert captured["kwargs"]["lazy_load_model"] is True

    def test_serve_command_preserves_mtp_scheduler_config_with_residency(
        self, monkeypatch
    ):
        """serve_command should keep batching/MTP config intact alongside residency."""
        import uvicorn

        import vllm_mlx.cli as cli
        import vllm_mlx.server as srv

        captured = {}

        def fake_load_model(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        class FakeSchedulerConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        monkeypatch.setattr(srv, "load_model", fake_load_model)
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
        monkeypatch.setitem(
            sys.modules,
            "vllm_mlx.scheduler",
            SimpleNamespace(SchedulerConfig=FakeSchedulerConfig),
        )

        args = SimpleNamespace(
            model="mlx-community/Qwen3-0.6B-8bit",
            host="127.0.0.1",
            port=8000,
            max_num_seqs=256,
            prefill_batch_size=8,
            completion_batch_size=32,
            enable_prefix_cache=True,
            disable_prefix_cache=False,
            prefix_cache_size=100,
            cache_memory_mb=None,
            cache_memory_percent=0.20,
            no_memory_aware_cache=False,
            kv_cache_quantization=False,
            kv_cache_quantization_bits=8,
            kv_cache_quantization_group_size=64,
            kv_cache_min_quantize_tokens=256,
            stream_interval=7,
            max_tokens=32768,
            continuous_batching=True,
            use_paged_cache=False,
            paged_cache_block_size=64,
            max_cache_blocks=1000,
            chunked_prefill_tokens=0,
            enable_mtp=True,
            mtp_num_draft_tokens=4,
            mtp_optimistic=True,
            prefill_step_size=2048,
            specprefill=False,
            specprefill_threshold=8192,
            specprefill_keep_pct=0.3,
            specprefill_draft_model=None,
            mcp_config=None,
            api_key=None,
            rate_limit=0,
            timeout=300.0,
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            reasoning_parser=None,
            mllm=False,
            default_temperature=None,
            default_top_p=None,
            served_model_name=None,
            embedding_model=None,
            gpu_memory_utilization=0.90,
            enable_metrics=False,
            download_timeout=120,
            download_retries=3,
            mllm_prefill_step_size=0,
            lazy_load_model=True,
            auto_unload_idle_seconds=300,
        )

        cli.serve_command(args)

        scheduler_config = captured["kwargs"]["scheduler_config"]
        assert captured["kwargs"]["use_batching"] is True
        assert captured["kwargs"]["stream_interval"] == 7
        assert captured["kwargs"]["auto_unload_idle_seconds"] == 300
        assert captured["kwargs"]["lazy_load_model"] is True
        assert scheduler_config.enable_mtp is True
        assert scheduler_config.mtp_num_draft_tokens == 4
        assert scheduler_config.mtp_optimistic is True

    def test_server_main_wires_lazy_load_model_without_idle_unload(
        self, monkeypatch
    ):
        """The python -m vllm_mlx.server entrypoint should forward lazy startup too."""
        import vllm_mlx.server as srv

        captured = {}

        def fake_load_model(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(srv, "load_model", fake_load_model)
        monkeypatch.setattr(srv, "load_embedding_model", lambda *args, **kwargs: None)
        monkeypatch.setattr(srv.uvicorn, "run", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vllm_mlx.server",
                "--model",
                "mlx-community/Qwen3-0.6B-8bit",
                "--lazy-load-model",
            ],
        )

        srv.main()

        assert captured["kwargs"]["auto_unload_idle_seconds"] == 0.0
        assert captured["kwargs"]["lazy_load_model"] is True

    def test_server_main_preserves_use_batching_with_residency_flags(
        self, monkeypatch
    ):
        """server.main should forward use_batching and residency knobs together."""
        import vllm_mlx.server as srv

        captured = {}

        def fake_load_model(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(srv, "load_model", fake_load_model)
        monkeypatch.setattr(srv, "load_embedding_model", lambda *args, **kwargs: None)
        monkeypatch.setattr(srv.uvicorn, "run", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vllm_mlx.server",
                "--model",
                "mlx-community/Qwen3-0.6B-8bit",
                "--continuous-batching",
                "--auto-unload-idle-seconds",
                "300",
                "--lazy-load-model",
            ],
        )

        srv.main()

        assert captured["kwargs"]["use_batching"] is True
        assert captured["kwargs"]["auto_unload_idle_seconds"] == 300.0
        assert captured["kwargs"]["lazy_load_model"] is True

    def test_serve_command_describes_lazy_startup_without_claiming_model_is_loaded(
        self, monkeypatch, capsys
    ):
        """Lazy startup output should mention the first request, not claim eager load."""
        import uvicorn

        import vllm_mlx.cli as cli
        import vllm_mlx.server as srv

        monkeypatch.setattr(srv, "load_model", lambda *args, **kwargs: None)
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)

        args = SimpleNamespace(
            model="mlx-community/Qwen3-0.6B-8bit",
            host="127.0.0.1",
            port=8000,
            max_num_seqs=256,
            prefill_batch_size=8,
            completion_batch_size=32,
            enable_prefix_cache=True,
            disable_prefix_cache=False,
            prefix_cache_size=100,
            cache_memory_mb=None,
            cache_memory_percent=0.20,
            no_memory_aware_cache=False,
            kv_cache_quantization=False,
            kv_cache_quantization_bits=8,
            kv_cache_quantization_group_size=64,
            kv_cache_min_quantize_tokens=256,
            stream_interval=1,
            max_tokens=32768,
            continuous_batching=False,
            use_paged_cache=False,
            paged_cache_block_size=64,
            max_cache_blocks=1000,
            chunked_prefill_tokens=0,
            enable_mtp=False,
            mtp_num_draft_tokens=1,
            mtp_optimistic=False,
            prefill_step_size=2048,
            specprefill=False,
            specprefill_threshold=8192,
            specprefill_keep_pct=0.3,
            specprefill_draft_model=None,
            mcp_config=None,
            api_key=None,
            rate_limit=0,
            timeout=300.0,
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            reasoning_parser=None,
            mllm=False,
            default_temperature=None,
            default_top_p=None,
            served_model_name=None,
            embedding_model=None,
            gpu_memory_utilization=0.90,
            enable_metrics=False,
            download_timeout=120,
            download_retries=3,
            mllm_prefill_step_size=None,
            lazy_load_model=True,
            auto_unload_idle_seconds=0.0,
        )

        cli.serve_command(args)
        out = capsys.readouterr().out

        assert "Loading model: mlx-community/Qwen3-0.6B-8bit" not in out
        assert "first request" in out


class TestResidencyManagerContracts:
    """Lock in the high-risk lifecycle invariants at the manager layer."""

    @pytest.mark.asyncio
    async def test_concurrent_acquire_single_flights_initial_load(self):
        """Concurrent acquires for one model should perform only one load."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        create_calls = 0
        started = 0

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                return None

        async def engine_factory(spec):
            nonlocal create_calls
            create_calls += 1
            await asyncio.sleep(0)
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=300,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        engines = await asyncio.gather(
            manager.acquire("default"),
            manager.acquire("default"),
            manager.acquire("default"),
        )

        assert create_calls == 1
        assert started == 1
        assert len({id(engine) for engine in engines}) == 1

        for _ in engines:
            await manager.release("default")

    @pytest.mark.asyncio
    async def test_unload_if_idle_respects_active_requests(self):
        """Idle unload should be blocked while a resident still has users."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        stopped = 0

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                nonlocal stopped
                stopped += 1

        now = {"value": 1000.0}

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        await manager.acquire("default")
        now["value"] += 120.0

        unloaded = await manager.unload_if_idle("default")

        assert unloaded is False
        assert stopped == 0
        assert manager.get_status("default")["state"] == "loaded"

        await manager.release("default")

    @pytest.mark.asyncio
    async def test_release_updates_last_used_at(self):
        """release() should refresh the timestamp used by idle-unload policy."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        now = {"value": 1000.0}

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                return None

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        await manager.acquire("default")
        before_release = manager.get_status("default")["last_used_at"]

        now["value"] = 1125.0
        await manager.release("default")

        after_release = manager.get_status("default")["last_used_at"]
        assert before_release != after_release
        assert after_release == 1125.0

    @pytest.mark.asyncio
    async def test_unload_after_idle_threshold_and_reload(self):
        """A released idle resident should unload and later reload cleanly."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        stopped = 0
        started = 0

        class FakeEngine:
            def __init__(self, generation):
                self.generation = generation

            async def start(self):
                nonlocal started
                started += 1
                return None

            async def stop(self):
                nonlocal stopped
                stopped += 1

        now = {"value": 1000.0}

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine(created)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = await manager.acquire("default")
        await manager.release("default")

        now["value"] += 120.0
        unloaded = await manager.unload_if_idle("default")

        assert unloaded is True
        assert stopped == 1
        assert manager.get_status("default")["state"] == "unloaded"

        second = await manager.acquire("default")

        assert created == 2
        assert started == 2
        assert first is not second
        assert manager.get_status("default")["state"] == "loaded"

        await manager.release("default")

    @pytest.mark.asyncio
    async def test_cancelled_cold_load_does_not_wedge_future_acquires(self):
        """Cancelling one waiter should not poison the shared resident load."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        load_gate = asyncio.Event()

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            await load_gate.wait()
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        await asyncio.sleep(0)

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        load_gate.set()
        second = await manager.acquire("default")

        assert second is not None
        assert created == 2
        assert started == 1
        assert stopped == 0
        assert manager.get_status("default")["state"] == "loaded"

        await manager.release("default")

    @pytest.mark.asyncio
    async def test_last_cancelled_waiter_cancels_cold_load(self):
        """Cancelling the final waiter should unwind the in-flight resident load."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        load_gate = asyncio.Event()
        start_entered = asyncio.Event()
        start_cancelled = asyncio.Event()

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1
                start_entered.set()
                try:
                    await load_gate.wait()
                except asyncio.CancelledError:
                    start_cancelled.set()
                    raise

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        try:
            await asyncio.wait_for(start_entered.wait(), timeout=1.0)

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            await asyncio.wait_for(start_cancelled.wait(), timeout=1.0)

            status = manager.get_status("default")
            assert status["state"] == "unloaded"
            assert manager._residents["default"]._loading_task is None
            assert stopped == 1

            load_gate.set()
            second = await asyncio.wait_for(manager.acquire("default"), timeout=1.0)

            assert second is not None
            assert created == 2
            assert started == 2
            assert manager.get_status("default")["state"] == "loaded"

            await manager.release("default")
        finally:
            load_gate.set()
            if not first.done():
                first.cancel()
                with suppress(asyncio.CancelledError):
                    await first

    @pytest.mark.asyncio
    async def test_cancelled_waiter_does_not_cancel_shared_cold_load(self):
        """A canceled waiter should not kill a cold load another waiter still needs."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        load_gate = asyncio.Event()
        start_entered = asyncio.Event()
        start_cancelled = asyncio.Event()

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1
                start_entered.set()
                try:
                    await load_gate.wait()
                except asyncio.CancelledError:
                    start_cancelled.set()
                    raise

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        second = asyncio.create_task(manager.acquire("default"))
        try:
            await asyncio.wait_for(start_entered.wait(), timeout=1.0)

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            await asyncio.sleep(0)
            assert not start_cancelled.is_set()

            load_gate.set()
            engine = await asyncio.wait_for(second, timeout=1.0)

            assert engine is not None
            assert created == 1
            assert started == 1
            assert stopped == 0
            assert manager.get_status("default")["state"] == "loaded"

            await manager.release("default")
        finally:
            load_gate.set()
            for task in (first, second):
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

    @pytest.mark.asyncio
    async def test_cancelled_last_waiter_suppresses_load_failure_from_cancel(self):
        """Abandoned cold loads should still surface as cancellation to the waiter."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        start_entered = asyncio.Event()

        class FakeEngine:
            def __init__(self, generation):
                self.generation = generation

            async def start(self):
                nonlocal started
                started += 1
                if self.generation != 1:
                    return None
                start_entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as exc:
                    raise RuntimeError("boom-from-start") from exc

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine(created)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(start_entered.wait(), timeout=1.0)
            first.cancel()
            await first

        status = manager.get_status("default")
        assert status["state"] == "unloaded"
        assert status["last_error"] is None
        assert manager._residents["default"]._loading_task is None
        assert stopped == 1

        second = await asyncio.wait_for(manager.acquire("default"), timeout=1.0)

        assert second is not None
        assert second.generation == 2
        assert created == 2
        assert started == 2
        assert manager.get_status("default")["state"] == "loaded"

        await manager.release("default")

    @pytest.mark.asyncio
    async def test_cancelled_prepare_for_start_does_not_finish_after_stop(self):
        """Cancelled cold loads should not let prepare_for_start keep mutating after stop."""
        import threading

        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        prepare_entered = threading.Event()
        allow_prepare_finish = threading.Event()
        prepare_finished = threading.Event()
        prepare_finished_after_stop = threading.Event()
        stop_called = threading.Event()

        class FakeEngine:
            def __init__(self, generation):
                self.generation = generation

            def prepare_for_start(self):
                prepare_entered.set()
                allow_prepare_finish.wait()
                time.sleep(0.01)
                if stop_called.is_set():
                    prepare_finished_after_stop.set()
                prepare_finished.set()

            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                nonlocal stopped
                stopped += 1
                stop_called.set()

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine(created)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        try:
            assert await asyncio.wait_for(
                asyncio.to_thread(prepare_entered.wait, 1.0),
                timeout=1.0,
            )

            first.cancel()
            allow_prepare_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await first

            assert await asyncio.wait_for(
                asyncio.to_thread(prepare_finished.wait, 1.0),
                timeout=1.0,
            )
            assert not prepare_finished_after_stop.is_set()

            await asyncio.wait_for(
                _wait_for_resident_state(manager, "default", "unloaded"),
                timeout=1.0,
            )

            status = manager.get_status("default")
            assert status["state"] == "unloaded"
            assert manager._residents["default"]._loading_task is None

            second = await asyncio.wait_for(manager.acquire("default"), timeout=1.0)

            assert second is not None
            assert second.generation == 2
            assert created == 2
            assert started == 1
            assert stopped == 1
            assert manager.get_status("default")["state"] == "loaded"

            await manager.release("default")
        finally:
            allow_prepare_finish.set()
            if not first.done():
                first.cancel()
                with suppress(asyncio.CancelledError):
                    await first

    @pytest.mark.asyncio
    async def test_late_joining_waiter_retries_abandoned_cold_load(self):
        """A waiter joining during abandoned-load cleanup should retry instead of inheriting cancel."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        start_entered = asyncio.Event()
        stop_entered = asyncio.Event()
        allow_stop = asyncio.Event()

        class FakeEngine:
            def __init__(self, generation):
                self.generation = generation

            async def start(self):
                nonlocal started
                started += 1
                if self.generation != 1:
                    return None
                start_entered.set()
                await asyncio.Event().wait()

            async def stop(self):
                nonlocal stopped
                stopped += 1
                stop_entered.set()
                await allow_stop.wait()

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine(created)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        try:
            await asyncio.wait_for(start_entered.wait(), timeout=1.0)

            first.cancel()

            await asyncio.wait_for(stop_entered.wait(), timeout=1.0)

            second = asyncio.create_task(manager.acquire("default"))
            try:
                await asyncio.sleep(0)

                allow_stop.set()
                with pytest.raises(asyncio.CancelledError):
                    await first
                engine = await asyncio.wait_for(second, timeout=1.0)

                assert engine is not None
                assert engine.generation == 2
                assert created == 2
                assert started == 2
                assert stopped == 1
                assert manager.get_status("default")["state"] == "loaded"
            finally:
                if not second.done():
                    second.cancel()
                    with suppress(asyncio.CancelledError):
                        await second

            await manager.release("default")
        finally:
            allow_stop.set()
            if not first.done():
                first.cancel()
                with suppress(asyncio.CancelledError):
                    await first

    @pytest.mark.asyncio
    async def test_cancelled_shared_load_during_state_commit_recovers_cleanly(self):
        """Cancelling the shared load during state commit should stop the engine and recover."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0

        class CommitGateLock:
            def __init__(self):
                self._real = asyncio.Lock()
                self.block_next_enter = False
                self.commit_waiting = asyncio.Event()
                self.allow_commit = asyncio.Event()

            async def __aenter__(self):
                if self.block_next_enter:
                    self.block_next_enter = False
                    self.commit_waiting.set()
                    await self.allow_commit.wait()
                await self._real.acquire()
                return self

            async def __aexit__(self, exc_type, exc, tb):
                self._real.release()

        lock = CommitGateLock()

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine()

        async def on_engine_loaded(spec, engine):
            lock.block_next_enter = True

        manager = ResidencyManager(
            engine_factory=engine_factory,
            on_engine_loaded=on_engine_loaded,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager._lock = lock
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        await lock.commit_waiting.wait()

        loading_task = manager._residents["default"]._loading_task
        assert loading_task is not None
        loading_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await first

        assert stopped == 1
        assert manager.get_status("default")["state"] == "unloaded"

        lock.allow_commit.set()
        second = await asyncio.wait_for(manager.acquire("default"), timeout=1.0)

        assert created == 2
        assert started == 2
        assert manager.get_status("default")["state"] == "loaded"

        await manager.release("default")

    @pytest.mark.asyncio
    async def test_cancelled_load_cleanup_handles_legacy_tasks_without_uncancel_api(
        self, monkeypatch
    ):
        """Cancellation cleanup should still work on supported runtimes without Task.uncancel()."""
        import vllm_mlx.lifecycle as lifecycle_mod
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager, ResidentState

        stopped = 0

        class LegacyTask:
            def cancel(self):
                return None

        class FakeEngine:
            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            raise AssertionError("engine_factory should not be called")

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        resident = manager._residents["default"]
        resident.state = ResidentState.LOADING
        resident._loading_task = object()

        monkeypatch.setattr(
            lifecycle_mod.asyncio,
            "current_task",
            lambda: LegacyTask(),
        )

        await manager._cleanup_cancelled_load(resident, FakeEngine())

        assert stopped == 1
        assert resident.state == ResidentState.UNLOADED
        assert resident._loading_task is None

    @pytest.mark.asyncio
    async def test_shutdown_cancels_inflight_cold_load(self):
        """shutdown() should not allow a cold load to complete afterward."""
        import threading

        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        started = 0
        load_gate = asyncio.Event()

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                return None

        async def engine_factory(spec):
            await load_gate.wait()
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        acquire_task = asyncio.create_task(manager.acquire("default"))
        await asyncio.sleep(0)

        shutdown_task = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)

        load_gate.set()
        await shutdown_task

        with pytest.raises(asyncio.CancelledError):
            await acquire_task

        assert started == 0
        assert manager.get_engine("default") is None
        assert manager.get_status("default")["state"] == "unloaded"

    @pytest.mark.asyncio
    async def test_shutdown_canceled_prepare_error_unwinds_to_unloaded(self):
        """shutdown() should suppress prepare errors from a canceled cold load."""
        import threading

        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        prepare_entered = threading.Event()
        allow_prepare_finish = threading.Event()
        started = 0

        class FakeEngine:
            def prepare_for_start(self):
                prepare_entered.set()
                allow_prepare_finish.wait()
                raise RuntimeError("prepare boom")

            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                return None

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        acquire_task = asyncio.create_task(manager.acquire("default"))
        assert await asyncio.wait_for(
            asyncio.to_thread(prepare_entered.wait, 1.0),
            timeout=1.0,
        )

        shutdown_task = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)
        allow_prepare_finish.set()
        await shutdown_task

        with pytest.raises(asyncio.CancelledError):
            await acquire_task

        status = manager.get_status("default")
        assert started == 0
        assert manager.get_engine("default") is None
        assert status["state"] == "unloaded"
        assert status["last_error"] is None

    @pytest.mark.asyncio
    async def test_shutdown_raises_if_loaded_resident_cannot_unload(self):
        """shutdown() should not report success when resident unload fails."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                raise RuntimeError("stop boom")

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        await manager.acquire("default")
        await manager.release("default")

        with pytest.raises(RuntimeError):
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_attempts_later_residents_after_one_unload_failure(self):
        """shutdown() should keep unloading later residents after one unload fails."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        stopped = []

        class FakeEngine:
            def __init__(self, model_key):
                self.model_key = model_key

            async def start(self):
                return None

            async def stop(self):
                stopped.append(self.model_key)
                if self.model_key == "a":
                    raise RuntimeError("stop boom")

        async def engine_factory(spec):
            return FakeEngine(spec.model_key)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="a", model_name="model-a"))
        manager.register_model(ModelSpec(model_key="b", model_name="model-b"))

        await manager.acquire("a")
        await manager.release("a")
        await manager.acquire("b")
        await manager.release("b")

        with pytest.raises(RuntimeError, match="a"):
            await manager.shutdown()

        assert stopped == ["a", "b"]
        assert manager.get_engine("b") is None
        assert manager.get_status("b")["state"] == "unloaded"

    @pytest.mark.asyncio
    async def test_acquire_retries_if_idle_unload_wins_the_boundary(self, monkeypatch):
        """Acquire should not hand back an engine that unloaded in the claim gap."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        stopped = 0
        started = 0
        now = {"value": 1000.0}

        class FakeEngine:
            def __init__(self, generation):
                self.generation = generation

            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine(created)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        original = await manager.acquire("default")
        await manager.release("default")
        now["value"] += 120.0

        entered_gap = asyncio.Event()
        continue_gap = asyncio.Event()
        original_ensure_loaded = manager.ensure_loaded

        async def delayed_ensure_loaded(model_key):
            engine = await original_ensure_loaded(model_key)
            if not entered_gap.is_set():
                entered_gap.set()
                await continue_gap.wait()
            return engine

        monkeypatch.setattr(manager, "ensure_loaded", delayed_ensure_loaded)

        acquire_task = asyncio.create_task(manager.acquire("default"))
        await entered_gap.wait()

        unloaded = await manager.unload_if_idle("default")
        continue_gap.set()
        replacement = await acquire_task

        assert unloaded is True
        assert stopped == 1
        assert created == 2
        assert started == 2
        assert replacement is not original
        assert manager.get_status("default")["state"] == "loaded"
        assert manager.get_status("default")["active_requests"] == 1

        await manager.release("default")

    @pytest.mark.asyncio
    async def test_failed_unload_keeps_live_engine_tracked(self):
        """Unload failure should not orphan a still-live engine."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        now = {"value": 1000.0}

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                raise RuntimeError("boom")

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        original = await manager.acquire("default")
        await manager.release("default")
        now["value"] += 120.0

        unloaded = await manager.unload_if_idle("default")
        replacement = await manager.acquire("default")

        assert unloaded is False
        assert manager.get_engine("default") is original
        assert replacement is original
        assert manager.get_status("default")["state"] == "loaded"
        assert manager.get_status("default")["last_error"] == "boom"
        assert created == 1

        await manager.release("default")

    @pytest.mark.asyncio
    async def test_register_model_rejects_replacing_live_resident(self):
        """register_model() should not orphan an already loaded resident."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        stopped = 0

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="first-model"))

        await manager.acquire("default")
        await manager.release("default")

        with pytest.raises(RuntimeError, match="Cannot replace resident model"):
            manager.register_model(
                ModelSpec(model_key="default", model_name="replacement-model")
            )

        await manager.shutdown()

        assert created == 1
        assert stopped == 1

    @pytest.mark.asyncio
    async def test_failed_cold_load_cleans_up_partial_engine(self):
        """A start() failure should still stop the partially built engine."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        stopped = 0

        class FakeEngine:
            async def start(self):
                raise RuntimeError("boom")

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        with pytest.raises(RuntimeError, match="boom"):
            await manager.acquire("default")

        assert manager.get_status("default")["state"] == "failed"
        assert stopped == 1

    @pytest.mark.asyncio
    async def test_cancelled_waiter_does_not_cancel_shared_unload(self):
        """A canceled acquire waiter should not cancel the shared unload task."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        now = {"value": 1000.0}
        stop_gate = asyncio.Event()

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                await stop_gate.wait()

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        await manager.acquire("default")
        await manager.release("default")
        now["value"] += 120.0

        unload_task = asyncio.create_task(manager.unload_if_idle("default"))
        await asyncio.sleep(0)

        acquire_task = asyncio.create_task(manager.acquire("default"))
        await asyncio.sleep(0)
        acquire_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await acquire_task

        stop_gate.set()
        unloaded = await unload_task

        assert unloaded is True
        assert manager.get_status("default")["state"] == "unloaded"

    @pytest.mark.asyncio
    async def test_cancelled_unload_waiter_does_not_cancel_shared_unload_task(self):
        """Cancelling one unload waiter should not cancel the shared unload operation."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        now = {"value": 1000.0}
        stop_gate = asyncio.Event()

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                await stop_gate.wait()

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        await manager.acquire("default")
        await manager.release("default")
        now["value"] += 120.0

        first_waiter = asyncio.create_task(manager.unload_if_idle("default"))
        await asyncio.sleep(0)

        second_waiter = asyncio.create_task(manager.unload_if_idle("default"))
        await asyncio.sleep(0)

        first_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_waiter

        stop_gate.set()
        unloaded = await second_waiter

        assert unloaded is True
        assert manager.get_status("default")["state"] == "unloaded"


class TestResidencyManagerEdgeCases:
    """Edge-case coverage for manager state transitions and error recovery."""

    @pytest.mark.asyncio
    async def test_unload_if_idle_when_already_unloaded(self):
        """unload_if_idle on an unloaded resident should return False, not corrupt state."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        class FakeEngine:
            async def start(self):
                pass

            async def stop(self):
                pass

        async def factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            factory, time_fn=lambda: 1000.0, auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        # Never loaded — should be a no-op
        result = await manager.unload_if_idle("m")
        assert result is False
        assert manager.get_status("m")["state"] == "unloaded"

    @pytest.mark.asyncio
    async def test_unload_if_idle_during_active_load(self):
        """unload_if_idle should return False when a load is in progress."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        load_gate = asyncio.Event()

        class FakeEngine:
            async def start(self):
                await load_gate.wait()

            async def stop(self):
                pass

        async def factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            factory, time_fn=lambda: 1000.0, auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        load_task = asyncio.create_task(manager.ensure_loaded("m"))
        await asyncio.sleep(0)  # let load start

        result = await manager.unload_if_idle("m")
        assert result is False

        load_gate.set()
        await load_task
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_register_model_after_failure_replaces_entry(self):
        """A model in FAILED state should be replaceable via register_model."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        call_count = 0

        class FailEngine:
            async def start(self):
                raise RuntimeError("boom")

            async def stop(self):
                pass

        class GoodEngine:
            async def start(self):
                pass

            async def stop(self):
                pass

        async def factory(spec):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FailEngine()
            return GoodEngine()

        manager = ResidencyManager(factory, auto_unload_idle_seconds=0)
        spec = ModelSpec(model_key="m", model_name="test")
        manager.register_model(spec)

        with pytest.raises(RuntimeError, match="boom"):
            await manager.ensure_loaded("m")
        assert manager.get_status("m")["state"] == "failed"

        # Re-register should succeed because the model is dormant/failed
        manager.register_model(spec)
        assert manager.get_status("m")["state"] == "unloaded"

        engine = await manager.ensure_loaded("m")
        assert engine is not None
        assert manager.get_status("m")["state"] == "loaded"
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_engine_factory_raises_before_engine_created(self):
        """If the factory itself raises (not engine.start), state should be FAILED."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        async def failing_factory(spec):
            raise RuntimeError("model not found")

        manager = ResidencyManager(failing_factory, auto_unload_idle_seconds=0)
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        with pytest.raises(RuntimeError, match="model not found"):
            await manager.ensure_loaded("m")

        status = manager.get_status("m")
        assert status["state"] == "failed"
        assert "model not found" in status["last_error"]

    @pytest.mark.asyncio
    async def test_rapid_acquire_release_refcount_stays_consistent(self):
        """Rapid acquire/release cycles should keep active_requests consistent."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        class FakeEngine:
            async def start(self):
                pass

            async def stop(self):
                pass

        async def factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            factory, time_fn=lambda: 1000.0, auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        # Rapid sequential acquire/release
        for _ in range(50):
            await manager.acquire("m")
            await manager.release("m")

        assert manager.get_status("m")["active_requests"] == 0

        # Concurrent acquire then sequential release
        engines = await asyncio.gather(*[manager.acquire("m") for _ in range(20)])
        assert manager.get_status("m")["active_requests"] == 20
        for _ in engines:
            await manager.release("m")
        assert manager.get_status("m")["active_requests"] == 0

    @pytest.mark.asyncio
    async def test_shutdown_shields_unload_from_cancellation(self):
        """Cancelling shutdown() should not orphan a half-stopped engine."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        stop_started = asyncio.Event()
        stop_gate = asyncio.Event()
        stopped = 0

        class FakeEngine:
            async def start(self):
                pass

            async def stop(self):
                nonlocal stopped
                stop_started.set()
                await stop_gate.wait()
                stopped += 1

        async def factory(spec):
            return FakeEngine()

        manager = ResidencyManager(factory, auto_unload_idle_seconds=0)
        manager.register_model(ModelSpec(model_key="m", model_name="test"))
        await manager.ensure_loaded("m")

        shutdown_task = asyncio.create_task(manager.shutdown())
        await stop_started.wait()

        # Cancel shutdown while engine.stop() is in flight
        shutdown_task.cancel()
        await asyncio.sleep(0)

        # Let engine.stop() complete
        stop_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await shutdown_task

        # The engine should still have been fully stopped
        assert stopped == 1
        assert manager.get_status("m")["state"] == "unloaded"


class TestCompletionStreamingRelease:
    """Verify the completion endpoint releases residency on all paths."""

    @pytest.mark.asyncio
    async def test_completion_nonstreaming_error_releases_active_request(self, monkeypatch):
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

        async def fake_acquire(raw_request, *, total_timeout=None, deadline=None, count_activity=True):
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
        assert releases["count"] == 1, (
            "Non-streaming completion must release residency on generation errors"
        )

    @pytest.mark.asyncio
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
                    text="done", new_text="done",
                    finish_reason="stop",
                    completion_tokens=1, prompt_tokens=1,
                    finished=True,
                )

        async def fake_acquire(raw_request, *, total_timeout=None, deadline=None, count_activity=True):
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

        assert releases["count"] == 1, (
            "Streaming completion must release residency via cleanup callback"
        )


class TestStatusEndpointEngineRace:
    """Verify status/health endpoints handle concurrent engine unload."""

    @pytest.mark.asyncio
    async def test_status_endpoint_handles_engine_unloaded_during_call(self, monkeypatch):
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

    @pytest.mark.asyncio
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
    """Tool parser should use the request-local engine, not the global."""

    @pytest.mark.asyncio
    async def test_parse_tool_calls_uses_local_engine_not_global(self, monkeypatch):
        """_parse_tool_calls_with_parser should use the locally-acquired engine."""
        import vllm_mlx.server as srv

        # Global engine is None (model unloaded between acquire and parser init)
        monkeypatch.setattr(srv, "_engine", None)
        monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(srv, "_tool_call_parser", "hermes")
        monkeypatch.setattr(srv, "_tool_parser_instance", None)

        # The parser init reads _engine — with a None global, it should
        # either use the local engine or fail gracefully (not AttributeError)
        try:
            text, tools = srv._parse_tool_calls_with_parser("hello", None)
        except AttributeError:
            pytest.fail(
                "_parse_tool_calls_with_parser crashed because it read "
                "the global _engine (None) instead of using the local engine"
            )


class TestLifecycleFailureHandling:
    """Regression coverage for lifecycle failure paths."""

    @pytest.mark.asyncio
    async def test_anthropic_validation_error_releases_resident(self, monkeypatch):
        """Malformed Anthropic payloads should not touch residency at all."""
        from pydantic import ValidationError

        import vllm_mlx.server as srv

        calls = {"acquires": 0, "releases": 0}

        class FakeRequest:
            async def json(self):
                return {}

        class FakeEngine:
            preserve_native_tool_format = False

        async def fake_acquire():
            calls["acquires"] += 1
            return FakeEngine()

        async def fake_release():
            calls["releases"] += 1

        monkeypatch.setattr(srv, "_acquire_default_engine", fake_acquire)
        monkeypatch.setattr(srv, "_release_default_engine", fake_release)

        with pytest.raises(ValidationError):
            await srv.create_anthropic_message(FakeRequest())

        assert calls["acquires"] == 0
        assert calls["releases"] == 0

    @pytest.mark.asyncio
    async def test_chat_completion_prep_error_releases_resident(self, monkeypatch):
        """Prep failures after acquire should still release chat residency."""
        import vllm_mlx.server as srv

        calls = {"acquires": 0, "releases": 0}

        class FakeEngine:
            is_mllm = False
            preserve_native_tool_format = False

        async def fake_acquire():
            calls["acquires"] += 1
            return FakeEngine()

        async def fake_release():
            calls["releases"] += 1

        def fake_extract(messages, preserve_native_format):
            return ([{"role": "user", "content": "hi"}], [], [])

        def fake_convert_tools(_tools):
            raise RuntimeError("boom")

        monkeypatch.setattr(srv, "_acquire_default_engine", fake_acquire)
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
            timeout=None,
        )

        with pytest.raises(RuntimeError, match="boom"):
            await srv.create_chat_completion(request, SimpleNamespace())

        assert calls["acquires"] == 1
        assert calls["releases"] == 1

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
            assert "Lifecycle cleanup failed while preserving the original exception" in caplog.text
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
                monkeypatch.setattr(engine, "_start_llm", unexpected_start_llm, raising=False)
            else:
                async def unexpected_start_mllm():
                    raise AssertionError("_start_mllm should not run after cancellation")

                monkeypatch.setattr(engine, "_is_mllm", True, raising=False)
                monkeypatch.setattr(engine, "_start_mllm", unexpected_start_mllm, raising=False)

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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_run_blocking_startup_work_waits_for_thread_under_repeated_cancel(self):
        """Repeated cancellation should not return before blocking startup work finishes."""
        import threading
        import time

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

    @pytest.mark.asyncio
    async def test_blocking_cache_io_waits_for_thread_under_repeated_cancel(
        self,
    ):
        """Repeated cancellation should not return before cache I/O thread finishes."""
        import threading
        import time

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

        task = asyncio.create_task(srv._run_blocking_engine_cache_io(blocking_io, FakeEngine()))
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

    @pytest.mark.asyncio
    async def test_prepare_engine_start_waits_for_thread_under_repeated_cancel(self):
        """Repeated cancellation of a residency cold load must not return before
        prepare_for_start() finishes, otherwise the thread can keep mutating
        model state past request/shutdown boundaries."""
        import threading
        import time

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

    @pytest.mark.asyncio
    async def test_run_blocking_startup_work_does_not_livelock_on_cancelled_inner_task(self):
        """If the inner to_thread task ends up cancelled, the drain loop must
        exit instead of spinning forever on CancelledError."""
        from vllm_mlx.engine.base import run_blocking_startup_work

        def work_that_will_be_cancelled():
            raise asyncio.CancelledError()

        task = asyncio.create_task(run_blocking_startup_work(work_that_will_be_cancelled))
        await asyncio.sleep(0)
        task.cancel()

        # Must complete promptly — a livelock would hang here
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
            pytest.fail(
                "Drain loop livelocked — load_task did not complete within 2s"
            )

        # Clean up
        with suppress(Exception):
            await manager.shutdown()

    @pytest.mark.asyncio
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
            monkeypatch.setattr(engine, "_start_llm", cancellable_start_phase, raising=False)
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
        fake_manager = SimpleNamespace(get_engine=lambda model_key: engine_state["engine"])

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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
            if created_loop["loop"] is not None and not created_loop["loop"].is_closed():
                created_loop["loop"].close()
            monkeypatch.setattr(srv, "_engine", None, raising=False)

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_lazy_cold_acquire_does_not_block_event_loop(self, monkeypatch):
        """Cold resident startup should not freeze unrelated event-loop work."""
        import time

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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
            return ([{"role": "user", "content": "hi"}], [], [])

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

    @pytest.mark.asyncio
    async def test_completion_disconnect_covers_cold_resident_acquire(self, monkeypatch):
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

        request_task = asyncio.create_task(srv.create_completion(request, FakeRequest()))
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_anthropic_messages_refresh_idle_unload_activity(
        self, monkeypatch
    ):
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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


class TestHasMediaContentShared:
    """Verify the shared has_media_content helper handles both dict and Pydantic messages."""

    def test_detects_image_url_in_dict_messages(self):
        from vllm_mlx.api.utils import has_media_content

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
            },
        ]
        assert has_media_content(messages) is True

    def test_returns_false_for_text_only_dicts(self):
        from vllm_mlx.api.utils import has_media_content

        messages = [
            {"role": "user", "content": "just text"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        assert has_media_content(messages) is False

    def test_detects_video_url_in_pydantic_style_messages(self):
        from vllm_mlx.api.utils import has_media_content

        class FakePart:
            def __init__(self, type_):
                self.type = type_

        class FakeMsg:
            def __init__(self, content):
                self.content = content

        messages = [FakeMsg([FakePart("text"), FakePart("video_url")])]
        assert has_media_content(messages) is True

    def test_handles_audio_type(self):
        from vllm_mlx.api.utils import has_media_content

        messages = [{"role": "user", "content": [{"type": "audio"}]}]
        assert has_media_content(messages) is True

    def test_handles_none_content(self):
        from vllm_mlx.api.utils import has_media_content

        messages = [{"role": "system", "content": None}]
        assert has_media_content(messages) is False

    def test_handles_string_content(self):
        from vllm_mlx.api.utils import has_media_content

        messages = [{"role": "user", "content": "hello"}]
        assert has_media_content(messages) is False


class TestLifecycleLoopIdleEvent:
    """Verify that the lifecycle loop uses an asyncio.Event for gating."""

    @pytest.mark.asyncio
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
            assert iterations == 0, (
                f"Loop iterated {iterations} times while event was cleared"
            )

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

        result = srv._public_lifecycle_status({
            "state": "loaded",
            "last_error": None,
        })
        assert result["last_error"] is None

    def test_raw_error_replaced_with_category(self):
        import vllm_mlx.server as srv

        result = srv._public_lifecycle_status({
            "state": "failed",
            "last_error": "OSError: /tmp/model not found",
        })
        assert result["last_error"] == "model_load_failed"

    def test_returns_none_for_none_input(self):
        import vllm_mlx.server as srv

        assert srv._public_lifecycle_status(None) is None

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_health_omits_last_error_for_healthy_resident(
        self, monkeypatch
    ):
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

    @pytest.mark.asyncio
    async def test_health_and_status_agree_on_empty_string_error(
        self, monkeypatch
    ):
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

        result = srv._public_lifecycle_status({
            "state": "failed",
            "last_error": "",
        })
        assert result["last_error"] == "model_load_failed"


class TestSuspendCancellationDedup:
    """Verify lifecycle.py uses the shared suspend_cancellation from base."""

    def test_lifecycle_does_not_define_its_own_suspend_cancellation(self):
        """lifecycle.py should import suspend_cancellation, not redefine it."""
        import vllm_mlx.lifecycle as lc
        from vllm_mlx.engine.base import suspend_cancellation

        assert lc.suspend_cancellation is suspend_cancellation

    @pytest.mark.asyncio
    async def test_residency_manager_uses_shared_suspend_cancellation(self):
        """ResidencyManager cleanup paths should work with the shared helper."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        stopped = 0

        class FakeEngine:
            async def start(self):
                pass

            async def stop(self):
                nonlocal stopped
                stopped += 1

            def prepare_for_start(self):
                pass

        async def factory(spec):
            return FakeEngine()

        manager = ResidencyManager(factory, auto_unload_idle_seconds=0)
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        engine = await manager.ensure_loaded("m")
        assert engine is not None
        await manager.shutdown()
        assert stopped == 1


class TestResponseModelFieldUsesServedName:
    """Verify response .model echoes _model_name (the served name), not
    whatever the client sent in request.model.

    Each test monkeypatches _validate_model_name to a no-op so the request
    can carry a distinct model string, proving the response field is sourced
    from the server-side served name rather than the request echo-back.
    """

    @pytest.mark.asyncio
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

        async def fake_acquire(raw_request, *, total_timeout=None, deadline=None, count_activity=True):
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

    @pytest.mark.asyncio
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

        async def fake_acquire(raw_request, *, total_timeout=None, deadline=None, count_activity=True):
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

    @pytest.mark.asyncio
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

        async def fake_acquire(raw_request, *, total_timeout=None, deadline=None, count_activity=True):
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
