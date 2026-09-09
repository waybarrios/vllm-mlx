# SPDX-License-Identifier: Apache-2.0
"""Tests for Prometheus server metrics."""

import asyncio
import platform
import sys
from types import SimpleNamespace

import pytest

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


class FakeEngine:
    """Small fake engine for metrics endpoint tests."""

    model_name = "metrics-model"
    is_mllm = False
    preserve_native_tool_format = False
    tokenizer = None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def generate(self, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        return GenerationOutput(
            text="Hello",
            tokens=[1, 2],
            prompt_tokens=4,
            completion_tokens=2,
            finish_reason="stop",
        )

    async def stream_generate(self, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        yield GenerationOutput(
            text="Hel",
            new_text="Hel",
            prompt_tokens=4,
            completion_tokens=1,
            finished=False,
        )
        yield GenerationOutput(
            text="Hello",
            new_text="lo",
            prompt_tokens=4,
            completion_tokens=2,
            finish_reason="stop",
            finished=True,
        )

    async def chat(self, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        return GenerationOutput(
            text="Hello from chat",
            tokens=[1, 2, 3],
            prompt_tokens=5,
            completion_tokens=3,
            finish_reason="stop",
        )

    async def stream_chat(self, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        yield GenerationOutput(
            text="Hel",
            new_text="Hel",
            prompt_tokens=5,
            completion_tokens=1,
            finished=False,
        )
        yield GenerationOutput(
            text="Hello",
            new_text="lo",
            prompt_tokens=5,
            completion_tokens=2,
            finish_reason="stop",
            finished=True,
        )

    def get_stats(self):
        return {
            "engine_type": "simple",
            "is_mllm": False,
            "num_waiting": 2,
            "num_running": 1,
            "steps_executed": 7,
            "uptime_seconds": 42.5,
            "metal_active_memory_gb": 1.25,
            "metal_peak_memory_gb": 2.5,
            "metal_cache_memory_gb": 0.5,
            "memory_aware_cache": {
                "entry_count": 3,
                "hits": 4,
                "misses": 1,
                "evictions": 0,
                "hit_rate": 0.8,
                "memory_utilization": 0.25,
                "tokens_saved": 128,
                "current_memory_mb": 64,
                "max_memory_mb": 256,
            },
        }


@pytest.fixture()
def metrics_client(monkeypatch):
    """Create a TestClient with a fresh metrics collector."""
    from fastapi.testclient import TestClient

    import vllm_mlx.server as server
    from vllm_mlx.metrics import MetricsCollector

    collector = MetricsCollector()
    monkeypatch.setattr(server, "_metrics", collector)
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_model_name", "metrics-model")
    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_mcp_manager", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_parser_instance", None)
    monkeypatch.setattr(server, "_tool_call_parser", None)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
    monkeypatch.setattr(server, "_default_timeout", 30.0)
    monkeypatch.setattr(server, "_default_max_tokens", 128)
    monkeypatch.setattr(
        server,
        "_rate_limiter",
        server.RateLimiter(requests_per_minute=60, enabled=False),
    )

    with TestClient(server.app) as client:
        yield client, server, collector


class TestMetricsEndpoint:
    """Tests for Prometheus /metrics exposure and accounting."""

    def test_metrics_endpoint_disabled_returns_404(self, metrics_client):
        client, _server, collector = metrics_client

        collector.configure(enabled=False)
        response = client.get("/metrics")

        assert response.status_code == 404

    def test_metrics_endpoint_reports_dead_gauges_without_engine_or_manager(
        self, metrics_client
    ):
        client, server, collector = metrics_client

        collector.configure(enabled=True)
        assert server._engine is None
        assert server._model_manager is None

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "vllm_mlx_model_loaded 0.0" in response.text
        assert "vllm_mlx_scheduler_waiting_requests 0.0" in response.text

    def test_metrics_endpoint_scrapes_registry_mode_engine(
        self, metrics_client, monkeypatch
    ):
        client, server, collector = metrics_client

        collector.configure(enabled=True)

        class FakeModelManager:
            def __init__(self, engine):
                self._engine = engine

            def get_metrics_engine(self):
                return self._engine

            async def shutdown(self):
                return None

        monkeypatch.setattr(server, "_model_manager", FakeModelManager(FakeEngine()))

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "vllm_mlx_model_loaded 1.0" in response.text
        assert "vllm_mlx_scheduler_waiting_requests 2.0" in response.text
        assert (
            'vllm_mlx_cache_type{cache_type="memory_aware_cache"} 1.0' in response.text
        )

    def test_metrics_endpoint_registry_mode_dead_gauges_when_idle(
        self, metrics_client, monkeypatch
    ):
        client, server, collector = metrics_client

        collector.configure(enabled=True)

        class FakeModelManager:
            def get_metrics_engine(self):
                return None

            async def shutdown(self):
                return None

        monkeypatch.setattr(server, "_model_manager", FakeModelManager())

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "vllm_mlx_model_loaded 0.0" in response.text

    def test_metrics_endpoint_scrapes_runtime_stats(self, metrics_client, monkeypatch):
        client, server, collector = metrics_client

        collector.configure(enabled=True)
        monkeypatch.setattr(server, "_engine", FakeEngine())

        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain; version=")
        assert "charset=utf-8" in response.headers["content-type"]
        assert "vllm_mlx_model_loaded 1.0" in response.text
        assert 'vllm_mlx_engine_type{engine_type="simple"} 1.0' in response.text
        assert "vllm_mlx_scheduler_waiting_requests 2.0" in response.text
        assert (
            'vllm_mlx_cache_type{cache_type="memory_aware_cache"} 1.0' in response.text
        )

    def test_metrics_collapse_unmatched_paths(self, metrics_client, monkeypatch):
        client, server, collector = metrics_client

        collector.configure(enabled=True)
        monkeypatch.setattr(server, "_engine", FakeEngine())

        miss = client.get("/definitely-not-a-real-route")
        scrape = client.get("/metrics")

        assert miss.status_code == 404
        assert (
            'vllm_mlx_http_requests_total{method="GET",path="__unmatched__",status_code="404"} 1.0'
            in scrape.text
        )

    def test_completion_request_updates_metrics(self, metrics_client, monkeypatch):
        client, server, collector = metrics_client

        collector.configure(enabled=True)
        monkeypatch.setattr(server, "_engine", FakeEngine())

        response = client.post(
            "/v1/completions",
            json={
                "model": "metrics-model",
                "prompt": "Hello",
                "max_tokens": 8,
            },
        )
        scrape = client.get("/metrics")

        assert response.status_code == 200
        assert (
            'vllm_mlx_inference_requests_total{endpoint="completions",result="success",stream="false"} 1.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_prompt_tokens_total{endpoint="completions",stream="false"} 4.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_completion_tokens_total{endpoint="completions",stream="false"} 2.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_http_requests_total{method="POST",path="/v1/completions",status_code="200"} 1.0'
            in scrape.text
        )

    def test_streaming_chat_records_ttft_and_stream_metrics(
        self, metrics_client, monkeypatch
    ):
        client, server, collector = metrics_client

        collector.configure(enabled=True)
        monkeypatch.setattr(server, "_engine", FakeEngine())

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "metrics-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "max_tokens": 8,
            },
        ) as response:
            body = "".join(response.iter_text())

        scrape = client.get("/metrics")

        assert response.status_code == 200
        assert "data: [DONE]" in body
        assert (
            'vllm_mlx_inference_requests_total{endpoint="chat_completions",result="success",stream="true"} 1.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_inference_ttft_seconds_count{endpoint="chat_completions",stream="true"} 1.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_prompt_tokens_total{endpoint="chat_completions",stream="true"} 5.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_completion_tokens_total{endpoint="chat_completions",stream="true"} 2.0'
            in scrape.text
        )


class TestMetricsMiddlewareStreamingTiming:
    """Regression coverage for `_metrics_middleware`'s in-flight gauge timing.

    `call_next()` (Starlette's BaseHTTPMiddleware) resolves as soon as the
    ASGI response has *started* -- for a StreamingResponse this is long
    before the body finishes sending. These tests exercise
    `_metrics_middleware` directly, with a fake `call_next` returning a
    controllable fake streaming response, instead of driving a real
    FastAPI/TestClient/uvicorn round trip -- this makes the exact moment the
    gauge changes deterministic and independent of any real inference or
    real async I/O, per `TestDisconnectGuard`'s established pattern for this
    file.
    """

    PATH = "/v1/chat/completions"

    def _make_collector_and_request(self, monkeypatch):
        import vllm_mlx.server as server
        from vllm_mlx.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.configure(enabled=True)
        monkeypatch.setattr(server, "_metrics", collector)
        monkeypatch.setattr(
            server, "_metrics_path_for_request", lambda request: self.PATH
        )
        request = SimpleNamespace(method="POST")
        return server, collector, request

    @staticmethod
    def _inflight_value(collector, path):
        payload, _ = collector.render_metrics(engine=None, mcp_manager=None)
        needle = f'vllm_mlx_http_requests_in_flight{{method="POST",path="{path}"}}'
        for line in payload.decode().splitlines():
            if line.startswith(needle):
                return float(line.split()[-1])
        return None

    @staticmethod
    def _total_count(collector, path, status_code):
        payload, _ = collector.render_metrics(engine=None, mcp_manager=None)
        needle = (
            f'vllm_mlx_http_requests_total{{method="POST",path="{path}",'
            f'status_code="{status_code}"}}'
        )
        for line in payload.decode().splitlines():
            if line.startswith(needle):
                return float(line.split()[-1])
        return 0.0

    @pytest.mark.anyio
    async def test_streaming_body_holds_gauge_until_exhausted(self, monkeypatch):
        server, collector, request = self._make_collector_and_request(monkeypatch)
        resume = asyncio.Event()

        async def slow_body():
            yield b"data: role-preamble\n\n"
            await resume.wait()
            yield b"data: [DONE]\n\n"

        async def call_next(_request):
            return SimpleNamespace(body_iterator=slow_body(), status_code=200)

        # Prometheus only emits a line for a labeled gauge once it has been
        # touched at least once -- untouched, it's absent, not 0.0.
        assert self._inflight_value(collector, self.PATH) is None

        response = await server._metrics_middleware(request, call_next)
        # call_next() has already resolved here -- this is exactly the
        # moment the OLD code decremented the gauge, before any real
        # generation has happened.
        assert self._inflight_value(collector, self.PATH) == 1.0

        body_iter = response.body_iterator.__aiter__()
        first_chunk = await body_iter.__anext__()
        assert first_chunk == b"data: role-preamble\n\n"
        assert self._inflight_value(collector, self.PATH) == 1.0, (
            "must still be in flight after only the content-free preamble "
            "chunk has been sent -- this is the exact case the old gauge "
            "got wrong"
        )

        resume.set()
        remaining = [chunk async for chunk in body_iter]

        assert remaining == [b"data: [DONE]\n\n"]
        assert (
            self._inflight_value(collector, self.PATH) == 0.0
        ), "must be decremented once the body is actually exhausted"

    @pytest.mark.anyio
    async def test_error_mid_stream_decrements_exactly_once(self, monkeypatch):
        server, collector, request = self._make_collector_and_request(monkeypatch)

        async def erroring_body():
            yield b"data: role-preamble\n\n"
            raise RuntimeError("boom")

        async def call_next(_request):
            return SimpleNamespace(body_iterator=erroring_body(), status_code=200)

        response = await server._metrics_middleware(request, call_next)
        assert self._inflight_value(collector, self.PATH) == 1.0

        received = []
        with pytest.raises(RuntimeError, match="boom"):
            async for chunk in response.body_iterator:
                received.append(chunk)

        assert received == [b"data: role-preamble\n\n"]
        assert (
            self._inflight_value(collector, self.PATH) == 0.0
        ), "a mid-stream error must not leak the gauge stuck at 1"

        # The generator is already closed by the propagated exception,
        # closing it again (e.g. GC, or a caller's own cleanup) must not
        # double-decrement past zero.
        await response.body_iterator.aclose()
        assert self._inflight_value(collector, self.PATH) == 0.0

    @pytest.mark.anyio
    async def test_call_next_raising_before_any_response_finishes_once(
        self, monkeypatch
    ):
        server, collector, request = self._make_collector_and_request(monkeypatch)

        async def call_next(_request):
            raise RuntimeError("engine acquisition failed")

        with pytest.raises(RuntimeError, match="engine acquisition failed"):
            await server._metrics_middleware(request, call_next)

        assert self._inflight_value(collector, self.PATH) == 0.0
        assert self._total_count(collector, self.PATH, 500) == 1.0

    @pytest.mark.anyio
    async def test_non_streaming_single_chunk_body_still_settles_immediately(
        self, monkeypatch
    ):
        """Non-streaming responses are unaffected: their entire body is
        already produced before `call_next()` returns, so the returned
        body_iterator yields once and is done -- net timing is unchanged
        from before this fix."""
        server, collector, request = self._make_collector_and_request(monkeypatch)

        async def single_chunk_body():
            yield b'{"id": "cmpl-1", "object": "chat.completion"}'

        async def call_next(_request):
            return SimpleNamespace(body_iterator=single_chunk_body(), status_code=200)

        response = await server._metrics_middleware(request, call_next)
        assert self._inflight_value(collector, self.PATH) == 1.0

        received = [chunk async for chunk in response.body_iterator]

        assert received == [b'{"id": "cmpl-1", "object": "chat.completion"}']
        assert self._inflight_value(collector, self.PATH) == 0.0
