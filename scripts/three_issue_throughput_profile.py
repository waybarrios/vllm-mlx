"""Opt-in pytest phase timings for the unchanged throughput diagnostic.

Wall times are inclusive and may overlap across async tasks and threads. Thread
CPU time includes other work on that thread while an async call is suspended. MLX
calls remain lazy: this plugin never adds synchronization or model warmup.
Instrumented measurements are diagnostic only, never acceptance evidence.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import inspect
from itertools import count
import json
import os
from pathlib import Path
import threading
import time

import pytest

_TRACE = pytest.StashKey[dict]()


class CallTimings:
    def __init__(self):
        self.calls = []
        self.started = time.perf_counter()
        self._ids = count()
        self._parent = ContextVar("throughput_parent", default=None)

    @contextmanager
    def span(self, name):
        call = {
            "id": next(self._ids),
            "parent_id": self._parent.get(),
            "name": name,
            "thread_id": threading.get_ident(),
            "error": None,
        }
        token = self._parent.set(call["id"])
        started = time.perf_counter()
        cpu = time.thread_time()
        try:
            yield
        except BaseException as error:
            call["error"] = type(error).__name__
            raise
        finally:
            call.update(
                start_seconds=started - self.started,
                wall_seconds=time.perf_counter() - started,
                thread_cpu_seconds=time.thread_time() - cpu,
            )
            self._parent.reset(token)
            self.calls.append(call)

    def wrap(self, name, function):
        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def measured(*args, **kwargs):
                with self.span(name):
                    return await function(*args, **kwargs)

        else:

            @wraps(function)
            def measured(*args, **kwargs):
                with self.span(name):
                    return function(*args, **kwargs)

        return measured


@contextmanager
def profile_calls(targets):
    trace = CallTimings()
    with pytest.MonkeyPatch.context() as patch:
        for owner, attribute, name in targets:
            patch.setattr(owner, attribute, trace.wrap(name, getattr(owner, attribute)))
        yield trace


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if (
        item.path.name != "test_continuous_batching.py"
        or item.originalname != "test_batching_improves_throughput"
    ):
        yield
        return

    if not os.environ.get("VLLM_MLX_THROUGHPUT_PROFILE"):
        raise pytest.UsageError("Phase timings require an explicit output path")
    modules = {
        name: importlib.import_module(name)
        for name in (
            "vllm_mlx.scheduler",
            "vllm_mlx.engine_core",
            "vllm_mlx.output_collector",
            "mlx_lm.generate",
            "mlx.core",
        )
    }
    scheduler = modules["vllm_mlx.scheduler"].Scheduler
    engine = modules["vllm_mlx.engine_core"].EngineCore
    generator = modules["mlx_lm.generate"].BatchGenerator
    prefill = modules["mlx_lm.generate"].PromptProcessingBatch
    decode = modules["mlx_lm.generate"].GenerationBatch
    collector = modules["vllm_mlx.output_collector"].RequestOutputCollector
    targets = [
        (scheduler, "step", "scheduler.step"),
        (scheduler, "_schedule_waiting", "scheduler.admit"),
        (scheduler, "_process_batch_responses", "scheduler.outputs"),
        (scheduler, "_cleanup_finished", "scheduler.cleanup"),
        (generator, "next", "generator.next"),
        (prefill, "prompt", "generator.prefill"),
        (prefill, "generate", "generator.start_decode"),
        (decode, "next", "generator.decode"),
        (decode, "_step", "generator.forward"),
        (engine, "add_request", "engine.add_request"),
        (engine, "_cleanup_request", "engine.cleanup"),
        (collector, "put", "collector.put"),
        (collector, "get", "collector.get"),
    ]
    for name in ("eval", "async_eval", "clear_cache", "synchronize"):
        if callable(getattr(modules["mlx.core"], name, None)):
            targets.append((modules["mlx.core"], name, f"mlx.{name}"))

    sources = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        if name.startswith("vllm_mlx.") and not path.is_relative_to(
            item.config.rootpath.resolve()
        ):
            raise pytest.UsageError(f"Profile imported a different source tree: {name}")
        sources[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    test_path = item.path.resolve()
    with profile_calls(targets) as trace:
        yield
    item.stash[_TRACE] = {
        "nodeid": item.nodeid,
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "sources": sources,
        "calls": trace.calls,
        "installed_hooks": [name for _, _, name in targets],
        "accepted": False,
        "scope": "Inclusive timings with observer overhead; no synchronization, warmup or acceptance change",
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if call.when != "call" or _TRACE not in item.stash:
        return
    trace = item.stash[_TRACE]
    trace["outcome"] = outcome.get_result().outcome
    output = Path(os.environ["VLLM_MLX_THROUGHPUT_PROFILE"])
    traces = json.loads(output.read_text()) if output.exists() else []
    traces.append(trace)
    output.write_text(json.dumps(traces, indent=2) + "\n")
