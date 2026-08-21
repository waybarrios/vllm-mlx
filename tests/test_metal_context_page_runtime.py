# SPDX-License-Identifier: Apache-2.0
"""Lifecycle and ownership tests for the phase-two page runtime."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import types

import numpy as np
import pytest

from vllm_mlx.attention_backend import (
    AttentionGeometry,
    ContextBackend,
    PrefixHandle,
    RequestHandle,
    numpy_paged_decode_attention,
)
from vllm_mlx.metal_context_runtime import (
    MetalContextCapabilityError,
    MetalContextPageRuntime,
    MetalContextRuntimeError,
)


@pytest.fixture
def geometry() -> AttentionGeometry:
    return AttentionGeometry(
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        block_size=16,
    )


def _values(
    tokens: int,
    *,
    kv_heads: int = 2,
    head_dim: int = 128,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = rng.normal(0.0, 0.2, size=(tokens, kv_heads, head_dim)).astype(np.float32)
    values = rng.normal(0.0, 0.2, size=keys.shape).astype(np.float32)
    return keys, values


def _as_bf16(values: np.ndarray) -> np.ndarray:
    return (
        (values.view(np.uint32) >> np.uint32(16)).astype(np.uint16).astype(np.uint32)
        << 16
    ).view(np.float32)


def _runtime(
    geometry: AttentionGeometry, *, max_pages: int = 8
) -> MetalContextPageRuntime:
    return MetalContextPageRuntime(geometry, max_pages=max_pages, execution="numpy")


def test_runtime_implements_protocol_and_preallocates_no_resident_pages(geometry):
    runtime = _runtime(geometry, max_pages=4)

    assert isinstance(runtime, ContextBackend)
    metrics = runtime.metrics()
    assert metrics["resident_pages"] == 0
    assert metrics["free_pages"] == 4
    assert metrics["execution"] == "numpy"
    runtime.shutdown()


def test_allocation_append_decode_and_model_oracle_match(geometry):
    runtime = _runtime(geometry, max_pages=4)
    request = runtime.allocate_request("request-a", max_tokens=32)
    pages = runtime.allocate_pages(request, 2)
    keys, values = _values(17, seed=7)

    runtime.append_kv(request, 0, keys, values)
    assert len(pages) == 2
    table = runtime.page_table(request)
    np.testing.assert_array_equal(table, np.asarray([[0, 1]], dtype=np.int32))

    query = np.random.default_rng(8).normal(0.0, 0.2, size=(4, 128)).astype(np.float32)
    output = runtime.paged_decode_attention(request, 0, query)

    # Reconstruct the same page layout from the runtime's public diagnostic
    # metadata and the known source values.  The runtime's NumPy execution
    # path is required to call the shared foundation oracle, not a second
    # attention implementation.
    key_pages = np.zeros((2, 2, 16, 128), dtype=np.float32)
    value_pages = np.zeros_like(key_pages)
    key_pages[0, :, :16] = np.transpose(_as_bf16(keys[:16]), (1, 0, 2))
    key_pages[1, :, :1] = np.transpose(_as_bf16(keys[16:]), (1, 0, 2))
    value_pages[0, :, :16] = np.transpose(_as_bf16(values[:16]), (1, 0, 2))
    value_pages[1, :, :1] = np.transpose(_as_bf16(values[16:]), (1, 0, 2))
    expected = numpy_paged_decode_attention(
        query[None],
        key_pages,
        value_pages,
        table,
        np.asarray([17], dtype=np.int32),
        block_size=16,
        num_kv_heads=2,
    )[0]
    np.testing.assert_allclose(output, expected, rtol=4e-4, atol=4e-4)
    metrics = runtime.metrics()
    assert metrics["append_tokens"] == 17
    assert metrics["dispatches"] == 1
    assert metrics["oracle_dispatches"] == 1
    assert metrics["native_dispatches"] == 0
    assert metrics["dispatch_failures"] == metrics["native_failures"] == 0
    assert metrics["oracle_failures"] == 0


def test_query_rank_and_zero_length_decode_are_deterministic(geometry):
    runtime = _runtime(geometry, max_pages=1)
    request = runtime.allocate_request("empty", max_tokens=16)
    query = np.ones((4, 128), dtype=np.float32)

    squeezed = runtime.paged_decode_attention(request, 0, query)
    batched = runtime.paged_decode_attention(request, 0, query[None])
    np.testing.assert_array_equal(squeezed, np.zeros((4, 128), dtype=np.float32))
    np.testing.assert_array_equal(batched, np.zeros((1, 4, 128), dtype=np.float32))
    runtime.shutdown()


def test_prefix_attach_fork_and_copy_on_write_preserve_immutable_pages(geometry):
    runtime = _runtime(geometry, max_pages=6)
    base = runtime.allocate_request("base", max_tokens=32)
    runtime.allocate_pages(base, 2)
    keys, values = _values(17, seed=12)
    runtime.append_kv(base, 0, keys, values)
    runtime.append_kv(base, 1, keys, values)
    prefix = runtime.create_prefix(base, token_count=17)
    forked = runtime.fork_prefix(prefix)

    first = runtime.allocate_request("first", max_tokens=32)
    second = runtime.allocate_request("second", max_tokens=32)
    runtime.attach_prefix(first, prefix)
    runtime.attach_prefix(second, forked)
    before = runtime.page_table(first).copy()
    assert runtime.page_refcounts()[:2] == (5, 5)

    new_keys, new_values = _values(1, seed=13)
    runtime.append_kv(first, 0, new_keys, new_values)

    after = runtime.page_table(first)
    np.testing.assert_array_equal(after[0, 0], before[0, 0])
    assert after[0, 1] != before[0, 1]
    np.testing.assert_array_equal(runtime.page_table(second), before)
    np.testing.assert_array_equal(runtime.page_table(base), before)
    assert runtime.metrics()["cow_events"] == 1
    shared_pages = runtime.metrics()["shared_pages"]
    assert isinstance(shared_pages, int) and shared_pages >= 1

    runtime.release(first)
    runtime.release(second)
    runtime.release(base)
    runtime.release_prefix(prefix)
    runtime.release_prefix(forked)
    assert runtime.evict() == 3
    assert runtime.metrics()["free_pages"] == 6
    runtime.shutdown()


def test_prefix_requires_equal_nonzero_population_for_every_layer(geometry):
    runtime = _runtime(geometry, max_pages=3)
    request = runtime.allocate_request("incomplete", max_tokens=32)
    runtime.allocate_pages(request, 2)
    keys, values = _values(1, seed=14)
    runtime.append_kv(request, 0, keys, values)

    with pytest.raises(ValueError, match="equal, non-zero"):
        runtime.create_prefix(request)

    keys2, values2 = _values(2, seed=15)
    runtime.append_kv(request, 1, keys2, values2)
    with pytest.raises(ValueError, match="equal, non-zero"):
        runtime.create_prefix(request)

    runtime.append_kv(request, 0, keys2[1:], values2[1:])
    prefix = runtime.create_prefix(request)
    runtime.release_prefix(prefix)
    runtime.release(request)
    runtime.shutdown()


@pytest.mark.parametrize("execution", ["numpy", "native"])
def test_prefix_token_count_must_be_the_complete_length_in_both_modes(
    geometry, execution
):
    native_module = _native_module() if execution == "native" else None
    runtime = MetalContextPageRuntime(
        geometry,
        max_pages=2,
        execution=execution,
        native_module=native_module,
    )
    request = runtime.allocate_request(f"complete-prefix-{execution}", max_tokens=32)
    runtime.allocate_pages(request, 2)
    keys, values = _values(17, seed=31)
    runtime.append_kv(request, 0, keys, values)
    runtime.append_kv(request, 1, keys, values)

    with pytest.raises(ValueError, match="fully populated"):
        runtime.create_prefix(request, token_count=1)
    prefix = runtime.create_prefix(request, token_count=17)
    runtime.release_prefix(prefix)
    runtime.release(request)
    runtime.shutdown()


@pytest.mark.parametrize("execution", ["numpy", "native"])
def test_page_allocation_honors_request_max_tokens_in_both_modes(geometry, execution):
    native_module = _native_module() if execution == "native" else None
    runtime = MetalContextPageRuntime(
        geometry,
        max_pages=4,
        execution=execution,
        native_module=native_module,
    )
    request = runtime.allocate_request(f"page-capacity-{execution}", max_tokens=1)

    with pytest.raises(ValueError, match="request max_tokens capacity"):
        runtime.allocate_pages(request, 2)
    assert len(runtime.allocate_pages(request, 1)) == 1
    with pytest.raises(ValueError, match="request max_tokens capacity"):
        runtime.allocate_pages(request, 1)

    runtime.release(request)
    runtime.shutdown()


def test_shared_tail_cow_oom_is_atomic_and_does_not_mutate_prefix(geometry):
    runtime = _runtime(geometry, max_pages=1)
    base = runtime.allocate_request("base", max_tokens=16)
    runtime.allocate_pages(base, 1)
    keys, values = _values(1, seed=17)
    runtime.append_kv(base, 0, keys, values)
    runtime.append_kv(base, 1, keys, values)
    prefix = runtime.create_prefix(base, token_count=1)
    branch = runtime.allocate_request("branch", max_tokens=16)
    runtime.attach_prefix(branch, prefix)
    before = runtime.page_table(branch)

    with pytest.raises(MemoryError, match="out of pages"):
        runtime.append_kv(branch, 0, keys, values)

    np.testing.assert_array_equal(runtime.page_table(branch), before)
    np.testing.assert_array_equal(runtime.page_table(base), before)
    assert runtime.metrics()["cow_events"] == 0
    assert runtime.metrics()["oom_events"] == 0
    runtime.shutdown()


def test_append_missing_page_plus_cow_oom_is_fully_transactional(geometry):
    """A failed 17-token append cannot leak a reserved tail page."""

    runtime = _runtime(geometry, max_pages=2)
    base = runtime.allocate_request("transaction-base", max_tokens=32)
    runtime.allocate_pages(base, 1)
    one_key, one_value = _values(1, seed=18)
    runtime.append_kv(base, 0, one_key, one_value)
    runtime.append_kv(base, 1, one_key, one_value)
    prefix = runtime.create_prefix(base)
    branch = runtime.allocate_request("transaction-branch", max_tokens=32)
    runtime.attach_prefix(branch, prefix)

    before_table = runtime.page_table(branch)
    before_refs = runtime.page_refcounts()
    before_metrics = dict(runtime.metrics())
    many_keys, many_values = _values(17, seed=19)

    with pytest.raises(MemoryError, match="out of pages"):
        runtime.append_kv(branch, 0, many_keys, many_values)

    np.testing.assert_array_equal(runtime.page_table(branch), before_table)
    assert runtime.page_refcounts() == before_refs
    assert dict(runtime.metrics()) == before_metrics

    runtime.release(branch)
    runtime.release(base)
    runtime.release_prefix(prefix)
    runtime.shutdown()


def test_eviction_skips_referenced_pages_and_recovers_capacity(geometry):
    runtime = _runtime(geometry, max_pages=2)
    first = runtime.allocate_request("first", max_tokens=16)
    second = runtime.allocate_request("second", max_tokens=16)
    runtime.allocate_pages(first, 1)
    runtime.allocate_pages(second, 1)
    assert runtime.evict() == 0

    runtime.release(first)
    assert runtime.evict(target_pages=1) == 1
    assert runtime.metrics()["free_pages"] == 1

    third = runtime.allocate_request("third", max_tokens=32)
    runtime.allocate_pages(third, 1)
    with pytest.raises(MemoryError, match="out of pages"):
        runtime.allocate_pages(third, 1)
    runtime.release(second)
    runtime.release(third)
    assert runtime.evict() == 2
    runtime.shutdown()


def test_release_and_shutdown_are_idempotent_and_teardown_is_deterministic(geometry):
    runtime = _runtime(geometry, max_pages=1)
    request = runtime.allocate_request("cancelled", max_tokens=16)
    runtime.allocate_pages(request, 1)
    runtime.release(request)
    runtime.release(request)
    assert runtime.metrics()["released_requests"] == 1
    assert runtime.metrics()["release_calls"] == 2

    runtime.shutdown()
    runtime.shutdown()
    assert runtime.metrics()["shutdown"] is True
    assert runtime.metrics()["free_pages"] == 1
    with pytest.raises(MetalContextRuntimeError, match="shut down"):
        runtime.allocate_request("late", max_tokens=16)


def test_oracle_mutable_state_is_serialized_across_concurrent_layers(geometry):
    runtime = _runtime(geometry, max_pages=4)
    request = runtime.allocate_request("concurrent", max_tokens=32)
    keys, values = _values(8, seed=44)

    with ThreadPoolExecutor(max_workers=4) as executor:
        append_futures = [
            executor.submit(runtime.append_kv, request, layer, keys, values)
            for layer in (0, 1)
        ]
        metric_futures = [executor.submit(runtime.metrics) for _ in range(2)]
        for future in append_futures:
            future.result()
        for metric_future in metric_futures:
            metric_future.result()

    assert runtime.page_table(request).shape == (1, 1)
    assert runtime.metrics()["append_tokens"] == 16
    runtime.shutdown()


@pytest.mark.parametrize("execution", ["numpy", "native"])
def test_snapshot_and_restore_fail_closed_until_persistence_package(
    geometry, execution
):
    native_module = _native_module() if execution == "native" else None
    runtime = MetalContextPageRuntime(
        geometry,
        max_pages=1,
        execution=execution,
        native_module=native_module,
    )
    request = runtime.allocate_request(f"persist-{execution}", max_tokens=16)
    runtime.allocate_pages(request, 1)
    keys, values = _values(1)
    runtime.append_kv(request, 0, keys, values)
    runtime.append_kv(request, 1, keys, values)
    prefix = runtime.create_prefix(request, token_count=1)

    with pytest.raises(NotImplementedError, match="snapshots are deferred"):
        runtime.snapshot(prefix, destination="/tmp/not-written")
    with pytest.raises(NotImplementedError, match="restore is deferred"):
        runtime.restore("/tmp/not-read")
    assert runtime.metrics()["snapshot_failures"] == 1
    assert runtime.metrics()["restore_failures"] == 1
    runtime.shutdown()
    assert runtime.metrics()["snapshot_failures"] == 1
    assert runtime.metrics()["restore_failures"] == 1


def test_unsupported_geometry_and_request_metadata_are_rejected():
    with pytest.raises(ValueError, match="head_dim=128"):
        MetalContextPageRuntime(
            AttentionGeometry(1, 2, 1, head_dim=64),
            max_pages=1,
            execution="numpy",
        )

    runtime = MetalContextPageRuntime(
        AttentionGeometry(1, 2, 1), max_pages=1, execution="numpy"
    )
    with pytest.raises(ValueError, match="request_id"):
        runtime.allocate_request("", max_tokens=16)
    with pytest.raises(ValueError, match="max_tokens"):
        runtime.allocate_request("bad", max_tokens=0)
    with pytest.raises(ValueError, match="INT32_MAX"):
        runtime.allocate_request("too-long", max_tokens=2**31)
    request = runtime.allocate_request("ok", max_tokens=16)
    assert runtime.allocate_pages(request, 0) == ()
    with pytest.raises(ValueError, match="already active"):
        runtime.allocate_request("ok", max_tokens=16)
    runtime.shutdown()


def test_numpy_constructor_rejects_native_int32_capacity_overflow():
    geometry = AttentionGeometry(1, 1, 1, block_size=16)
    with pytest.raises(ValueError, match="max_pages.*INT32_MAX"):
        MetalContextPageRuntime(
            geometry,
            max_pages=2**31,
            execution="numpy",
        )
    with pytest.raises(ValueError, match="int32 sequence limit"):
        MetalContextPageRuntime(
            geometry,
            max_pages=1,
            max_blocks_per_request=2**31,
            execution="numpy",
        )


class _FakeNativePageRuntime:
    """Small executable model of the compiled PageRuntime Python ABI."""

    instances: list["_FakeNativePageRuntime"] = []
    # Only the blocking test subclass materializes these synchronization
    # primitives; annotations let the shared instance registry remain typed.
    decode_entered: threading.Event
    decode_release: threading.Event

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple[str, object]] = []
        self.next_request = 100
        self.next_prefix = 200
        self.layer_lengths: dict[int, list[int]] = {}
        self.request_ids: dict[int, str] = {}
        self.max_tokens: dict[int, int] = {}
        self.pages: dict[int, tuple[int, ...]] = {}
        self.prefixes: dict[int, int] = {}
        self.attachments: dict[int, int] = {}
        self.released: set[int] = set()
        self.failures: dict[str, int] = {}
        self.post_failures: dict[str, int] = {}
        self.shutdown_called = False
        self.shutdown_state = False
        self.dispatches = 0
        self.dispatch_failures = 0
        self.counters: dict[str, int] = {
            "page_allocations": 0,
            "request_allocations": 0,
            "prefix_allocations": 0,
            "prefix_attaches": 0,
            "prefix_forks": 0,
            "prefix_tokens_attached": 0,
            "cow_events": 0,
            "evictions": 0,
            "pages_freed": 0,
            "releases": 0,
            "release_calls": 0,
            "released_requests": 0,
            "cancellations": 0,
            "append_tokens": 0,
            "oom_events": 0,
            "snapshot_failures": 0,
            "restore_failures": 0,
        }
        self.__class__.instances.append(self)

    def fail_next(self, operation: str) -> None:
        self.failures[operation] = self.failures.get(operation, 0) + 1

    def fail_after_mutation(self, operation: str) -> None:
        """Inject a fault after mutation while honoring native atomicity."""

        self.post_failures[operation] = self.post_failures.get(operation, 0) + 1

    def _maybe_fail(self, operation: str) -> None:
        remaining = self.failures.get(operation, 0)
        if remaining:
            self.failures[operation] = remaining - 1
            raise MemoryError(f"injected {operation} failure")

    def _maybe_fail_after(self, operation: str) -> None:
        remaining = self.post_failures.get(operation, 0)
        if remaining:
            self.post_failures[operation] = remaining - 1
            raise MemoryError(f"injected post-mutation {operation} failure")

    def allocate_request(self, *, request_id, max_tokens):
        self._maybe_fail("allocate_request")
        self.calls.append(("allocate_request", (request_id, max_tokens)))
        handle = self.next_request
        self.next_request += 1
        if request_id in self.request_ids.values():
            raise ValueError("request_id is already active")
        self.request_ids[handle] = request_id
        self.max_tokens[handle] = max_tokens
        self.layer_lengths[handle] = [0] * self.kwargs["num_layers"]
        self.pages[handle] = ()
        self.counters["request_allocations"] += 1
        try:
            self._maybe_fail_after("allocate_request")
        except BaseException:
            self.request_ids.pop(handle, None)
            self.max_tokens.pop(handle, None)
            self.layer_lengths.pop(handle, None)
            self.pages.pop(handle, None)
            self.next_request -= 1
            self.counters["request_allocations"] -= 1
            raise
        return handle

    def allocate_pages(self, *, request, count):
        self._maybe_fail("allocate_pages")
        self.calls.append(("allocate_pages", (request, count)))
        old = self.pages[request]
        capacity = (
            self.max_tokens[request] + self.kwargs["block_size"] - 1
        ) // self.kwargs["block_size"]
        if len(old) + count > capacity:
            raise ValueError("requested pages exceed the request max_tokens capacity")
        pages = old + tuple(range(len(old), len(old) + count))
        self.pages[request] = pages
        self.counters["page_allocations"] += count
        try:
            self._maybe_fail_after("allocate_pages")
        except BaseException:
            self.pages[request] = old
            self.counters["page_allocations"] -= count
            raise
        return pages

    def append_kv(self, *, request, layer, keys, values):
        self._maybe_fail("append_kv")
        self.calls.append(("append_kv", (request, layer, keys.copy(), values.copy())))
        token_count = int(keys.shape[0])
        self.layer_lengths[request][layer] += token_count
        self.counters["append_tokens"] += token_count

    def sequence_length(self, request):
        self._maybe_fail("sequence_length")
        self.calls.append(("sequence_length", request))
        lengths = self.layer_lengths[request]
        if (
            not lengths
            or not lengths[0]
            or any(length != lengths[0] for length in lengths)
        ):
            raise ValueError(
                "prefix creation requires all layers to have equal, non-zero lengths"
            )
        return lengths[0]

    def create_prefix(self, request):
        self._maybe_fail("create_prefix")
        self.calls.append(("create_prefix", request))
        prefix = self.next_prefix
        self.next_prefix += 1
        self.prefixes[prefix] = request
        self.counters["prefix_allocations"] += 1
        try:
            self._maybe_fail_after("create_prefix")
        except BaseException:
            self.prefixes.pop(prefix, None)
            self.next_prefix -= 1
            self.counters["prefix_allocations"] -= 1
            raise
        return prefix

    def fork_prefix(self, prefix):
        self._maybe_fail("fork_prefix")
        self.calls.append(("fork_prefix", prefix))
        forked = self.next_prefix
        self.next_prefix += 1
        self.prefixes[forked] = self.prefixes[prefix]
        self.counters["prefix_allocations"] += 1
        self.counters["prefix_forks"] += 1
        try:
            self._maybe_fail_after("fork_prefix")
        except BaseException:
            self.prefixes.pop(forked, None)
            self.next_prefix -= 1
            self.counters["prefix_allocations"] -= 1
            self.counters["prefix_forks"] -= 1
            raise
        return forked

    def attach_prefix(self, request, prefix):
        self._maybe_fail("attach_prefix")
        self.calls.append(("attach_prefix", (request, prefix)))
        source_request = self.prefixes[prefix]
        old_pages = self.pages[request]
        old_lengths = list(self.layer_lengths[request])
        old_attachment = self.attachments.get(request)
        self.pages[request] = self.pages[source_request]
        self.layer_lengths[request] = list(self.layer_lengths[source_request])
        self.attachments[request] = prefix
        self.counters["prefix_attaches"] += 1
        self.counters["prefix_tokens_attached"] += self.layer_lengths[request][0]
        try:
            self._maybe_fail_after("attach_prefix")
        except BaseException:
            attached_tokens = self.layer_lengths[request][0]
            self.pages[request] = old_pages
            self.layer_lengths[request] = old_lengths
            if old_attachment is None:
                self.attachments.pop(request, None)
            else:
                self.attachments[request] = old_attachment
            self.counters["prefix_attaches"] -= 1
            self.counters["prefix_tokens_attached"] -= attached_tokens
            raise

    def release_prefix(self, prefix):
        self._maybe_fail("release_prefix")
        self.calls.append(("release_prefix", prefix))
        prior = self.prefixes.pop(prefix, None)
        if prior is None:
            return
        try:
            self._maybe_fail_after("release_prefix")
        except BaseException:
            self.prefixes[prefix] = prior
            raise

    def release(self, *, request):
        self._maybe_fail("release")
        self.calls.append(("release", request))
        if request in self.released:
            return
        old_pages = self.pages[request]
        old_lengths = list(self.layer_lengths[request])
        old_attachment = self.attachments.get(request)
        self.counters["release_calls"] += 1
        self.counters["releases"] += 1
        self.counters["released_requests"] += 1
        self.released.add(request)
        self.pages[request] = ()
        self.layer_lengths[request] = [0] * self.kwargs["num_layers"]
        self.attachments.pop(request, None)
        try:
            self._maybe_fail_after("release")
        except BaseException:
            self.released.discard(request)
            self.pages[request] = old_pages
            self.layer_lengths[request] = old_lengths
            if old_attachment is not None:
                self.attachments[request] = old_attachment
            self.counters["release_calls"] -= 1
            self.counters["releases"] -= 1
            self.counters["released_requests"] -= 1
            raise

    def cancel(self, *, request):
        self._maybe_fail("cancel")
        self.calls.append(("cancel", request))
        if request in self.released:
            return
        old_pages = self.pages[request]
        old_lengths = list(self.layer_lengths[request])
        old_attachment = self.attachments.get(request)
        self.counters["release_calls"] += 1
        self.counters["cancellations"] += 1
        self.counters["released_requests"] += 1
        self.released.add(request)
        self.pages[request] = ()
        self.layer_lengths[request] = [0] * self.kwargs["num_layers"]
        self.attachments.pop(request, None)
        try:
            self._maybe_fail_after("cancel")
        except BaseException:
            self.released.discard(request)
            self.pages[request] = old_pages
            self.layer_lengths[request] = old_lengths
            if old_attachment is not None:
                self.attachments[request] = old_attachment
            self.counters["release_calls"] -= 1
            self.counters["cancellations"] -= 1
            self.counters["released_requests"] -= 1
            raise

    def evict(self, **kwargs):
        self.calls.append(("evict", kwargs))
        return 0

    def snapshot(self, *, prefix, destination):
        self._maybe_fail("snapshot")
        self.counters["snapshot_failures"] += 1
        raise NotImplementedError("snapshots are deferred")

    def restore(self, *, source):
        self._maybe_fail("restore")
        self.counters["restore_failures"] += 1
        raise NotImplementedError("restore is deferred")

    def page_table(self, request, layer=0):
        self._maybe_fail("page_table")
        self.calls.append(("page_table", (request, layer)))
        token_count = self.layer_lengths[request][layer]
        blocks = (token_count + self.kwargs["block_size"] - 1) // self.kwargs[
            "block_size"
        ]
        return self.pages[request][:blocks]

    def paged_decode(self, *, request, layer, query, scale):
        self.calls.append(("paged_decode", (request, layer, query.copy(), scale)))
        self.dispatches += 1
        return np.zeros((4, 128), dtype=np.float32).tobytes()

    def metrics(self):
        if self.shutdown_state:
            resident = 0
            requests = 0
            prefixes = 0
            free_pages = self.kwargs["max_pages"]
        else:
            live_requests = [
                handle for handle in self.layer_lengths if handle not in self.released
            ]
            resident = len(
                {page for handle in live_requests for page in self.pages[handle]}
            )
            requests = len(live_requests)
            prefixes = len(self.prefixes)
            free_pages = self.kwargs["max_pages"] - resident
        return {
            "resident_pages": resident,
            "referenced_pages": resident,
            "shared_pages": 0,
            "evictions": self.counters["evictions"],
            "pages_freed": self.counters["pages_freed"],
            "free_pages": free_pages,
            "requests": requests,
            "prefixes": prefixes,
            "page_allocations": self.counters["page_allocations"],
            "request_allocations": self.counters["request_allocations"],
            "prefix_allocations": self.counters["prefix_allocations"],
            "prefix_attaches": self.counters["prefix_attaches"],
            "prefix_forks": self.counters["prefix_forks"],
            "prefix_tokens_attached": self.counters["prefix_tokens_attached"],
            "cow_events": self.counters["cow_events"],
            "releases": self.counters["releases"],
            "release_calls": self.counters["release_calls"],
            "released_requests": self.counters["released_requests"],
            "cancellations": self.counters["cancellations"],
            "append_tokens": self.counters["append_tokens"],
            "oom_events": self.counters["oom_events"],
            "dispatches": self.dispatches,
            "dispatch_failures": self.dispatch_failures,
            "snapshot_failures": self.counters["snapshot_failures"],
            "restore_failures": self.counters["restore_failures"],
            "max_pages": self.kwargs["max_pages"],
            "max_blocks_per_request": self.kwargs["max_blocks_per_request"],
            "block_size": self.kwargs["block_size"],
            "shutdown": self.shutdown_state,
        }

    def shutdown(self):
        self.shutdown_called = True
        self.shutdown_state = True


def _native_module(page_runtime_type=_FakeNativePageRuntime):
    return types.SimpleNamespace(
        capabilities=lambda: {
            "available": True,
            "compiled": True,
            "metal_device": True,
            "abi_version": 1,
            "serving_ready": False,
        },
        PageRuntime=page_runtime_type,
    )


def _native_state(native: _FakeNativePageRuntime) -> dict[str, object]:
    return {
        "request_ids": dict(native.request_ids),
        "max_tokens": dict(native.max_tokens),
        "next_request": native.next_request,
        "next_prefix": native.next_prefix,
        "layer_lengths": {
            handle: list(lengths) for handle, lengths in native.layer_lengths.items()
        },
        "pages": dict(native.pages),
        "prefixes": dict(native.prefixes),
        "attachments": dict(native.attachments),
        "released": set(native.released),
        "counters": dict(native.counters),
        "shutdown": native.shutdown_state,
    }


def _adapter_native_state(runtime: MetalContextPageRuntime) -> dict[str, object]:
    # Native lifecycle state belongs to the compiled runtime.  The adapter's
    # only mutable native-side field is its local shutdown guard.
    return {"shutdown": runtime._shutdown}


def _native_prefix_fixture(
    geometry: AttentionGeometry,
) -> tuple[
    MetalContextPageRuntime,
    _FakeNativePageRuntime,
    RequestHandle,
    PrefixHandle,
]:
    runtime = MetalContextPageRuntime(
        geometry,
        max_pages=2,
        execution="native",
        native_module=_native_module(),
    )
    native = _FakeNativePageRuntime.instances[-1]
    request = runtime.allocate_request("failure-base", max_tokens=16)
    runtime.allocate_pages(request, 1)
    keys, values = _values(1)
    runtime.append_kv(request, 0, keys, values)
    runtime.append_kv(request, 1, keys, values)
    prefix = runtime.create_prefix(request)
    return runtime, native, request, prefix


def test_native_page_table_truncates_reserved_tail_and_preserves_prefix_lengths(
    geometry,
):
    runtime = MetalContextPageRuntime(
        geometry,
        max_pages=2,
        execution="native",
        native_module=_native_module(),
    )
    native = _FakeNativePageRuntime.instances[-1]
    request = runtime.allocate_request("layer-table", max_tokens=32)
    runtime.allocate_pages(request, 2)
    keys, values = _values(1, seed=51)
    runtime.append_kv(request, 0, keys, values)

    # Native allocation reserves two physical pages, but only layer zero has
    # a populated token.  The adapter must use its authoritative per-layer
    # lengths rather than exposing the native allocation watermark.
    np.testing.assert_array_equal(
        runtime.page_table(request, layer=0), np.asarray([[0]], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        runtime.page_table(request, layer=1), np.empty((1, 0), dtype=np.int32)
    )

    runtime.append_kv(request, 1, keys, values)
    prefix = runtime.create_prefix(request)
    branch = runtime.allocate_request("layer-table-branch", max_tokens=32)
    runtime.attach_prefix(branch, prefix)
    for layer in range(geometry.num_layers):
        np.testing.assert_array_equal(
            runtime.page_table(branch, layer=layer),
            np.asarray([[0]], dtype=np.int32),
        )
    assert native.layer_lengths[int(branch)] == [1, 1]
    runtime.release(branch)
    runtime.release(request)
    runtime.release_prefix(prefix)
    runtime.shutdown()


@pytest.mark.parametrize(
    "operation",
    [
        "allocate_request",
        "allocate_pages",
        "create_prefix",
        "fork_prefix",
        "attach_prefix",
        "release_prefix",
        "release",
        "cancel",
    ],
)
def test_native_memory_errors_leave_adapter_and_native_ownership_converged(
    geometry, operation
):
    if operation == "allocate_request":
        runtime = MetalContextPageRuntime(
            geometry,
            max_pages=2,
            execution="native",
            native_module=_native_module(),
        )
        native = _FakeNativePageRuntime.instances[-1]
        before_native = _native_state(native)
        before_adapter = _adapter_native_state(runtime)
        native.fail_after_mutation(operation)
        with pytest.raises(MemoryError, match="injected"):
            runtime.allocate_request("failure-request", max_tokens=16)
    elif operation == "allocate_pages":
        runtime = MetalContextPageRuntime(
            geometry,
            max_pages=2,
            execution="native",
            native_module=_native_module(),
        )
        native = _FakeNativePageRuntime.instances[-1]
        request = runtime.allocate_request("failure-pages", max_tokens=16)
        before_native = _native_state(native)
        before_adapter = _adapter_native_state(runtime)
        native.fail_after_mutation(operation)
        with pytest.raises(MemoryError, match="injected"):
            runtime.allocate_pages(request, 1)
    elif operation in {"create_prefix", "fork_prefix", "release_prefix"}:
        runtime, native, _request, prefix = _native_prefix_fixture(geometry)
        if operation == "fork_prefix":
            before_native = _native_state(native)
            before_adapter = _adapter_native_state(runtime)
            native.fail_after_mutation(operation)
            with pytest.raises(MemoryError, match="injected"):
                runtime.fork_prefix(prefix)
        else:
            before_native = _native_state(native)
            before_adapter = _adapter_native_state(runtime)
            native.fail_after_mutation(operation)
            with pytest.raises(MemoryError, match="injected"):
                (
                    runtime.release_prefix(prefix)
                    if operation == "release_prefix"
                    else runtime.create_prefix(_request)
                )
    else:
        runtime, native, request, prefix = _native_prefix_fixture(geometry)
        if operation == "attach_prefix":
            branch = runtime.allocate_request("failure-branch", max_tokens=16)
            before_native = _native_state(native)
            before_adapter = _adapter_native_state(runtime)
            native.fail_after_mutation(operation)
            with pytest.raises(MemoryError, match="injected"):
                runtime.attach_prefix(branch, prefix)
        else:
            before_native = _native_state(native)
            before_adapter = _adapter_native_state(runtime)
            native.fail_after_mutation(operation)
            with pytest.raises(MemoryError, match="injected"):
                (
                    runtime.cancel(request)
                    if operation == "cancel"
                    else runtime.release(request)
                )

    assert _native_state(native) == before_native
    after_adapter = _adapter_native_state(runtime)
    assert after_adapter == before_adapter
    runtime.shutdown()


def test_native_execution_requires_capabilities_and_never_silently_falls_back(
    geometry,
):
    unavailable = types.SimpleNamespace(
        capabilities=lambda: {
            "available": False,
            "compiled": True,
            "metal_device": False,
            "abi_version": 1,
            "reason": "test GPU unavailable",
        }
    )
    with pytest.raises(MetalContextCapabilityError, match="test GPU unavailable"):
        MetalContextPageRuntime(
            geometry, max_pages=2, execution="native", native_module=unavailable
        )

    runtime = MetalContextPageRuntime(
        geometry, max_pages=2, execution="native", native_module=_native_module()
    )
    native = _FakeNativePageRuntime.instances[-1]
    request = runtime.allocate_request("native", max_tokens=16)
    with pytest.raises(ValueError, match="already active"):
        runtime.allocate_request("native", max_tokens=16)
    runtime.allocate_pages(request, 1)
    keys, values = _values(1)
    runtime.append_kv(request, 0, keys, values)
    runtime.append_kv(request, 1, keys, values)
    prefix = runtime.create_prefix(request)
    forked = runtime.fork_prefix(prefix)
    runtime.attach_prefix(runtime.allocate_request("branch", max_tokens=16), prefix)
    output = runtime.paged_decode_attention(request, 0, np.ones((4, 128), np.float32))
    assert output.shape == (4, 128)
    assert not hasattr(runtime, "_keys")
    assert not hasattr(runtime, "_values")
    for shadow_name in (
        "_native_handle_request_ids",
        "_native_request_max_tokens",
        "_native_request_page_counts",
        "_native_request_layer_lengths",
        "_native_released_requests",
        "_native_released_prefixes",
        "_native_prefix_handles",
        "_native_prefix_token_counts",
        "_native_metrics_cache",
        "_requests",
        "_prefixes",
        "_released_requests",
        "_released_prefixes",
        "_request_ids",
        "_next_request",
        "_next_prefix",
        "_clock",
        "_counters",
    ):
        assert not hasattr(runtime, shadow_name)
    call_names = [name for name, _ in native.calls]
    assert call_names == [
        "allocate_request",
        "allocate_request",
        "allocate_pages",
        "append_kv",
        "append_kv",
        "sequence_length",
        "create_prefix",
        "fork_prefix",
        "allocate_request",
        "attach_prefix",
        "paged_decode",
    ]
    runtime.release(request)
    runtime.release(request)
    cancelled = runtime.allocate_request("cancelled", max_tokens=16)
    runtime.cancel(cancelled)
    runtime.cancel(cancelled)
    runtime.release_prefix(prefix)
    runtime.release_prefix(prefix)
    metrics = runtime.metrics()
    assert metrics["resident_pages"] == 1
    assert metrics["dispatches"] == 1
    assert metrics["native_dispatches"] == 1
    assert metrics["oracle_dispatches"] == 0
    assert metrics["dispatch_failures"] == metrics["native_failures"] == 0
    assert metrics["pages_allocated"] == 1
    assert metrics["append_tokens"] == 2
    assert metrics["prefix_attaches"] == 1
    assert metrics["prefix_forks"] == 1
    assert metrics["prefix_tokens_attached"] == 1
    assert metrics["release_calls"] == 2
    assert metrics["released_requests"] == 2
    assert metrics["cancellations"] == 1
    assert metrics["oom_events"] == 0
    runtime.shutdown()
    assert native.shutdown_called is True
    shutdown_metrics = runtime.metrics()
    assert shutdown_metrics["resident_pages"] == 0
    assert shutdown_metrics["referenced_pages"] == 0
    assert shutdown_metrics["requests"] == 0
    assert shutdown_metrics["prefixes"] == 0
    assert shutdown_metrics["free_pages"] == 2
    assert shutdown_metrics["pages_allocated"] == 1
    assert shutdown_metrics["append_tokens"] == 2
    assert shutdown_metrics["prefix_attaches"] == 1
    assert shutdown_metrics["prefix_forks"] == 1
    assert shutdown_metrics["release_calls"] == 2
    assert shutdown_metrics["released_requests"] == 2
    assert shutdown_metrics["cancellations"] == 1


def test_native_decodes_can_overlap_without_python_lifecycle_lock(geometry):
    class OverlapPageRuntime(_FakeNativePageRuntime):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.decode_barrier = threading.Barrier(2)

        def paged_decode(self, **kwargs):
            self.decode_barrier.wait(timeout=2)
            return super().paged_decode(**kwargs)

    runtime = MetalContextPageRuntime(
        geometry,
        max_pages=2,
        execution="native",
        native_module=_native_module(OverlapPageRuntime),
    )
    first = runtime.allocate_request("overlap-first", max_tokens=16)
    second = runtime.allocate_request("overlap-second", max_tokens=16)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                runtime.paged_decode_attention,
                request,
                0,
                np.ones((4, 128), dtype=np.float32),
            )
            for request in (first, second)
        ]
        outputs = [future.result(timeout=3) for future in futures]

    assert all(output.shape == (4, 128) for output in outputs)
    assert runtime.metrics()["dispatches"] == 2
    runtime.shutdown()


def test_native_decode_does_not_block_lifecycle_metrics(geometry):
    class BlockingPageRuntime(_FakeNativePageRuntime):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.decode_entered = threading.Event()
            self.decode_release = threading.Event()

        def paged_decode(self, **kwargs):
            self.decode_entered.set()
            if not self.decode_release.wait(timeout=3):
                raise RuntimeError("test decode release timed out")
            return super().paged_decode(**kwargs)

    runtime = MetalContextPageRuntime(
        geometry,
        max_pages=1,
        execution="native",
        native_module=_native_module(BlockingPageRuntime),
    )
    native = _FakeNativePageRuntime.instances[-1]
    request = runtime.allocate_request("lifecycle-during-decode", max_tokens=16)
    with ThreadPoolExecutor(max_workers=2) as executor:
        decode = executor.submit(
            runtime.paged_decode_attention,
            request,
            0,
            np.ones((4, 128), dtype=np.float32),
        )
        assert native.decode_entered.wait(timeout=2)
        metrics = executor.submit(runtime.metrics).result(timeout=1)
        assert metrics["requests"] == 1
        native.decode_release.set()
        assert decode.result(timeout=2).shape == (4, 128)
    runtime.shutdown()


def test_native_dispatch_error_is_reported_without_oracle_fallback(geometry):
    class FailingPageRuntime(_FakeNativePageRuntime):
        def paged_decode(self, **kwargs):
            self.dispatch_failures += 1
            raise RuntimeError("kernel command failed")

    runtime = MetalContextPageRuntime(
        geometry,
        max_pages=1,
        execution="native",
        native_module=_native_module(FailingPageRuntime),
    )
    request = runtime.allocate_request("native-error", max_tokens=16)
    runtime.allocate_pages(request, 1)
    keys, values = _values(1)
    runtime.append_kv(request, 0, keys, values)
    with pytest.raises(RuntimeError, match="kernel command failed"):
        runtime.paged_decode_attention(request, 0, np.ones((4, 128), np.float32))
    metrics = runtime.metrics()
    assert metrics["dispatches"] == 0
    assert metrics["native_dispatches"] == 0
    assert metrics["dispatch_failures"] == 1
    assert metrics["native_failures"] == 1
    assert metrics["oracle_dispatches"] == 0
    runtime.shutdown()


def test_native_capacity_failures_report_oom_events(geometry):
    class OOMPageRuntime(_FakeNativePageRuntime):
        def allocate_pages(self, **kwargs):
            self.counters["oom_events"] += 1
            raise MemoryError("capacity exhausted")

    runtime = MetalContextPageRuntime(
        geometry,
        max_pages=1,
        execution="native",
        native_module=_native_module(OOMPageRuntime),
    )
    request = runtime.allocate_request("native-oom", max_tokens=16)
    with pytest.raises(MemoryError, match="capacity exhausted"):
        runtime.allocate_pages(request, 1)
    assert runtime.metrics()["oom_events"] == 1
    assert runtime.metrics()["pages_allocated"] == 0
    runtime.shutdown()


def test_native_metrics_preserve_authoritative_snapshot_restore_counters(geometry):
    class MetricsPageRuntime(_FakeNativePageRuntime):
        def metrics(self):
            result = super().metrics()
            result.update({"snapshot_failures": 7, "restore_failures": 9})
            return result

    runtime = MetalContextPageRuntime(
        geometry,
        max_pages=1,
        execution="native",
        native_module=_native_module(MetricsPageRuntime),
    )
    assert runtime.metrics()["snapshot_failures"] == 7
    assert runtime.metrics()["restore_failures"] == 9
    runtime.shutdown()


def test_actual_compiled_page_runtime_is_used_when_available(geometry):
    native = pytest.importorskip("vllm_mlx._metal_context")
    capabilities = native.capabilities()
    if not capabilities.get("available") or not hasattr(native, "PageRuntime"):
        pytest.skip("compiled native PageRuntime is unavailable")

    runtime = MetalContextPageRuntime(
        geometry,
        max_pages=2,
        max_requests=2,
        execution="native",
        native_module=native,
    )
    assert isinstance(runtime._native_runtime, native.PageRuntime)
    request = runtime.allocate_request("compiled", max_tokens=16)
    runtime.allocate_pages(request, 1)
    keys, values = _values(1, seed=55)
    runtime.append_kv(request, 0, keys, values)
    runtime.append_kv(request, 1, keys, values)
    prefix = runtime.create_prefix(request)
    runtime.release(request)
    runtime.release(request)
    runtime.release_prefix(prefix)
    runtime.shutdown()
    runtime.shutdown()


def test_randomized_page_layouts_match_reference_oracle(geometry):
    """Exercise varied sequence lengths and GQA against the shared oracle."""

    rng = np.random.default_rng(99)
    for iteration in range(12):
        runtime = _runtime(geometry, max_pages=5)
        request = runtime.allocate_request(str(iteration), max_tokens=64)
        runtime.allocate_pages(request, 4)
        length = int(rng.integers(1, 65))
        keys, values = _values(length, seed=100 + iteration)
        runtime.append_kv(request, 0, keys, values)
        query = rng.normal(0.0, 0.2, size=(4, 128)).astype(np.float32)
        output = runtime.paged_decode_attention(request, 0, query)

        # Use a packed logical page view as a second independent model of the
        # append layout; the runtime page IDs may be physical/non-contiguous in
        # later allocation strategies, while the page-table contract remains
        # the only source of lookup order.
        pages = runtime.page_table(request)
        key_storage = np.zeros((5, 2, 16, 128), dtype=np.float32)
        value_storage = np.zeros_like(key_storage)
        for logical, page_id in enumerate(pages[0]):
            start = logical * 16
            end = min(start + 16, length)
            key_storage[page_id, :, : end - start] = np.transpose(
                _as_bf16(keys[start:end]), (1, 0, 2)
            )
            value_storage[page_id, :, : end - start] = np.transpose(
                _as_bf16(values[start:end]), (1, 0, 2)
            )
        expected = numpy_paged_decode_attention(
            query[None],
            key_storage,
            value_storage,
            pages,
            np.asarray([length], dtype=np.int32),
            block_size=16,
            num_kv_heads=2,
        )[0]
        np.testing.assert_allclose(output, expected, rtol=4e-4, atol=4e-4)
        runtime.shutdown()
