# SPDX-License-Identifier: Apache-2.0
"""Focused generator request-ownership regressions."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _assert_same_error(action, error, *args):
    with pytest.raises(type(error)) as exc_info:
        action(*args)
    assert exc_info.value is error


def _make_insert_generator():
    from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.uid_counter = 0
    generator.unprocessed_requests = []
    return generator


def _make_chunked_generator():
    from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchStats

    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator._next = lambda: []
    generator._pending_error_responses = []
    generator._aborted_request_ids = set()
    generator._aborted_request_uids = set()
    generator._prefill_progress = {}
    generator.active_batch = None
    generator.unprocessed_requests = []
    generator.prefix_cache = None
    generator.language_model = MagicMock()
    generator._stats = MLLMBatchStats()
    generator._think_suffix_len = 0
    generator.max_kv_size = 0
    return generator


class TestMLLMPendingRemovalOwnership:
    def test_process_pending_removals_requeues_failed_removal(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        removal_error = RuntimeError("remove failed")

        class FailingActiveBatch:
            uids = [7, 8]
            requests = [
                SimpleNamespace(uid=7, request_id="same"),
                SimpleNamespace(uid=8, request_id="other"),
            ]

            def __init__(self):
                self.fail = True

            def filter(self, keep_idx):
                if self.fail:
                    self.fail = False
                    raise removal_error
                self.uids = [self.uids[index] for index in keep_idx]
                self.requests = [self.requests[index] for index in keep_idx]

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator._pending_removal_lock = threading.Lock()
        generator._pending_removal_uids = {7}
        generator._aborted_request_ids = {"same"}
        generator._aborted_request_uids = {7}
        generator._prefill_progress = {"same": (2, 5)}
        generator.unprocessed_requests = []
        generator.active_batch = FailingActiveBatch()

        _assert_same_error(generator.process_pending_removals, removal_error)
        assert generator._pending_removal_uids == {7}
        assert generator._aborted_request_ids == {"same"}
        assert generator._aborted_request_uids == {7}
        assert generator._prefill_progress == {"same": (2, 5)}

        generator.process_pending_removals()
        assert generator._pending_removal_uids == set()
        assert generator.active_batch.uids == [8]
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()
        assert generator._prefill_progress == {}

    def test_generator_removal_retires_abort_marker_for_exact_uid(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        class FakeActiveBatch:
            def __init__(self):
                self.uids = [92, 95]
                self.requests = [
                    SimpleNamespace(uid=92, request_id="same"),
                    SimpleNamespace(uid=95, request_id="other"),
                ]

            def filter(self, keep_idx):
                self.uids = [self.uids[index] for index in keep_idx]
                self.requests = [self.requests[index] for index in keep_idx]

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.active_batch = FakeActiveBatch()
        generator.unprocessed_requests = [SimpleNamespace(uid=91, request_id="same")]
        generator._aborted_request_ids = {"same", "other"}
        generator._aborted_request_uids = {91, 92, 93}
        generator._prefill_progress = {
            "same": (2, 4),
            "other": (1, 3),
        }

        generator.remove([91, 92])
        assert generator.unprocessed_requests == []
        assert generator.active_batch.uids == [95]
        assert [request.request_id for request in generator.active_batch.requests] == [
            "other"
        ]
        assert generator._aborted_request_ids == {"other"}
        assert generator._aborted_request_uids == {93}
        assert generator._prefill_progress == {"other": (1, 3)}

        generator.abort_prefill("same", 93)
        replacement = SimpleNamespace(uid=94, request_id="same")
        retired = SimpleNamespace(uid=93, request_id="same")
        assert generator._consume_prefill_abort(replacement) is False
        assert generator._consume_prefill_abort(retired) is True


class TestMLLMGeneratorInsertOwnership:
    def test_generator_insert_metadata_failure_is_atomic(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        class FailingMetadataRequest:
            uid = 33
            request_id = "bad"
            videos = audio = None

            @property
            def images(self):
                raise RuntimeError("metadata failed")

        generator = _make_insert_generator()
        generator.uid_counter = 8
        existing = MLLMBatchRequest(uid=7, request_id="old", prompt="old")
        generator.unprocessed_requests = [existing]
        original_queue = generator.unprocessed_requests
        first = MLLMBatchRequest(uid=21, request_id="first", prompt="first")
        second = MLLMBatchRequest(uid=22, request_id="second", prompt="second")
        bad_request = FailingMetadataRequest()
        requests = [first, second, bad_request]
        original_uids = [request.uid for request in requests]

        with pytest.raises(RuntimeError, match="metadata failed"):
            generator.insert(requests)

        assert generator.uid_counter == 8
        assert [request.uid for request in requests] == original_uids
        assert generator.unprocessed_requests is original_queue
        assert [request.uid for request in original_queue] == [7]

    @pytest.mark.parametrize("mode", ["unprocessed", "active", "partial"])
    def test_generator_insert_rejects_live_uid_overlap_without_mutation(self, mode):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        generator = _make_insert_generator()
        generator.uid_counter = 4
        live = MLLMBatchRequest(uid=4, request_id="live", prompt="live")
        request = MLLMBatchRequest(uid=99, request_id="new", prompt="new")

        if mode == "unprocessed":
            generator.unprocessed_requests = [live]
        elif mode == "active":
            generator.active_batch = SimpleNamespace(uids=[4], requests=[live])
        else:
            generator._partial = {"request": live}

        original_queue = generator.unprocessed_requests
        with pytest.raises(RuntimeError, match="overlaps live"):
            generator.insert([request])

        assert request.uid == 99
        assert generator.uid_counter == 4
        assert generator.unprocessed_requests is original_queue
        assert live.uid == 4


class TestMLLMPrefillAbortPolling:
    def test_run_chunked_text_prefill_aborts_before_model_call(self):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            PrefillAbortedError,
        )

        generator = _make_chunked_generator()
        generator.prefill_step_size = 2
        request = MLLMBatchRequest(uid=1, request_id="same", prompt="long text")
        request.input_ids = mx.array([[1, 2, 3, 4, 5]])
        generator.abort_prefill(request.request_id, request.uid)
        replacement = MLLMBatchRequest(
            uid=2, request_id=request.request_id, prompt="replacement"
        )

        assert generator._consume_prefill_abort(replacement) is False
        with pytest.raises(PrefillAbortedError, match="same"):
            generator._run_chunked_text_prefill(request, cache=[])

        generator.language_model.assert_not_called()
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()

    def test_process_prompts_entry_poll_consumes_uid_abort(self, monkeypatch):
        import vllm_mlx.mllm_batch_generator as batch_module
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        generator = _make_chunked_generator()
        generator.prefill_step_size = 2
        generator.sampler = MagicMock()
        generator.model = SimpleNamespace(
            config=SimpleNamespace(image_token_index=None)
        )
        generator._preprocess_request = lambda request: None
        monkeypatch.setattr(
            "mlx_lm.models.cache.make_prompt_cache",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "mlx_lm.sample_utils.make_logits_processors",
            lambda **kwargs: [],
        )
        monkeypatch.setattr(
            "mlx_lm.sample_utils.make_sampler",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(batch_module.mx, "clear_cache", lambda: None)

        request = MLLMBatchRequest(uid=11, request_id="same", prompt="prompt")
        request.input_ids = mx.array([[1, 2, 3]])
        request.is_text_only = True
        generator.abort_prefill(request.request_id, request.uid)
        replacement = MLLMBatchRequest(
            uid=12, request_id=request.request_id, prompt="replacement"
        )
        assert generator._consume_prefill_abort(replacement) is False

        assert generator._process_prompts([request]) is None
        generator.language_model.assert_not_called()
        assert len(generator._pending_error_responses) == 1
        assert generator._pending_error_responses[0].finish_reason == "abort"
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()


class TestMLLMChunkedOwnership:
    def test_chunked_polling_consumes_uid_abort_and_cleans_partial(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            install_chunked_prefill_mllm,
        )

        generator = _make_chunked_generator()
        install_chunked_prefill_mllm(generator, budget=2)
        monkeypatch.setattr(mx, "clear_cache", lambda: None)

        request = MLLMBatchRequest(uid=3, request_id="same", prompt="long text")
        generator._partial = {
            "request": request,
            "cache": [],
            "remaining_ids": mx.array([[3, 4]]),
            "processed": 2,
            "total": 4,
            "cached_count": 0,
            "chunk_count": 1,
        }
        generator._prefill_progress[request.request_id] = (2, 4)
        generator.abort_prefill(request.request_id, request.uid)

        replacement = MLLMBatchRequest(
            uid=4, request_id=request.request_id, prompt="replacement"
        )
        assert generator._consume_prefill_abort(replacement) is False

        responses = generator._next()

        abort_responses = [
            response for response in responses if response.finish_reason == "abort"
        ]
        assert len(abort_responses) == 1
        assert abort_responses[0].uid == request.uid
        assert abort_responses[0].request_id == request.request_id
        assert generator._partial is None
        assert request.request_id not in generator._prefill_progress
        assert request.uid not in generator._aborted_request_uids

    def test_successful_partial_removal_retires_marker_and_progress(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            install_chunked_prefill_mllm,
        )

        class ActiveBatch:
            def __init__(self):
                self.uids = [4, 9]
                self.requests = [
                    SimpleNamespace(uid=4, request_id="active"),
                    SimpleNamespace(uid=9, request_id="keep"),
                ]

            def filter(self, keep_idx):
                self.uids = [self.uids[index] for index in keep_idx]
                self.requests = [self.requests[index] for index in keep_idx]

        generator = _make_chunked_generator()
        install_chunked_prefill_mllm(generator, budget=2)
        clear_cache = MagicMock()
        monkeypatch.setattr(mx, "clear_cache", clear_cache)
        partial_request = MLLMBatchRequest(
            uid=3, request_id="partial", prompt="long text"
        )
        generator._partial = {"request": partial_request}
        generator.active_batch = ActiveBatch()
        generator._aborted_request_ids = {"partial", "active", "keep"}
        generator._aborted_request_uids = {3, 4, 9}
        generator._prefill_progress = {
            "partial": (2, 5),
            "active": (1, 2),
            "keep": (1, 3),
        }

        generator.remove([3, 4])

        assert generator._partial is None
        assert generator.active_batch.uids == [9]
        assert generator._aborted_request_ids == {"keep"}
        assert generator._aborted_request_uids == {9}
        assert generator._prefill_progress == {"keep": (1, 3)}
        clear_cache.assert_called_once_with()

    def test_failed_partial_removal_preserves_ownership_for_retry(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            install_chunked_prefill_mllm,
        )

        class FailingActiveBatch:
            def __init__(self):
                self.uids = [4, 9]
                self.requests = [
                    SimpleNamespace(uid=4, request_id="active"),
                    SimpleNamespace(uid=9, request_id="keep"),
                ]
                self.fail = True

            def filter(self, keep_idx):
                if self.fail:
                    self.fail = False
                    raise RuntimeError("filter failed")
                self.uids = [self.uids[index] for index in keep_idx]
                self.requests = [self.requests[index] for index in keep_idx]

        generator = _make_chunked_generator()
        install_chunked_prefill_mllm(generator, budget=2)
        clear_cache = MagicMock()
        monkeypatch.setattr(mx, "clear_cache", clear_cache)
        partial_request = MLLMBatchRequest(
            uid=3, request_id="partial", prompt="long text"
        )
        partial = {"request": partial_request}
        generator._partial = partial
        generator.active_batch = FailingActiveBatch()
        generator._aborted_request_ids = {"partial", "active"}
        generator._aborted_request_uids = {3, 4}
        generator._prefill_progress = {
            "partial": (2, 5),
            "active": (1, 2),
        }

        with pytest.raises(RuntimeError, match="filter failed"):
            generator.remove([3, 4])
        assert generator._partial is partial
        assert generator.active_batch.uids == [4, 9]
        assert generator._aborted_request_ids == {"partial", "active"}
        assert generator._aborted_request_uids == {3, 4}
        assert generator._prefill_progress == {
            "partial": (2, 5),
            "active": (1, 2),
        }
        clear_cache.assert_not_called()

        generator.remove([3, 4])
        assert generator._partial is None
        assert generator.active_batch.uids == [9]
        assert generator._aborted_request_ids == set()
        assert generator._aborted_request_uids == set()
        assert generator._prefill_progress == {}
        clear_cache.assert_called_once_with()


class TestMLLMBatchFilterOwnership:
    def test_filter_is_atomic_across_nested_cache_failure(self):
        from copy import deepcopy

        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        class OwnerCache:
            def __init__(self, label, owners, *, fail):
                self.label = label
                self.owners = list(owners)
                self.values = [f"{label}-{owner}" for owner in owners]
                self.state = {
                    "owners": list(owners),
                    "values": list(self.values),
                    "metadata": {"owner_uids": list(owners)},
                }
                self.fail = fail
                self.filter_calls = 0

            def filter(self, keep_idx):
                indices = [int(index) for index in keep_idx.tolist()]
                self.filter_calls += 1
                self.owners = [self.owners[index] for index in indices]
                self.values = [self.values[index] for index in indices]
                self.state["owners"] = [
                    self.state["owners"][index] for index in indices
                ]
                self.state["values"] = [
                    self.state["values"][index] for index in indices
                ]
                self.state["metadata"]["owner_uids"] = [
                    self.state["metadata"]["owner_uids"][index] for index in indices
                ]
                self.state["staged"] = True
                if self.fail:
                    raise RuntimeError("nested cache filter failed")

        class CacheListLike:
            def __init__(self, *caches):
                self.caches = tuple(caches)
                self.container_state = {
                    "owners": (41, 42),
                    "filter_generation": 0,
                }

            def filter(self, keep_idx):
                indices = [int(index) for index in keep_idx.tolist()]
                self.container_state["owners"] = tuple(
                    self.container_state["owners"][index] for index in indices
                )
                self.container_state["filter_generation"] += 1
                for cache in self.caches:
                    cache.filter(keep_idx)

        requests = [
            MLLMBatchRequest(uid=41, request_id="owner-a", prompt="a"),
            MLLMBatchRequest(uid=42, request_id="owner-b", prompt="b"),
        ]
        owner_by_uid = {request.uid: request.request_id for request in requests}
        failing_cache = OwnerCache("failing", [41, 42], fail=True)
        stable_cache = OwnerCache("stable", [41, 42], fail=False)
        cache_group = CacheListLike(failing_cache, stable_cache)
        batch = MLLMBatch(
            uids=[41, 42],
            request_ids=["owner-a", "owner-b"],
            y=mx.array([410, 420]),
            logprobs=["logprob-a", "logprob-b"],
            max_tokens=[128, 256],
            num_tokens=[3, 5],
            cache=[cache_group],
            requests=requests,
            logits_processors=[["processor-a"], ["processor-b"]],
            samplers=["sampler-a", "sampler-b"],
        )

        original_lists = {
            field: getattr(batch, field)
            for field in (
                "uids",
                "request_ids",
                "logprobs",
                "max_tokens",
                "num_tokens",
                "requests",
                "logits_processors",
                "samplers",
            )
        }
        original_list_values = {
            field: deepcopy(value) for field, value in original_lists.items()
        }
        original_y = batch.y
        original_y_value = batch.y.tolist()
        original_cache_list = batch.cache
        original_cache_group = cache_group
        original_cache_children = cache_group.caches
        original_failing_state = failing_cache.state
        original_stable_state = stable_cache.state
        original_cache_snapshot = deepcopy(
            {
                "group": cache_group.container_state,
                "failing": failing_cache.__dict__,
                "stable": stable_cache.__dict__,
            }
        )
        original_owner_inputs = [
            (request.uid, request.request_id) for request in requests
        ]

        with pytest.raises(RuntimeError, match="nested cache filter failed"):
            batch.filter([1])

        for field, original in original_lists.items():
            assert getattr(batch, field) is original
            assert getattr(batch, field) == original_list_values[field]
        assert batch.y is original_y
        assert batch.y.tolist() == original_y_value
        assert batch.cache is original_cache_list
        assert batch.cache[0] is original_cache_group
        assert batch.cache[0].caches is original_cache_children
        assert failing_cache.state is original_failing_state
        assert stable_cache.state is original_stable_state
        assert {
            "group": cache_group.container_state,
            "failing": failing_cache.__dict__,
            "stable": stable_cache.__dict__,
        } == original_cache_snapshot
        assert owner_by_uid == {41: "owner-a", 42: "owner-b"}
        assert [
            (request.uid, request.request_id) for request in requests
        ] == original_owner_inputs

        failing_cache.fail = False
        batch.filter([1])

        assert batch.uids == [42]
        assert batch.request_ids == ["owner-b"]
        assert batch.logprobs == ["logprob-b"]
        assert batch.max_tokens == [256]
        assert batch.num_tokens == [5]
        assert batch.requests == [requests[1]]
        assert batch.logits_processors == [["processor-b"]]
        assert batch.samplers == ["sampler-b"]
        assert batch.y.tolist() == [420]
        assert dict(zip(batch.uids, batch.request_ids)) == {42: "owner-b"}

        published_group = batch.cache[0]
        assert isinstance(published_group.caches, tuple)
        assert published_group.container_state == {
            "owners": (42,),
            "filter_generation": 1,
        }
        for cache in published_group.caches:
            assert cache.owners == [42]
            assert cache.values == [f"{cache.label}-42"]
            assert cache.state == {
                "owners": [42],
                "values": [f"{cache.label}-42"],
                "metadata": {"owner_uids": [42]},
                "staged": True,
            }
            assert cache.filter_calls == 1
