# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible Embeddings API."""

import platform
import sys
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# Unit Tests - Pydantic Models
# =============================================================================


class TestEmbeddingModels:
    """Test embedding request/response Pydantic models."""

    def test_embedding_request_single_string(self):
        """Test EmbeddingRequest with a single input string."""
        from vllm_mlx.api.models import EmbeddingRequest

        req = EmbeddingRequest(model="test-model", input="Hello world")
        assert req.model == "test-model"
        assert req.input == "Hello world"
        assert req.encoding_format == "float"

    def test_embedding_response_serialization(self):
        """Test that EmbeddingResponse serializes to OpenAI-compatible JSON."""
        from vllm_mlx.api.models import (
            EmbeddingData,
            EmbeddingResponse,
            EmbeddingUsage,
        )

        response = EmbeddingResponse(
            data=[EmbeddingData(index=0, embedding=[1.0, 2.0, 3.0])],
            model="text-embedding-3-large",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
        )
        d = response.model_dump()
        assert d["object"] == "list"
        assert d["data"][0]["object"] == "embedding"
        assert d["data"][0]["index"] == 0
        assert d["data"][0]["embedding"] == [1.0, 2.0, 3.0]
        assert d["model"] == "text-embedding-3-large"
        assert d["usage"]["prompt_tokens"] == 5
        assert d["usage"]["total_tokens"] == 5


# =============================================================================
# Unit Tests - Embedding Engine
# =============================================================================


class TestEmbeddingEngineLoadWarning:
    """Test the startup warning for large, unbounded context windows."""

    def test_load_warns_on_large_uncapped_context(self):
        """No ceiling + a large model-reported context window warns at load time."""
        pytest.importorskip("mlx_embeddings")

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 40960
        mock_tokenizer = MagicMock()

        with (
            patch("mlx_embeddings.load", return_value=(mock_model, mock_tokenizer)),
            patch("vllm_mlx.embedding.logger") as mock_logger,
        ):
            engine.load()

        mock_logger.warning.assert_called_once()
        assert engine._max_length == 40960

    def test_load_does_not_warn_when_ceiling_set(self):
        """An explicit ceiling suppresses the large-context startup warning."""
        pytest.importorskip("mlx_embeddings")

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model", max_length_ceiling=4096)

        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 40960
        mock_tokenizer = MagicMock()

        with (
            patch("mlx_embeddings.load", return_value=(mock_model, mock_tokenizer)),
            patch("vllm_mlx.embedding.logger") as mock_logger,
        ):
            engine.load()

        mock_logger.warning.assert_not_called()
        assert engine._max_length == 4096

    def test_load_does_not_warn_for_small_context(self):
        """No ceiling, but a small (classic BERT-sized) context: no warning."""
        pytest.importorskip("mlx_embeddings")

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 512
        mock_tokenizer = MagicMock()

        with (
            patch("mlx_embeddings.load", return_value=(mock_model, mock_tokenizer)),
            patch("vllm_mlx.embedding.logger") as mock_logger,
        ):
            engine.load()

        mock_logger.warning.assert_not_called()


class TestEmbeddingEngine:
    """Test the EmbeddingEngine wrapper."""

    @patch("vllm_mlx.embedding.EmbeddingEngine.load")
    @patch(
        "vllm_mlx.embedding.EmbeddingEngine.is_loaded",
        new_callable=lambda: property(lambda self: True),
    )
    def test_embed_calls_model_directly(self, _mock_loaded, mock_load):
        """Test embed tokenizes and calls model directly (bypasses generate)."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]

        mock_model = MagicMock(return_value=mock_output)

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2], [3, 4]]),
            "attention_mask": np.array([[1, 1], [1, 1]]),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer

        engine._model = mock_model
        engine._tokenizer = mock_tokenizer
        result = engine.embed(["hello", "world"])

        mock_model.assert_called_once()
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]

    @patch("vllm_mlx.embedding.EmbeddingEngine.load")
    @patch(
        "vllm_mlx.embedding.EmbeddingEngine.is_loaded",
        new_callable=lambda: property(lambda self: True),
    )
    def test_embed_clears_mlx_cache_after_batch(self, _mock_loaded, mock_load):
        """Test embed releases MLX buffers after converting to Python lists."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]

        mock_model = MagicMock(return_value=mock_output)

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2], [3, 4]]),
            "attention_mask": np.array([[1, 1], [1, 1]]),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer

        engine._model = mock_model
        engine._tokenizer = mock_tokenizer

        with patch("vllm_mlx.embedding.mx.clear_cache") as mock_clear_cache:
            result = engine.embed(["hello", "world"])
            mock_clear_cache.assert_called_once()

        assert result[0] == [0.1, 0.2]

    def test_embed_uses_config_max_length(self):
        """Positive: truncation length follows model.config.max_position_embeddings."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.1, 0.2]]
        mock_model = MagicMock(return_value=mock_output)
        mock_model.config.max_position_embeddings = 8192

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": np.array([[1, 1]]),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer

        with patch.object(engine, "_ensure_loaded"):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            engine.embed(["hello"])

        assert mock_inner_tokenizer.call_args.kwargs["max_length"] == 8192

    def test_embed_defaults_to_512_without_config(self):
        """Negative: no usable config/tokenizer value falls back to 512."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.1, 0.2]]
        mock_model = MagicMock(return_value=mock_output)
        # .config.max_position_embeddings is a MagicMock (non-int) -> rejected.

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.model_max_length = MagicMock()  # non-int -> rejected
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": np.array([[1, 1]]),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer

        with patch.object(engine, "_ensure_loaded"):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            engine.embed(["hello"])

        assert mock_inner_tokenizer.call_args.kwargs["max_length"] == 512

    def test_embed_normalises_single_string(self):
        """Test that a single string input is wrapped into a list."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.5, 0.6]]

        mock_model = MagicMock(return_value=mock_output)

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": np.array([[1, 1]]),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer

        with patch.object(engine, "_ensure_loaded"):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            result = engine.embed("single text")

        assert len(result) == 1

    def test_count_tokens(self):
        """Test token counting for usage reporting."""
        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        engine._tokenizer = mock_tokenizer
        engine._model = MagicMock()  # mark as loaded

        count = engine.count_tokens(["hello", "world"])
        assert count == 10  # 5 tokens * 2 texts

    def test_count_tokens_falls_back_to_estimate_on_tokenizer_failure(self):
        """Usage reporting may use the ~4-chars-per-token estimate when the
        tokenizer itself can't encode the text — that's the one place the
        heuristic is still acceptable (see test_embed_error_policy_propagates_
        tokenizer_failure for why it must NOT apply to policy enforcement)."""
        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = RuntimeError("boom")
        engine._tokenizer = mock_tokenizer
        engine._model = MagicMock()

        count = engine.count_tokens("12345678")  # 8 chars -> //4 == 2
        assert count == 2

    def test_embed_error_policy_propagates_tokenizer_failure(self):
        """Regression (review blocker): overflow-policy enforcement must use
        an exact token count. It must NOT silently fall back to the
        ~4-chars-per-token heuristic on a tokenizer failure — that could
        let an over-limit input through, or reject a valid one with an
        inaccurate token_count. A genuine tokenizer failure should surface
        as an error, not be masked."""
        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine(
            "test-model", max_length_ceiling=10, overflow_policy="error"
        )

        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 8192
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = RuntimeError("tokenizer exploded")

        with patch.object(engine, "_ensure_loaded"):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            with pytest.raises(RuntimeError, match="tokenizer exploded"):
                engine.embed(["some text"])

        mock_model.assert_not_called()

    def test_effective_max_length_reflects_ceiling_clamp(self):
        """effective_max_length is what embed() actually uses (post-ceiling),
        not the raw ceiling — this backs /v1/status's `max_length` field."""
        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model", max_length_ceiling=4096)
        assert engine.effective_max_length is None  # not loaded yet

        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 512  # below the ceiling
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        # A 512-token model with a 4096 ceiling: the ceiling only clamps
        # downward, so the effective value stays 512, not the ceiling.
        assert engine.effective_max_length == 512

    def test_embed_ceiling_clamps_config_value(self):
        """Positive: max_length_ceiling clamps the model-derived max_length."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model", max_length_ceiling=256)

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.1, 0.2]]
        mock_model = MagicMock(return_value=mock_output)
        mock_model.config.max_position_embeddings = 8192

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": np.array([[1, 1]]),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer

        with patch.object(engine, "_ensure_loaded"):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            engine.embed(["hello"])

        assert mock_inner_tokenizer.call_args.kwargs["max_length"] == 256

    def test_embed_error_policy_raises_on_overflow(self):
        """overflow_policy='error' rejects an over-limit input instead of
        truncating it, and never reaches the model."""
        from vllm_mlx.embedding import EmbeddingEngine, EmbeddingLengthExceededError

        engine = EmbeddingEngine(
            "test-model", max_length_ceiling=10, overflow_policy="error"
        )

        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 8192
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(20))  # 20 > ceiling of 10

        with patch.object(engine, "_ensure_loaded"):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            with pytest.raises(EmbeddingLengthExceededError) as exc_info:
                engine.embed(["a very long text"])

        assert exc_info.value.text_index == 0
        assert exc_info.value.token_count == 20
        assert exc_info.value.max_length == 10
        mock_model.assert_not_called()

    def test_embed_error_policy_does_not_raise_within_limit(self):
        """overflow_policy='error' embeds normally when inputs are within limit."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine(
            "test-model", max_length_ceiling=10, overflow_policy="error"
        )

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.1, 0.2]]
        mock_model = MagicMock(return_value=mock_output)
        mock_model.config.max_position_embeddings = 8192

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": np.array([[1, 1]]),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        mock_tokenizer.encode.return_value = [1, 2, 3]  # within limit

        with patch.object(engine, "_ensure_loaded"):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            result = engine.embed(["short"])

        assert len(result) == 1

    def test_embed_truncate_policy_logs_and_increments_metric(self):
        """overflow_policy='truncate' (default) still truncates, but now
        observably: a warning is logged and the metric is incremented."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model", max_length_ceiling=10)

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.1, 0.2]]
        mock_model = MagicMock(return_value=mock_output)
        mock_model.config.max_position_embeddings = 8192

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": np.array([[1, 1]]),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        mock_tokenizer.encode.return_value = list(range(20))  # 20 > ceiling of 10

        with (
            patch.object(engine, "_ensure_loaded"),
            patch("vllm_mlx.embedding.metrics") as mock_metrics,
        ):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            result = engine.embed(["a very long text"])

        assert len(result) == 1
        mock_metrics.observe_embedding_truncated.assert_called_once_with(
            model="test-model"
        )
        # Still truncates to the effective max_length, unlike 'error'.
        assert mock_inner_tokenizer.call_args.kwargs["max_length"] == 10

    def test_embed_truncate_policy_logs_once_per_batch(self):
        """A batch with multiple over-limit texts logs a single aggregated
        warning (not one per text), so a large batch can't flood the log.
        The metric still increments once per over-limit text."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model", max_length_ceiling=10)

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.1]] * 3
        mock_model = MagicMock(return_value=mock_output)
        mock_model.config.max_position_embeddings = 8192

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2]] * 3),
            "attention_mask": np.array([[1, 1]] * 3),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        # All three texts exceed the ceiling of 10.
        mock_tokenizer.encode.return_value = list(range(20))

        with (
            patch.object(engine, "_ensure_loaded"),
            patch("vllm_mlx.embedding.logger") as mock_logger,
            patch("vllm_mlx.embedding.metrics") as mock_metrics,
        ):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            engine.embed(["long one", "long two", "long three"])

        mock_logger.warning.assert_called_once()
        assert mock_metrics.observe_embedding_truncated.call_count == 3

    def test_embed_handles_mixed_length_batch(self):
        """A batch mixing under- and over-limit texts must only flag the
        over-limit ones — the short texts in the same call must be neither
        truncated-and-warned-about nor otherwise affected by their
        over-limit neighbor."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model", max_length_ceiling=10)

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.1]] * 3
        mock_model = MagicMock(return_value=mock_output)
        mock_model.config.max_position_embeddings = 8192

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2]] * 3),
            "attention_mask": np.array([[1, 1]] * 3),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        # Only the middle text (20 tokens) exceeds the ceiling of 10; the
        # other two (5 and 8 tokens) are within limit.
        mock_tokenizer.encode.side_effect = [
            list(range(5)),
            list(range(20)),
            list(range(8)),
        ]

        with (
            patch.object(engine, "_ensure_loaded"),
            patch("vllm_mlx.embedding.logger") as mock_logger,
            patch("vllm_mlx.embedding.metrics") as mock_metrics,
        ):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            result = engine.embed(["short", "way too long", "also short"])

        # Only the one over-limit text (index 1) triggers a warning/metric.
        mock_logger.warning.assert_called_once()
        # warning(fmt, overflow_count, total_count, indices, longest, ...)
        _fmt, overflow_count, total_count, indices, longest = (
            mock_logger.warning.call_args.args[:5]
        )
        assert (overflow_count, total_count) == (1, 3)
        assert indices == "[1]"
        assert longest == 20
        assert mock_metrics.observe_embedding_truncated.call_count == 1
        # All three still get embedded in one call, in order.
        assert mock_model.call_count == 1
        assert len(result) == 3

    def test_embed_stays_a_single_batch_within_budget(self):
        """A typical request (well under the token budget) still runs as one
        forward pass — sub-batching must not change behavior for the common
        case, only bound the uncommon large-request one."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")  # default token_budget

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.1]] * 3
        mock_model = MagicMock(return_value=mock_output)
        mock_model.config.max_position_embeddings = 8192

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2]] * 3),
            "attention_mask": np.array([[1, 1]] * 3),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        mock_tokenizer.encode.return_value = [1, 2, 3]

        with patch.object(engine, "_ensure_loaded"):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            result = engine.embed(["a", "b", "c"])

        assert mock_model.call_count == 1
        assert len(result) == 3

    def test_embed_packs_texts_into_token_budget_bounded_batches(self):
        """Regression (review blocker): a request's texts must not all be
        padded together in a single unbounded batch — a large-context model
        given many/long inputs could otherwise blow up memory/compute for
        that one request. embed() packs them into sub-batches capped by the
        token budget instead, calling the model once per sub-batch, and
        reassembles results in the original input order."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model", token_budget=10)

        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 8192
        # Two forward passes expected: first batch holds 2 texts, second 1.
        output_a = MagicMock()
        output_a.text_embeds.tolist.return_value = [[0.1, 0.1], [0.2, 0.2]]
        output_b = MagicMock()
        output_b.text_embeds.tolist.return_value = [[0.3, 0.3]]
        mock_model.side_effect = [output_a, output_b]

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.side_effect = [
            {
                "input_ids": np.array([[1, 2, 3], [1, 2, 3]]),
                "attention_mask": np.array([[1, 1, 1], [1, 1, 1]]),
            },
            {
                "input_ids": np.array([[1, 2, 3, 4, 5, 6]]),
                "attention_mask": np.array([[1, 1, 1, 1, 1, 1]]),
            },
        ]
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        # Raw token counts for the three texts: 3, 3, 6.
        mock_tokenizer.encode.side_effect = [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2, 3, 4, 5],
        ]

        with (
            patch.object(engine, "_ensure_loaded"),
            patch("vllm_mlx.embedding.mx.clear_cache") as mock_clear_cache,
        ):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            result = engine.embed(["a", "b", "c"])

        assert mock_model.call_count == 2
        assert mock_clear_cache.call_count == 2
        assert mock_inner_tokenizer.call_count == 2
        # texts 0+1 (3 + 3 = 6 <= budget of 10) share a batch...
        assert mock_inner_tokenizer.call_args_list[0].args[0] == ["a", "b"]
        # ...adding text 2 (6 tokens) would exceed the budget, so it's alone.
        assert mock_inner_tokenizer.call_args_list[1].args[0] == ["c"]
        # Results land back in original input order regardless of batching.
        assert result == [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]

    def test_token_budget_accounts_for_padding(self):
        """A long input followed by many short inputs must not produce a batch
        whose padded tensor exceeds the configured budget."""
        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model", token_budget=10)
        counts = [8, 1, 1]

        batches = engine._token_budget_batches(counts)

        assert batches == [[0], [1, 2]]
        for batch in batches:
            if len(batch) > 1:
                assert len(batch) * max(counts[i] for i in batch) <= 10

    def test_embed_clears_cache_when_a_later_batch_fails(self):
        """Every attempted forward pass releases its cached MLX buffers,
        including the pass that raises."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model", token_budget=5)
        output = MagicMock()
        output.text_embeds.tolist.return_value = [[0.1, 0.1]]
        mock_model = MagicMock(side_effect=[output, RuntimeError("model failed")])
        mock_model.config.max_position_embeddings = 8192

        mock_inner_tokenizer = MagicMock(
            return_value={
                "input_ids": np.array([[1, 2, 3]]),
                "attention_mask": np.array([[1, 1, 1]]),
            }
        )
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        mock_tokenizer.encode.side_effect = [[0, 1, 2], [0, 1, 2]]

        with (
            patch.object(engine, "_ensure_loaded"),
            patch("vllm_mlx.embedding.mx.clear_cache") as mock_clear_cache,
            pytest.raises(RuntimeError, match="model failed"),
        ):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            engine.embed(["a", "b"])

        assert mock_model.call_count == 2
        assert mock_clear_cache.call_count == 2


# =============================================================================
# Integration Tests - FastAPI Endpoint
# =============================================================================


class TestEmbeddingsEndpoint:
    """Test the /v1/embeddings endpoint via TestClient."""

    @pytest.fixture()
    def client(self):
        """Create a FastAPI test client with mocked embedding engine."""
        from fastapi.testclient import TestClient

        from vllm_mlx.server import app

        return TestClient(app)

    def test_batch_input_preserves_order(self, client):
        """Test batch embedding returns vectors with correct indices."""
        import vllm_mlx.server as srv

        texts = ["first", "second", "third"]
        mock_engine = MagicMock()
        mock_engine.model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
        mock_engine.embed.return_value = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
        mock_engine.count_tokens.return_value = 9

        original = srv._embedding_engine
        srv._embedding_engine = mock_engine
        try:
            resp = client.post(
                "/v1/embeddings",
                json={"model": "mlx-community/all-MiniLM-L6-v2-4bit", "input": texts},
            )
        finally:
            srv._embedding_engine = original

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 3
        for i in range(3):
            assert body["data"][i]["index"] == i
        # Verify order matches
        assert body["data"][0]["embedding"] == [1.0, 0.0]
        assert body["data"][2]["embedding"] == [0.5, 0.5]

    def test_empty_input_returns_400(self, client):
        """Test that empty input list returns 400 error."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "mlx-community/all-MiniLM-L6-v2-4bit"

        original = srv._embedding_engine
        srv._embedding_engine = mock_engine
        try:
            resp = client.post(
                "/v1/embeddings",
                json={"model": "mlx-community/all-MiniLM-L6-v2-4bit", "input": []},
            )
        finally:
            srv._embedding_engine = original

        assert resp.status_code == 400

    def test_model_hot_swap(self, client):
        """Test that switching to another allowlisted model triggers reload."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "old-model"
        mock_engine.embed.return_value = [[0.1]]
        mock_engine.count_tokens.return_value = 1

        original = srv._embedding_engine
        srv._embedding_engine = mock_engine

        try:
            with patch("vllm_mlx.embedding.EmbeddingEngine") as mock_cls:
                new_engine = MagicMock()
                new_engine.model_name = "mlx-community/multilingual-e5-small-mlx"
                new_engine.embed.return_value = [[0.9]]
                new_engine.count_tokens.return_value = 1
                mock_cls.return_value = new_engine

                resp = client.post(
                    "/v1/embeddings",
                    json={
                        "model": "mlx-community/multilingual-e5-small-mlx",
                        "input": "test",
                    },
                )
                assert resp.status_code == 200
                mock_cls.assert_called_once_with(
                    "mlx-community/multilingual-e5-small-mlx",
                    max_length_ceiling=srv._embedding_max_length,
                    overflow_policy=srv._embedding_overflow_policy,
                )
                new_engine.load.assert_called_once()
        finally:
            srv._embedding_engine = original

    def test_model_locked_rejects_different_model(self, client):
        """Test that a locked embedding model rejects requests for different models."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "locked-model"

        original_engine = srv._embedding_engine
        original_locked = srv._embedding_model_locked
        srv._embedding_engine = mock_engine
        srv._embedding_model_locked = "locked-model"

        try:
            resp = client.post(
                "/v1/embeddings",
                json={"model": "other-model", "input": "test"},
            )
            assert resp.status_code == 400
            body = resp.json()
            assert "locked-model" in body["detail"]
            assert "other-model" in body["detail"]
        finally:
            srv._embedding_engine = original_engine
            srv._embedding_model_locked = original_locked

    def test_unknown_embedding_model_rejected(self, client):
        """Test that request-time embedding loads reject unknown models."""
        resp = client.post(
            "/v1/embeddings",
            json={"model": "attacker/unknown-embedding", "input": "test"},
        )

        assert resp.status_code == 400
        body = resp.json()
        assert "attacker/unknown-embedding" in body["detail"]
        assert "--embedding-model" in body["detail"]

    def test_overflow_error_policy_returns_structured_400(self, client):
        """Test overflow_policy='error' surfaces a structured 400 with the
        observed token count and effective max_length."""
        import vllm_mlx.server as srv
        from vllm_mlx.embedding import EmbeddingLengthExceededError

        mock_engine = MagicMock()
        mock_engine.model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
        mock_engine.count_tokens.return_value = 1400
        mock_engine.embed.side_effect = EmbeddingLengthExceededError(0, 1400, 1024)

        original = srv._embedding_engine
        srv._embedding_engine = mock_engine
        try:
            resp = client.post(
                "/v1/embeddings",
                json={
                    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
                    "input": "a very long text",
                },
            )
        finally:
            srv._embedding_engine = original

        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error"] == "embedding_input_too_long"
        assert detail["input_index"] == 0
        assert detail["token_count"] == 1400
        assert detail["max_length"] == 1024

    def test_load_embedding_model_passes_ceiling_and_policy(self):
        """Test load_embedding_model() wires the configured ceiling/policy
        globals through to EmbeddingEngine."""
        import vllm_mlx.server as srv

        original_engine = srv._embedding_engine
        original_ceiling = srv._embedding_max_length
        original_policy = srv._embedding_overflow_policy
        srv._embedding_engine = None
        srv._embedding_max_length = 999
        srv._embedding_overflow_policy = "error"

        try:
            with patch("vllm_mlx.embedding.EmbeddingEngine") as mock_cls:
                mock_cls.return_value = MagicMock()
                srv.load_embedding_model("some-model", lock=False)
                mock_cls.assert_called_once_with(
                    "some-model", max_length_ceiling=999, overflow_policy="error"
                )
        finally:
            srv._embedding_engine = original_engine
            srv._embedding_max_length = original_ceiling
            srv._embedding_overflow_policy = original_policy

    def test_status_reports_embedding_config(self, client):
        """Test GET /v1/status includes the effective embedding config."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
        mock_engine.effective_max_length = 1024

        original_engine = srv._embedding_engine
        original_ceiling = srv._embedding_max_length
        original_policy = srv._embedding_overflow_policy
        srv._embedding_engine = mock_engine
        srv._embedding_max_length = 1024
        srv._embedding_overflow_policy = "error"

        try:
            resp = client.get("/v1/status")
        finally:
            srv._embedding_engine = original_engine
            srv._embedding_max_length = original_ceiling
            srv._embedding_overflow_policy = original_policy

        assert resp.status_code == 200
        embedding_status = resp.json()["embedding"]
        assert embedding_status == {
            "model": "mlx-community/all-MiniLM-L6-v2-4bit",
            "max_length": 1024,
            "max_length_ceiling": 1024,
            "overflow_policy": "error",
        }

    def test_status_reports_effective_max_length_not_raw_ceiling(self, client):
        """Regression: a ceiling higher than what the loaded model actually
        supports must not be echoed back as `max_length` — /v1/status must
        report what embed() really applies, distinct from the configured
        ceiling (which is still reported, under `max_length_ceiling`)."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
        # A 512-token model with a 4096 ceiling: the ceiling only clamps
        # downward, so the engine still resolves to (and uses) 512.
        mock_engine.effective_max_length = 512

        original_engine = srv._embedding_engine
        original_ceiling = srv._embedding_max_length
        original_policy = srv._embedding_overflow_policy
        srv._embedding_engine = mock_engine
        srv._embedding_max_length = 4096
        srv._embedding_overflow_policy = "truncate"

        try:
            resp = client.get("/v1/status")
        finally:
            srv._embedding_engine = original_engine
            srv._embedding_max_length = original_ceiling
            srv._embedding_overflow_policy = original_policy

        embedding_status = resp.json()["embedding"]
        assert embedding_status["max_length"] == 512
        assert embedding_status["max_length_ceiling"] == 4096

    def test_status_max_length_ceiling_is_null_not_string_when_unset(self, client):
        """Regression: with no --embedding-max-length configured,
        max_length_ceiling must be JSON null (int | None) — a mixed-type
        field is a broken API contract for anything that types this
        response."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
        mock_engine.effective_max_length = 8192

        original_engine = srv._embedding_engine
        original_ceiling = srv._embedding_max_length
        srv._embedding_engine = mock_engine
        srv._embedding_max_length = None  # no ceiling configured

        try:
            resp = client.get("/v1/status")
        finally:
            srv._embedding_engine = original_engine
            srv._embedding_max_length = original_ceiling

        embedding_status = resp.json()["embedding"]
        assert embedding_status["max_length_ceiling"] is None
        assert isinstance(embedding_status["max_length"], int)


# =============================================================================
# Slow Integration Test - Real Model
# =============================================================================


@pytest.mark.slow
class TestEmbeddingsRealModel:
    """Integration tests with a real mlx-embeddings model."""

    @pytest.fixture(scope="class")
    def engine(self):
        pytest.importorskip("mlx_embeddings")
        from vllm_mlx.embedding import EmbeddingEngine

        eng = EmbeddingEngine("mlx-community/all-MiniLM-L6-v2-4bit")
        eng.load()
        return eng

    def test_single_embedding_shape(self, engine):
        """Test that a single text produces a correctly shaped vector."""
        result = engine.embed("Hello world")
        assert len(result) == 1
        assert len(result[0]) > 0  # non-empty embedding
        assert all(isinstance(v, float) for v in result[0])
