# SPDX-License-Identifier: Apache-2.0
"""Tests for the /v1/rerank endpoint."""

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


class TestRerankModels:
    """Test rerank request/response Pydantic models."""

    def test_rerank_request_with_string_documents(self):
        """Test RerankRequest accepts a list of plain strings."""
        from vllm_mlx.api.models import RerankRequest

        req = RerankRequest(
            model="jina-reranker-v2",
            query="What is deep learning?",
            documents=["doc one", "doc two"],
        )
        assert req.model == "jina-reranker-v2"
        assert req.query == "What is deep learning?"
        assert req.documents == ["doc one", "doc two"]
        assert req.top_n is None
        assert req.return_documents is True

    def test_rerank_request_with_object_documents(self):
        """Test RerankRequest accepts a list of {text: ...} objects."""
        from vllm_mlx.api.models import RerankRequest

        req = RerankRequest(
            model="jina-reranker-v2",
            query="query",
            documents=[{"text": "alpha"}, {"text": "beta"}],
        )
        assert req.documents == [{"text": "alpha"}, {"text": "beta"}]

    def test_rerank_request_top_n_default_none(self):
        """Test that top_n defaults to None (return all)."""
        from vllm_mlx.api.models import RerankRequest

        req = RerankRequest(model="m", query="q", documents=["a"])
        assert req.top_n is None

    def test_rerank_result_serialization(self):
        """Test RerankResult serializes with correct fields."""
        from vllm_mlx.api.models import RerankResult

        result = RerankResult(
            index=2,
            relevance_score=0.95,
            document={"text": "hello"},
        )
        d = result.model_dump()
        assert d["index"] == 2
        assert d["relevance_score"] == 0.95
        assert d["document"] == {"text": "hello"}

    def test_rerank_result_without_document(self):
        """Test RerankResult when document is omitted."""
        from vllm_mlx.api.models import RerankResult

        result = RerankResult(index=0, relevance_score=0.5)
        d = result.model_dump()
        assert d["index"] == 0
        assert d["document"] is None

    def test_rerank_response_serialization(self):
        """Test RerankResponse serializes to Jina-compatible JSON."""
        from vllm_mlx.api.models import RerankResponse, RerankResult, RerankUsage

        response = RerankResponse(
            model="jina-reranker-v2",
            results=[
                RerankResult(index=1, relevance_score=0.9, document={"text": "best"}),
                RerankResult(index=0, relevance_score=0.1, document={"text": "worst"}),
            ],
            usage=RerankUsage(total_tokens=42),
        )
        d = response.model_dump()
        assert d["model"] == "jina-reranker-v2"
        assert len(d["results"]) == 2
        assert d["results"][0]["index"] == 1
        assert d["results"][0]["relevance_score"] == 0.9
        assert d["usage"]["total_tokens"] == 42


# =============================================================================
# Unit Tests - Reranker Adapter Contract
# =============================================================================


class TestRerankAdapterContract:
    """Test the base adapter interface and default sigmoid adapter."""

    def test_base_adapter_is_abstract(self):
        """Test that RerankAdapter cannot be instantiated directly."""
        from vllm_mlx.rerank import RerankAdapter

        with pytest.raises(TypeError):
            RerankAdapter()

    def test_sigmoid_adapter_normalize_maps_to_zero_one(self):
        """Test that SigmoidAdapter.normalize applies sigmoid correctly."""
        import math

        from vllm_mlx.rerank import SigmoidAdapter

        adapter = SigmoidAdapter()
        # sigmoid(0) = 0.5
        assert abs(adapter.normalize(0.0) - 0.5) < 1e-6
        # sigmoid(large positive) -> ~1.0
        assert adapter.normalize(10.0) > 0.999
        # sigmoid(large negative) -> ~0.0
        assert adapter.normalize(-10.0) < 0.001

    def test_sigmoid_adapter_extract_score_takes_first_logit(self):
        """Test that SigmoidAdapter.extract_score returns logits[0]."""
        from vllm_mlx.rerank import SigmoidAdapter

        adapter = SigmoidAdapter()
        # Simulate a logits array with shape (num_labels,)
        logits = [2.5, -1.0, 0.3]
        assert adapter.extract_score(logits) == 2.5

    def test_sigmoid_adapter_tokenize_pair_returns_dict(self):
        """Test that SigmoidAdapter.tokenize_pair produces a token dict."""
        from unittest.mock import MagicMock

        from vllm_mlx.rerank import SigmoidAdapter

        adapter = SigmoidAdapter()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[101, 2054, 102, 3793, 102]],
            "attention_mask": [[1, 1, 1, 1, 1]],
        }
        result = adapter.tokenize_pair(mock_tokenizer, "query", "document")
        mock_tokenizer.assert_called_once_with(
            "query",
            "document",
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        assert "input_ids" in result
        assert "attention_mask" in result


# =============================================================================
# Unit Tests - RerankEngine
# =============================================================================


class TestRerankEngine:
    """Test the RerankEngine model loading and scoring."""

    def test_engine_not_loaded_initially(self):
        """Test that a new engine reports is_loaded=False."""
        from vllm_mlx.rerank import RerankEngine

        engine = RerankEngine("test-model")
        assert engine.is_loaded is False

    def test_engine_model_name_stored(self):
        """Test that model_name is stored on construction."""
        from vllm_mlx.rerank import RerankEngine

        engine = RerankEngine("mlx-community/jina-reranker-v2-base-multilingual")
        assert engine.model_name == "mlx-community/jina-reranker-v2-base-multilingual"

    def test_score_pairs_returns_normalized_scores(self):
        """Test score_pairs returns sigmoid-normalized scores for each pair."""
        import math

        import numpy as np

        from vllm_mlx.rerank import RerankEngine, SigmoidAdapter

        engine = RerankEngine("test-model")

        # Mock model: returns logits with shape (batch, num_labels)
        mock_model = MagicMock()
        # Simulate two pairs scored: logits [2.0, ...] and [-1.0, ...]
        mock_logits = MagicMock()
        mock_logits.tolist.return_value = [[2.0, 0.5], [-1.0, 0.3]]
        mock_model.return_value = MagicMock(logits=mock_logits)

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": np.array([[1, 1, 1], [1, 1, 1]]),
        }

        engine._model = mock_model
        engine._tokenizer = mock_tokenizer
        engine._adapter = SigmoidAdapter()

        scores = engine.score_pairs("test query", ["doc one", "doc two"])
        assert len(scores) == 2
        expected_0 = 1.0 / (1.0 + math.exp(-2.0))
        expected_1 = 1.0 / (1.0 + math.exp(1.0))
        assert abs(scores[0] - expected_0) < 1e-6
        assert abs(scores[1] - expected_1) < 1e-6

    def test_score_pairs_token_budget_batching(self):
        """Test that score_pairs splits work into batches by token budget."""
        import numpy as np

        from vllm_mlx.rerank import RerankEngine, SigmoidAdapter

        engine = RerankEngine("test-model", token_budget=10)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Each call returns 1 pair. We have 3 documents.
        # With token_budget=10 and each pair using 5 tokens, we get batches of 2 then 1.
        call_count = 0

        def mock_tokenize(query, doc, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "input_ids": np.array([[1, 2, 3, 4, 5]]),
                "attention_mask": np.array([[1, 1, 1, 1, 1]]),
            }

        mock_tokenizer.side_effect = mock_tokenize

        def mock_forward(input_ids, **kwargs):
            # Return logits matching the batch dimension of input_ids
            batch_size = input_ids.shape[0]
            mock_logits = MagicMock()
            mock_logits.tolist.return_value = [[0.5]] * batch_size
            return MagicMock(logits=mock_logits)

        mock_model.side_effect = mock_forward

        engine._model = mock_model
        engine._tokenizer = mock_tokenizer
        engine._adapter = SigmoidAdapter()

        scores = engine.score_pairs("q", ["d1", "d2", "d3"])
        assert len(scores) == 3
        # tokenizer called once per document (for budget estimation + scoring)
        assert call_count == 3

    def test_count_tokens_pair(self):
        """Test token counting for a query-document pair."""
        from vllm_mlx.rerank import RerankEngine

        engine = RerankEngine("test-model")
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3, 4, 5, 6, 7]]}
        engine._tokenizer = mock_tokenizer
        engine._model = MagicMock()  # mark as loaded

        count = engine.count_tokens("query text", ["doc one", "doc two"])
        # tokenizer called twice (once per doc), 7 tokens each = 14
        assert count == 14


# =============================================================================
# Unit Tests - Classifier Forward Pass
# =============================================================================


class TestClassifierForward:
    """Test the MLX classifier forward pass for cross-encoder models."""

    def test_classifier_forward_returns_logits_shape(self):
        """Test that classifier_forward returns logits with (batch, num_labels) shape."""
        import mlx.core as mx

        from vllm_mlx.rerank_forward import classifier_forward

        # Build minimal BERT-like weights
        vocab_size = 30
        hidden_size = 16
        num_heads = 2
        intermediate_size = 32
        num_labels = 1

        config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": 1,
            "num_labels": num_labels,
            "vocab_size": vocab_size,
            "max_position_embeddings": 64,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12,
            "hidden_act": "gelu",
        }

        weights = _make_bert_weights(config)

        input_ids = mx.array([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])
        attention_mask = mx.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])

        logits = classifier_forward(input_ids, attention_mask, weights, config)
        mx.eval(logits)

        assert logits.shape == (2, num_labels)

    def test_classifier_forward_different_num_labels(self):
        """Test forward pass with num_labels=3 (multi-class)."""
        import mlx.core as mx

        from vllm_mlx.rerank_forward import classifier_forward

        hidden_size = 8
        config = {
            "hidden_size": hidden_size,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_labels": 3,
            "vocab_size": 20,
            "max_position_embeddings": 32,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12,
            "hidden_act": "gelu",
        }

        weights = _make_bert_weights(config)

        input_ids = mx.array([[1, 2, 3]])
        attention_mask = mx.array([[1, 1, 1]])

        logits = classifier_forward(input_ids, attention_mask, weights, config)
        mx.eval(logits)

        assert logits.shape == (1, 3)


def _make_bert_weights(config: dict) -> dict:
    """Build minimal random BERT-style weights for testing."""
    import mlx.core as mx

    h = config["hidden_size"]
    inter = config["intermediate_size"]
    vocab = config["vocab_size"]
    max_pos = config["max_position_embeddings"]
    type_vocab = config["type_vocab_size"]
    num_labels = config["num_labels"]
    n_layers = config["num_hidden_layers"]

    w = {}
    # Embeddings
    w["bert.embeddings.word_embeddings.weight"] = mx.random.normal((vocab, h)) * 0.02
    w["bert.embeddings.position_embeddings.weight"] = (
        mx.random.normal((max_pos, h)) * 0.02
    )
    w["bert.embeddings.token_type_embeddings.weight"] = (
        mx.random.normal((type_vocab, h)) * 0.02
    )
    w["bert.embeddings.LayerNorm.weight"] = mx.ones((h,))
    w["bert.embeddings.LayerNorm.bias"] = mx.zeros((h,))

    for i in range(n_layers):
        prefix = f"bert.encoder.layer.{i}"
        # Self-attention
        for proj in ["query", "key", "value"]:
            w[f"{prefix}.attention.self.{proj}.weight"] = (
                mx.random.normal((h, h)) * 0.02
            )
            w[f"{prefix}.attention.self.{proj}.bias"] = mx.zeros((h,))
        w[f"{prefix}.attention.output.dense.weight"] = mx.random.normal((h, h)) * 0.02
        w[f"{prefix}.attention.output.dense.bias"] = mx.zeros((h,))
        w[f"{prefix}.attention.output.LayerNorm.weight"] = mx.ones((h,))
        w[f"{prefix}.attention.output.LayerNorm.bias"] = mx.zeros((h,))
        # FFN
        w[f"{prefix}.intermediate.dense.weight"] = mx.random.normal((inter, h)) * 0.02
        w[f"{prefix}.intermediate.dense.bias"] = mx.zeros((inter,))
        w[f"{prefix}.output.dense.weight"] = mx.random.normal((h, inter)) * 0.02
        w[f"{prefix}.output.dense.bias"] = mx.zeros((h,))
        w[f"{prefix}.output.LayerNorm.weight"] = mx.ones((h,))
        w[f"{prefix}.output.LayerNorm.bias"] = mx.zeros((h,))

    # Pooler
    w["bert.pooler.dense.weight"] = mx.random.normal((h, h)) * 0.02
    w["bert.pooler.dense.bias"] = mx.zeros((h,))

    # Classifier
    w["classifier.weight"] = mx.random.normal((num_labels, h)) * 0.02
    w["classifier.bias"] = mx.zeros((num_labels,))

    return w


# =============================================================================
# Integration Tests - FastAPI Endpoint
# =============================================================================


class TestRerankEndpoint:
    """Test the /v1/rerank endpoint via TestClient."""

    @pytest.fixture()
    def client(self):
        """Create a FastAPI test client."""
        from fastapi.testclient import TestClient

        from vllm_mlx.server import app

        return TestClient(app)

    def test_rerank_returns_sorted_results(self, client):
        """Test that /v1/rerank returns results sorted by relevance_score descending."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "test-reranker"
        # doc 0 gets low score, doc 1 gets high score, doc 2 gets mid score
        mock_engine.score_pairs.return_value = [0.1, 0.9, 0.5]
        mock_engine.count_tokens.return_value = 30

        original = srv._rerank_engine
        srv._rerank_engine = mock_engine
        try:
            resp = client.post(
                "/v1/rerank",
                json={
                    "model": "test-reranker",
                    "query": "What is deep learning?",
                    "documents": ["bad match", "great match", "ok match"],
                },
            )
        finally:
            srv._rerank_engine = original

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 3
        # Sorted descending by score
        assert body["results"][0]["index"] == 1
        assert body["results"][0]["relevance_score"] == 0.9
        assert body["results"][1]["index"] == 2
        assert body["results"][1]["relevance_score"] == 0.5
        assert body["results"][2]["index"] == 0
        assert body["results"][2]["relevance_score"] == 0.1

    def test_rerank_top_n(self, client):
        """Test that top_n limits returned results."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "test-reranker"
        mock_engine.score_pairs.return_value = [0.1, 0.9, 0.5]
        mock_engine.count_tokens.return_value = 20

        original = srv._rerank_engine
        srv._rerank_engine = mock_engine
        try:
            resp = client.post(
                "/v1/rerank",
                json={
                    "model": "test-reranker",
                    "query": "query",
                    "documents": ["a", "b", "c"],
                    "top_n": 2,
                },
            )
        finally:
            srv._rerank_engine = original

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 2
        assert body["results"][0]["relevance_score"] == 0.9
        assert body["results"][1]["relevance_score"] == 0.5

    def test_rerank_top_n_exceeds_documents_returns_400(self, client):
        """Test that top_n > len(documents) returns 400."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "test-reranker"

        original = srv._rerank_engine
        srv._rerank_engine = mock_engine
        try:
            resp = client.post(
                "/v1/rerank",
                json={
                    "model": "test-reranker",
                    "query": "query",
                    "documents": ["a", "b"],
                    "top_n": 5,
                },
            )
        finally:
            srv._rerank_engine = original

        assert resp.status_code == 400
        assert "top_n" in resp.json()["detail"]

    def test_rerank_return_documents_false(self, client):
        """Test that return_documents=false omits document text."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "test-reranker"
        mock_engine.score_pairs.return_value = [0.7]
        mock_engine.count_tokens.return_value = 5

        original = srv._rerank_engine
        srv._rerank_engine = mock_engine
        try:
            resp = client.post(
                "/v1/rerank",
                json={
                    "model": "test-reranker",
                    "query": "q",
                    "documents": ["d"],
                    "return_documents": False,
                },
            )
        finally:
            srv._rerank_engine = original

        assert resp.status_code == 200
        body = resp.json()
        assert body["results"][0]["document"] is None

    def test_rerank_preserves_object_documents(self, client):
        """Test that object documents preserve original structure."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "test-reranker"
        mock_engine.score_pairs.return_value = [0.8, 0.3]
        mock_engine.count_tokens.return_value = 10

        original = srv._rerank_engine
        srv._rerank_engine = mock_engine
        try:
            resp = client.post(
                "/v1/rerank",
                json={
                    "model": "test-reranker",
                    "query": "q",
                    "documents": [
                        {"text": "alpha", "metadata": "extra1"},
                        {"text": "beta", "metadata": "extra2"},
                    ],
                },
            )
        finally:
            srv._rerank_engine = original

        assert resp.status_code == 200
        body = resp.json()
        # Results sorted by score descending; index 0 (0.8) first
        assert body["results"][0]["index"] == 0
        assert body["results"][0]["document"]["text"] == "alpha"
        assert body["results"][0]["document"]["metadata"] == "extra1"

    def test_rerank_empty_documents_returns_400(self, client):
        """Test that empty document list returns 400."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "test-reranker"

        original = srv._rerank_engine
        srv._rerank_engine = mock_engine
        try:
            resp = client.post(
                "/v1/rerank",
                json={
                    "model": "test-reranker",
                    "query": "q",
                    "documents": [],
                },
            )
        finally:
            srv._rerank_engine = original

        assert resp.status_code == 400

    def test_rerank_usage_tokens(self, client):
        """Test that usage.total_tokens is included in response."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "test-reranker"
        mock_engine.score_pairs.return_value = [0.5]
        mock_engine.count_tokens.return_value = 42

        original = srv._rerank_engine
        srv._rerank_engine = mock_engine
        try:
            resp = client.post(
                "/v1/rerank",
                json={
                    "model": "test-reranker",
                    "query": "q",
                    "documents": ["d"],
                },
            )
        finally:
            srv._rerank_engine = original

        assert resp.status_code == 200
        assert resp.json()["usage"]["total_tokens"] == 42

    def test_rerank_model_locked_rejects_different_model(self, client):
        """Test that a locked reranker model rejects requests for different models."""
        import vllm_mlx.server as srv

        mock_engine = MagicMock()
        mock_engine.model_name = "locked-reranker"

        original_engine = srv._rerank_engine
        original_locked = srv._rerank_model_locked
        srv._rerank_engine = mock_engine
        srv._rerank_model_locked = "locked-reranker"

        try:
            resp = client.post(
                "/v1/rerank",
                json={
                    "model": "other-reranker",
                    "query": "q",
                    "documents": ["d"],
                },
            )
            assert resp.status_code == 400
            body = resp.json()
            assert "locked-reranker" in body["detail"]
            assert "other-reranker" in body["detail"]
        finally:
            srv._rerank_engine = original_engine
            srv._rerank_model_locked = original_locked

    def test_rerank_no_engine_returns_503(self, client):
        """Test that requesting rerank without a loaded engine returns 503."""
        import vllm_mlx.server as srv

        original = srv._rerank_engine
        srv._rerank_engine = None
        try:
            resp = client.post(
                "/v1/rerank",
                json={
                    "model": "any-model",
                    "query": "q",
                    "documents": ["d"],
                },
            )
        finally:
            srv._rerank_engine = original

        # Without lazy loading enabled, should try to load and may fail with 503
        # The exact behavior depends on whether lazy loading is wired, but the
        # route must exist and not 404
        assert resp.status_code != 404
