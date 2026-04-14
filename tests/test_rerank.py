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
