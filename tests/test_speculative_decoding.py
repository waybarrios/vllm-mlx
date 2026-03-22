# SPDX-License-Identifier: Apache-2.0
"""Tests for speculative decoding with a separate draft model (SimpleEngine path)."""

import pytest

try:
    import mlx.core as mx  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


# ---------------------------------------------------------------------------
# Tests: CLI args
# ---------------------------------------------------------------------------


class TestCLIArgs:
    def test_draft_model_arg_parsing(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--draft-model", type=str, default=None)
        parser.add_argument("--num-draft-tokens", type=int, default=3)

        args = parser.parse_args(
            ["--draft-model", "/path/to/model", "--num-draft-tokens", "5"]
        )
        assert args.draft_model == "/path/to/model"
        assert args.num_draft_tokens == 5

    def test_default_num_draft_tokens(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--num-draft-tokens", type=int, default=3)

        args = parser.parse_args([])
        assert args.num_draft_tokens == 3


# ---------------------------------------------------------------------------
# Tests: SimpleEngine draft model
# ---------------------------------------------------------------------------


class TestSimpleEngineDraftModel:
    def test_draft_model_params_stored(self):
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(
            model_name="test-model",
            draft_model_path="/path/to/draft",
            num_draft_tokens=5,
        )
        assert engine._draft_model_path == "/path/to/draft"
        assert engine._num_draft_tokens == 5

    def test_no_draft_model_by_default(self):
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(model_name="test-model")
        assert engine._draft_model_path is None
        assert engine._num_draft_tokens == 3


class TestMLXLanguageModelDraftModel:
    def test_draft_model_params_stored(self):
        from vllm_mlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel(
            model_name="test-model",
            draft_model_path="/path/to/draft",
            num_draft_tokens=5,
        )
        assert model._draft_model_path == "/path/to/draft"
        assert model._num_draft_tokens == 5
        assert model.draft_model is None

    def test_no_draft_model_by_default(self):
        from vllm_mlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel(model_name="test-model")
        assert model._draft_model_path is None
        assert model.draft_model is None
