# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleEngine prompt prefix caching."""

from unittest.mock import MagicMock, patch

import pytest


def _make_stream_chunks(texts, finish_on_last=True):
    """Create mock stream_generate chunks."""
    chunks = []
    for i, text in enumerate(texts):
        chunk = MagicMock()
        chunk.text = text
        chunk.finished = finish_on_last and (i == len(texts) - 1)
        chunk.finish_reason = "stop" if chunk.finished else None
        chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_model():
    """Create a mock LLM model with tokenizer."""
    model = MagicMock()
    model.model = MagicMock()  # The underlying nn.Module for make_prompt_cache
    model.tokenizer = MagicMock()
    model.tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])

    chunks = _make_stream_chunks(["hello", " world"])
    model.stream_generate = MagicMock(return_value=iter(chunks))
    return model


class TestPrefixCacheIntegration:
    """Test prefix cache behavior in SimpleEngine.stream_generate."""

    @pytest.mark.asyncio
    async def test_cache_miss_on_first_request(self, mock_model):
        """First request should miss cache and create a fresh one."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.simple.make_prompt_cache") as mock_make_cache:
            mock_make_cache.return_value = [MagicMock()]

            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            results = []
            async for chunk in engine.stream_generate(prompt="test", max_tokens=10):
                results.append(chunk)

            assert len(results) == 2
            # make_prompt_cache should have been called (cache miss)
            mock_make_cache.assert_called_once_with(mock_model.model)

    @pytest.mark.asyncio
    async def test_cache_disabled_when_size_zero(self, mock_model):
        """prompt_cache_size=0 should disable caching entirely."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model", prompt_cache_size=0)
            engine._model = mock_model
            engine._loaded = True

            assert engine._prompt_cache is None

            results = []
            async for chunk in engine.stream_generate(prompt="test", max_tokens=10):
                results.append(chunk)

            assert len(results) == 2
            # stream_generate should have been called with the raw prompt string
            call_kwargs = mock_model.stream_generate.call_args
            assert call_kwargs.kwargs.get("prompt") == "test" or call_kwargs[1].get("prompt") == "test"
            # No prompt_cache kwarg should be passed
            assert "prompt_cache" not in (call_kwargs.kwargs or call_kwargs[1])

    @pytest.mark.asyncio
    async def test_stop_clears_cache(self, mock_model):
        """Stopping the engine should clear the prompt cache."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model", prompt_cache_size=10)
            engine._model = mock_model
            engine._loaded = True

            old_cache = engine._prompt_cache
            await engine.stop()
            # Cache should be a new instance after stop
            assert engine._prompt_cache is not old_cache

    @pytest.mark.asyncio
    async def test_configurable_cache_size(self):
        """prompt_cache_size should be passed through to LRUPromptCache."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False), \
             patch("vllm_mlx.engine.simple.LRUPromptCache") as mock_lru:
            engine = SimpleEngine("test-model", prompt_cache_size=7)
            mock_lru.assert_called_once_with(max_size=7)


class TestBillingHeaderStripping:
    """Test that x-anthropic- tracking headers are stripped from system blocks."""

    def test_billing_header_stripped(self):
        """System blocks starting with x-anthropic- should be dropped."""
        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
        from vllm_mlx.api.anthropic_models import AnthropicMessage, AnthropicRequest

        req = AnthropicRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=100,
            system=[
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.42; cch=abc123"},
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        )
        result = anthropic_to_openai(req)
        system_msg = result.messages[0]
        assert system_msg.role == "system"
        assert "x-anthropic" not in system_msg.content
        assert system_msg.content == "You are a helpful assistant."

    def test_non_anthropic_headers_preserved(self):
        """System blocks not starting with x-anthropic- should be kept."""
        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
        from vllm_mlx.api.anthropic_models import AnthropicMessage, AnthropicRequest

        req = AnthropicRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=100,
            system=[
                {"type": "text", "text": "Be helpful."},
                {"type": "text", "text": "Be concise."},
            ],
        )
        result = anthropic_to_openai(req)
        system_msg = result.messages[0]
        assert system_msg.content == "Be helpful.\nBe concise."

    def test_all_anthropic_headers_stripped(self):
        """If ALL system blocks are x-anthropic-, system should be empty."""
        from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
        from vllm_mlx.api.anthropic_models import AnthropicMessage, AnthropicRequest

        req = AnthropicRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=100,
            system=[
                {"type": "text", "text": "x-anthropic-billing-header: cch=abc"},
                {"type": "text", "text": "x-anthropic-other: foo"},
            ],
        )
        result = anthropic_to_openai(req)
        # System message should still exist but be empty
        system_msg = result.messages[0]
        assert system_msg.role == "system"
        assert system_msg.content == ""
