# SPDX-License-Identifier: Apache-2.0
"""Tests for models with no chat_template (e.g., MedGemma).

MedGemma's HuggingFace processor has apply_chat_template() as a method
(inherited from base class) but no chat_template configured. Calling
apply_chat_template() raises ValueError. vllm-mlx should fall back to
a plain-text prompt format instead of crashing.
"""

import pytest


class FakeProcessorNoTemplate:
    """Simulates a HuggingFace processor with no chat_template set."""

    chat_template = None  # No template configured

    def apply_chat_template(self, messages, **kwargs):
        """Raises like the real processor does when no template is set."""
        raise ValueError(
            "Cannot use apply_chat_template when no chat_template is set. "
            "Provide a chat_template or use a model with one."
        )


class FakeProcessorWithTemplate:
    """Simulates a working processor for comparison."""

    chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


class TestBatchedEngineChatTemplateFallback:
    """Test that _apply_chat_template handles missing templates gracefully."""

    def _make_engine_stub(self, processor, is_mllm=True):
        """Create a minimal stub with the fields _apply_chat_template needs."""
        from vllm_mlx.engine.batched import BatchedEngine

        # Create a bare object that has the method but not the full engine
        stub = object.__new__(BatchedEngine)
        stub._is_mllm = is_mllm
        stub._processor = processor
        stub._model_name = "test-model"
        # _apply_chat_template falls through to tokenizer if processor fails
        # Give it a tokenizer that also has no template
        stub._tokenizer = None
        return stub

    def test_no_template_processor_does_not_crash(self):
        """A processor with no chat_template should fall back, not raise."""
        processor = FakeProcessorNoTemplate()
        stub = self._make_engine_stub(processor)
        messages = [{"role": "user", "content": "What is this image?"}]

        # This should NOT raise — it should fall back gracefully
        result = stub._apply_chat_template(messages)
        assert isinstance(result, str)
        assert "What is this image?" in result

    def test_no_template_produces_readable_prompt(self):
        """Fallback prompt should include role and content."""
        processor = FakeProcessorNoTemplate()
        stub = self._make_engine_stub(processor)
        messages = [
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": "Describe this X-ray."},
        ]

        result = stub._apply_chat_template(messages)
        assert "medical assistant" in result
        assert "X-ray" in result

    def test_working_processor_still_works(self):
        """A processor WITH a template should still use it normally."""
        processor = FakeProcessorWithTemplate()
        stub = self._make_engine_stub(processor)
        messages = [{"role": "user", "content": "Hello"}]

        result = stub._apply_chat_template(messages)
        assert "Hello" in result

    def test_no_template_with_tools_does_not_crash(self):
        """Missing template + tools should also fall back gracefully."""
        processor = FakeProcessorNoTemplate()
        stub = self._make_engine_stub(processor)
        messages = [{"role": "user", "content": "Check vitals"}]
        tools = [{"type": "function", "function": {"name": "get_vitals"}}]

        result = stub._apply_chat_template(messages, tools=tools)
        assert isinstance(result, str)
        assert "Check vitals" in result
