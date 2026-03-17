"""Tests for message role normalization."""


class TestNormalizeMessageRoles:
    """Test _normalize_message_roles helper."""

    def test_developer_mapped_to_system(self):
        """developer role should be mapped to system."""
        from vllm_mlx.server import _normalize_message_roles

        messages = [
            {"role": "developer", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        result = _normalize_message_roles(messages)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant"
        assert result[1]["role"] == "user"

    def test_standard_roles_unchanged(self):
        """system, user, assistant, tool roles should pass through unchanged."""
        from vllm_mlx.server import _normalize_message_roles

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "tool", "content": "result"},
        ]
        result = _normalize_message_roles(messages)
        assert [m["role"] for m in result] == ["system", "user", "assistant", "tool"]

    def test_empty_messages(self):
        """Empty list should return empty list."""
        from vllm_mlx.server import _normalize_message_roles

        assert _normalize_message_roles([]) == []

    def test_does_not_mutate_input(self):
        """Input list and dicts should not be modified."""
        from vllm_mlx.server import _normalize_message_roles

        original = [{"role": "developer", "content": "test"}]
        result = _normalize_message_roles(original)
        assert original[0]["role"] == "developer"  # unchanged
        assert result[0]["role"] == "system"  # mapped copy

    def test_preserves_all_fields(self):
        """All message fields besides role should be preserved."""
        from vllm_mlx.server import _normalize_message_roles

        messages = [
            {"role": "developer", "content": "test", "name": "dev", "extra": 42},
        ]
        result = _normalize_message_roles(messages)
        assert result[0] == {
            "role": "system",
            "content": "test",
            "name": "dev",
            "extra": 42,
        }

    def test_multimodal_content_preserved(self):
        """List content (multimodal) should be preserved as-is."""
        from vllm_mlx.server import _normalize_message_roles

        content = [
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        messages = [{"role": "developer", "content": content}]
        result = _normalize_message_roles(messages)
        assert result[0]["role"] == "system"
        assert result[0]["content"] is content
