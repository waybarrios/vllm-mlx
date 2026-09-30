# SPDX-License-Identifier: Apache-2.0
"""Total-context admission limits for generation requests."""

from typing import Any

DEFAULT_MAX_MODEL_LEN = 65_536


def encode_prompt(tokenizer: Any, prompt: str) -> Any:
    """Encode a prompt through the same tokenizer seam used by admission."""
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        return encode(prompt)
    nested = getattr(tokenizer, "tokenizer", None)
    nested_encode = getattr(nested, "encode", None)
    if callable(nested_encode):
        return nested_encode(prompt)
    raise RuntimeError("Engine tokenizer is unavailable for context admission")


class ContextLengthExceeded(ValueError):
    """Raised when prompt and requested output exceed the configured context."""

    code = "context_length_exceeded"

    def __init__(
        self,
        *,
        prompt_tokens: int,
        max_tokens: int,
        max_model_len: int,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.total_tokens = prompt_tokens + max_tokens
        super().__init__(
            f"Request requires {prompt_tokens} prompt tokens plus {max_tokens} "
            f"output tokens, exceeding max_model_len={max_model_len}"
        )


def validate_context_length(
    *,
    prompt_tokens: int,
    max_tokens: int,
    max_model_len: int,
) -> None:
    """Reject a generation request that cannot fit in the configured context."""
    if prompt_tokens + max_tokens > max_model_len:
        raise ContextLengthExceeded(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
        )
