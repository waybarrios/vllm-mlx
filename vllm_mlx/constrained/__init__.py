# SPDX-License-Identifier: Apache-2.0
"""
Constrained decoding for grammar-guided generation.

Provides logits processors that mask token probabilities during generation
so the model can only emit sequences matching a target grammar (e.g. a JSON
schema).  Used by the ``response_format`` parameter on the chat completion
and Anthropic Messages endpoints.
"""

from .json_schema_processor import (
    ConstrainedDecodingError,
    JSONSchemaLogitsProcessor,
    LMFormatEnforcerNotAvailableError,
    UnsupportedJSONSchemaError,
    is_available,
)
from .llguidance_schema_processor import (
    LLGuidanceJSONSchemaLogitsProcessor,
    is_available as is_strict_json_schema_available,
)
from .thinking_processor import ThinkingAwareLogitsProcessor

__all__ = [
    "ConstrainedDecodingError",
    "JSONSchemaLogitsProcessor",
    "LLGuidanceJSONSchemaLogitsProcessor",
    "LMFormatEnforcerNotAvailableError",
    "ThinkingAwareLogitsProcessor",
    "UnsupportedJSONSchemaError",
    "is_available",
    "is_strict_json_schema_available",
]
