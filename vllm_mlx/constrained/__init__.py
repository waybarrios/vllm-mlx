# SPDX-License-Identifier: Apache-2.0
"""
Constrained decoding for grammar-guided generation.

Provides logits processors that mask token probabilities during generation
so the model can only emit sequences matching a target grammar (e.g. a JSON
schema).  Used by the ``response_format`` parameter on the chat completion
and Anthropic Messages endpoints.
"""

from .json_schema_processor import (
    JSONSchemaLogitsProcessor,
    LMFormatEnforcerNotAvailableError,
    is_available,
)

__all__ = [
    "JSONSchemaLogitsProcessor",
    "LMFormatEnforcerNotAvailableError",
    "is_available",
]
