"""Capability normalization for external speculative drafters."""

from typing import Any


def explicit_bool(value: Any) -> bool | None:
    """Return only literal Boolean capability declarations."""
    return value if type(value) is bool else None


def continuous_batching_capability(drafter: Any) -> bool | None:
    """Return an external drafter's explicit CB capability, if declared."""
    return explicit_bool(getattr(drafter, "supports_continuous_batching", None))
