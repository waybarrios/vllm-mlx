# SPDX-License-Identifier: Apache-2.0
"""MTP adapter registry mapping model families to MTP module implementations."""

from .deepseek_v3_mtp import DeepSeekV3MTPModule, StandardMTPModule
from .simple_mtp import SimpleMTPModule

MTP_ADAPTER_REGISTRY: dict[str, type] = {
    "standard": StandardMTPModule,
    "simple": SimpleMTPModule,
    # Backward compatibility aliases
    "deepseek_v3": StandardMTPModule,
}

__all__ = [
    "MTP_ADAPTER_REGISTRY",
    "StandardMTPModule",
    "SimpleMTPModule",
    "DeepSeekV3MTPModule",
]
