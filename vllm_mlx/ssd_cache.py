# SPDX-License-Identifier: Apache-2.0
"""
SSD KV cache tiering for vllm-mlx.

This module provides a cold-tier disk cache that sits behind
MemoryAwarePrefixCache. Evicted entries spill to NVMe instead of being
discarded, and cold-tier fetches reload from disk asynchronously with
RAM budget reservation before the read completes.

Key design:
- SQLite for atomic metadata index (no mutable JSON)
- Async writer thread for non-blocking spills
- Per-layer serializer interface for hybrid cache types
- Atomic temp-file + rename writes for crash consistency
- Metrics exposed from day one
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024
_BYTES_PER_GB = 1024 * 1024 * 1024


@dataclass(frozen=True)
class SSDCacheConfig:
    """Configuration for SSD cache tier.

    Attributes:
        cache_dir: Directory for SSD cache files. None = auto-detect
            (~/.cache/vllm-mlx/ssd_cache/{model}/).
        max_size_gb: Maximum total size of SSD cache in GB.
        max_entries: Maximum number of entries in SSD cache.
        file_permissions: Unix permission bits for cache data files.
        dir_permissions: Unix permission bits for cache directories.
        spill_queue_size: Max pending spill operations before dropping.
        retention_seconds: Optional max age for cache entries (None = no expiry).
    """

    cache_dir: str | None = None
    max_size_gb: float = 10.0
    max_entries: int = 10000
    file_permissions: int = 0o600
    dir_permissions: int = 0o700
    spill_queue_size: int = 64
    retention_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.max_size_gb <= 0:
            raise ValueError(f"max_size_gb must be > 0, got {self.max_size_gb}")
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.spill_queue_size < 1:
            raise ValueError(
                f"spill_queue_size must be >= 1, got {self.spill_queue_size}"
            )

    @property
    def max_size_bytes(self) -> int:
        """Maximum cache size in bytes."""
        return int(self.max_size_gb * _BYTES_PER_GB)


@dataclass
class SSDCacheStats:
    """Statistics for SSD cache tier — exposed from day one.

    Attributes:
        spill_count: Number of entries spilled to SSD.
        spill_bytes: Total bytes written to SSD.
        ssd_hits: Number of successful SSD cache lookups.
        ssd_misses: Number of SSD cache lookup misses.
        reload_latency_sum: Sum of reload latencies in seconds.
        reload_bytes: Total bytes read from SSD.
        promotion_failures: Number of failed promotions (RAM budget exhausted).
    """

    spill_count: int = 0
    spill_bytes: int = 0
    ssd_hits: int = 0
    ssd_misses: int = 0
    reload_latency_sum: float = 0.0
    reload_bytes: int = 0
    promotion_failures: int = 0

    def to_dict(self) -> dict:
        total_lookups = self.ssd_hits + self.ssd_misses
        hit_rate = self.ssd_hits / total_lookups if total_lookups > 0 else 0.0
        avg_latency_ms = (
            (self.reload_latency_sum / self.ssd_hits * 1000)
            if self.ssd_hits > 0
            else 0.0
        )
        return {
            "spill_count": self.spill_count,
            "spill_bytes": self.spill_bytes,
            "ssd_hits": self.ssd_hits,
            "ssd_misses": self.ssd_misses,
            "ssd_hit_rate": round(hit_rate, 4),
            "reload_latency_sum_s": round(self.reload_latency_sum, 4),
            "avg_reload_latency_ms": round(avg_latency_ms, 2),
            "reload_bytes": self.reload_bytes,
            "promotion_failures": self.promotion_failures,
        }
