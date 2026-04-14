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

import array as _array
import hashlib
import logging
import os
import sqlite3
import time
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


def _tokens_to_blob(tokens: tuple[int, ...]) -> bytes:
    """Serialize token tuple to a compact binary blob for SQLite storage.

    Uses the full token sequence as a binary blob for prefix matching.
    """
    arr = _array.array("i", tokens)
    return arr.tobytes()


def _blob_to_tokens(blob: bytes) -> tuple[int, ...]:
    """Deserialize binary blob back to token tuple."""
    arr = _array.array("i")
    arr.frombytes(blob)
    return tuple(arr)


def _tokens_hash(tokens: tuple[int, ...]) -> str:
    """Compute SHA-256 hex digest of a token sequence for use as primary key."""
    return hashlib.sha256(_tokens_to_blob(tokens)).hexdigest()


class SSDIndex:
    """SQLite-backed index for SSD cache entries.

    Uses SQLite for atomic metadata operations instead of mutable JSON.
    The token sequence is stored as a binary blob for prefix-searchable
    representation. The primary key is a SHA-256 hash of the token sequence.

    Thread safety: SQLite in WAL mode with serialized threading. Individual
    operations are atomic. Callers must not share a single SSDIndex instance
    across threads without external synchronization.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir
        db_path = os.path.join(cache_dir, "index.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entries (
                token_hash   TEXT PRIMARY KEY,
                tokens_blob  BLOB NOT NULL,
                num_tokens   INTEGER NOT NULL,
                file_path    TEXT NOT NULL,
                memory_bytes INTEGER NOT NULL,
                created_at   REAL NOT NULL,
                accessed_at  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_entries_accessed
                ON entries(accessed_at);

            CREATE INDEX IF NOT EXISTS idx_entries_num_tokens
                ON entries(num_tokens);
            """)
        # Insert schema version if not present
        cur = self._conn.execute("SELECT COUNT(*) FROM schema_version")
        if cur.fetchone()[0] == 0:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (self._SCHEMA_VERSION,),
            )
        self._conn.commit()

    def insert_entry(
        self,
        tokens_key: tuple[int, ...],
        file_path: str,
        memory_bytes: int,
        num_tokens: int,
    ) -> None:
        """Insert or replace a cache entry in the index."""
        now = time.time()
        token_hash = _tokens_hash(tokens_key)
        tokens_blob = _tokens_to_blob(tokens_key)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO entries
                (token_hash, tokens_blob, num_tokens, file_path,
                 memory_bytes, created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (token_hash, tokens_blob, num_tokens, file_path, memory_bytes, now, now),
        )
        self._conn.commit()

    def lookup_exact(self, tokens_key: tuple[int, ...]) -> dict | None:
        """Look up an exact token sequence. Returns dict or None."""
        token_hash = _tokens_hash(tokens_key)
        cur = self._conn.execute(
            "SELECT file_path, memory_bytes, num_tokens FROM entries WHERE token_hash = ?",
            (token_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "file_path": row["file_path"],
            "memory_bytes": row["memory_bytes"],
            "num_tokens": row["num_tokens"],
        }

    def lookup_prefix(self, query_tokens: tuple[int, ...]) -> list[dict]:
        """Find entries whose token sequence is a prefix of query_tokens.

        Scans entries with num_tokens <= len(query_tokens) and compares the
        full stored token blob against the corresponding prefix of query_tokens.

        Returns list of dicts sorted by num_tokens descending (longest prefix first).
        """
        query_len = len(query_tokens)
        query_blob = _tokens_to_blob(query_tokens)

        cur = self._conn.execute(
            "SELECT token_hash, tokens_blob, num_tokens, file_path, memory_bytes "
            "FROM entries WHERE num_tokens <= ? ORDER BY num_tokens DESC",
            (query_len,),
        )

        results = []
        for row in cur:
            stored_blob = row["tokens_blob"]
            n = row["num_tokens"]
            # Compare: the stored blob must equal the first n tokens of the query
            # Each int is 4 bytes in the array('i') encoding
            prefix_blob = query_blob[: n * 4]
            if stored_blob == prefix_blob:
                results.append(
                    {
                        "token_hash": row["token_hash"],
                        "file_path": row["file_path"],
                        "memory_bytes": row["memory_bytes"],
                        "num_tokens": n,
                    }
                )
        return results

    def delete_entry(self, tokens_key: tuple[int, ...]) -> None:
        """Delete an entry by token sequence."""
        token_hash = _tokens_hash(tokens_key)
        self._conn.execute("DELETE FROM entries WHERE token_hash = ?", (token_hash,))
        self._conn.commit()

    def get_lru(self, limit: int = 10) -> list[dict]:
        """Get the least recently used entries, ordered oldest first."""
        cur = self._conn.execute(
            "SELECT token_hash, tokens_blob, num_tokens, file_path, memory_bytes "
            "FROM entries ORDER BY accessed_at ASC LIMIT ?",
            (limit,),
        )
        results = []
        for row in cur:
            results.append(
                {
                    "token_hash": row["token_hash"],
                    "tokens_blob": row["tokens_blob"],
                    "file_path": row["file_path"],
                    "memory_bytes": row["memory_bytes"],
                    "num_tokens": row["num_tokens"],
                }
            )
        return results

    def get_total_bytes(self) -> int:
        """Get total memory_bytes across all entries."""
        cur = self._conn.execute("SELECT COALESCE(SUM(memory_bytes), 0) FROM entries")
        return cur.fetchone()[0]

    def get_entry_count(self) -> int:
        """Get number of entries in the index."""
        cur = self._conn.execute("SELECT COUNT(*) FROM entries")
        return cur.fetchone()[0]

    def touch(self, tokens_key: tuple[int, ...]) -> None:
        """Update accessed_at timestamp for an entry (marks as recently used)."""
        token_hash = _tokens_hash(tokens_key)
        self._conn.execute(
            "UPDATE entries SET accessed_at = ? WHERE token_hash = ?",
            (time.time(), token_hash),
        )
        self._conn.commit()

    def all_entries(self) -> list[dict]:
        """Return all entries (for startup reconciliation)."""
        cur = self._conn.execute(
            "SELECT token_hash, tokens_blob, num_tokens, file_path, memory_bytes "
            "FROM entries ORDER BY accessed_at DESC"
        )
        results = []
        for row in cur:
            results.append(
                {
                    "token_hash": row["token_hash"],
                    "tokens_blob": row["tokens_blob"],
                    "file_path": row["file_path"],
                    "memory_bytes": row["memory_bytes"],
                    "num_tokens": row["num_tokens"],
                }
            )
        return results

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
