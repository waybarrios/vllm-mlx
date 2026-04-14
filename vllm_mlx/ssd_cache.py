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
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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


# Support matrix: maps cache type names to their serializer status
SERIALIZER_SUPPORT_MATRIX = {
    "KVCache": "supported",
    "RotatingKVCache": "supported",  # Serialized as KVCache (keys/values/offset)
    "ArraysCache": "supported",
    "MambaCache": "supported",  # Legacy name for ArraysCache
    "_QuantizedCacheWrapper": "not_supported_spill_dequantized",
}


class LayerSerializer(ABC):
    """Interface for per-layer cache serialization.

    Each implementation handles a specific cache type's serialization
    to/from safetensors files with metadata.
    """

    @abstractmethod
    def serialize_layer(
        self, layer: Any, layer_idx: int, file_path: str
    ) -> dict[str, Any]:
        """Serialize a single cache layer to a file.

        Args:
            layer: The cache layer object.
            layer_idx: Index of this layer in the cache list.
            file_path: Path to write the safetensors file.

        Returns:
            Metadata dict with at least 'layer_type' key.
        """
        ...

    @abstractmethod
    def deserialize_layer(self, file_path: str, metadata: dict[str, Any]) -> dict:
        """Deserialize a single cache layer from a file.

        Args:
            file_path: Path to the safetensors file.
            metadata: Metadata dict from serialize_layer.

        Returns:
            Dict with layer state (keys/values/offset or state list).
        """
        ...


class KVCacheSerializer(LayerSerializer):
    """Serializer for KVCache and RotatingKVCache layers.

    Handles layers with .keys, .values, .offset attributes.
    RotatingKVCache also has .max_size, .keep, .step, ._idx.
    """

    def serialize_layer(
        self, layer: Any, layer_idx: int, file_path: str
    ) -> dict[str, Any]:
        from safetensors.numpy import save_file

        keys_np = np.array(layer.keys)
        values_np = np.array(layer.values)

        tensors = {
            f"layer_{layer_idx}_keys": keys_np,
            f"layer_{layer_idx}_values": values_np,
        }
        save_file(tensors, file_path)

        metadata = {
            "layer_type": "KVCache",
            "layer_idx": layer_idx,
            "offset": layer.offset,
        }
        # Preserve RotatingKVCache extra attributes
        for attr in ("max_size", "keep", "step", "_idx"):
            if hasattr(layer, attr):
                metadata[attr] = getattr(layer, attr)

        return metadata

    def deserialize_layer(self, file_path: str, metadata: dict[str, Any]) -> dict:
        from safetensors.numpy import load_file

        layer_idx = metadata["layer_idx"]
        tensors = load_file(file_path)

        result = {
            "keys": tensors[f"layer_{layer_idx}_keys"],
            "values": tensors[f"layer_{layer_idx}_values"],
            "offset": metadata["offset"],
        }
        for attr in ("max_size", "keep", "step", "_idx"):
            if attr in metadata:
                result[attr] = metadata[attr]
        return result


class ArraysCacheSerializer(LayerSerializer):
    """Serializer for ArraysCache (Mamba/linear attention) layers.

    Handles layers with .state attribute containing a list of arrays.
    """

    def serialize_layer(
        self, layer: Any, layer_idx: int, file_path: str
    ) -> dict[str, Any]:
        from safetensors.numpy import save_file

        state = layer.state
        tensors = {}
        for i, arr in enumerate(state):
            tensors[f"layer_{layer_idx}_state_{i}"] = np.array(arr)

        save_file(tensors, file_path)

        return {
            "layer_type": "ArraysCache",
            "layer_idx": layer_idx,
            "num_arrays": len(state),
        }

    def deserialize_layer(self, file_path: str, metadata: dict[str, Any]) -> dict:
        from safetensors.numpy import load_file

        layer_idx = metadata["layer_idx"]
        num_arrays = metadata["num_arrays"]
        tensors = load_file(file_path)

        state = []
        for i in range(num_arrays):
            state.append(tensors[f"layer_{layer_idx}_state_{i}"])
        return {"state": state}


def get_serializer_for_layer(layer: Any) -> LayerSerializer:
    """Return the appropriate serializer for a cache layer.

    Dispatches based on duck-typing:
    - If layer has .keys and .values and .offset -> KVCacheSerializer
    - If layer has .state and it's a list -> ArraysCacheSerializer

    Raises ValueError for unsupported layer types.
    """
    if hasattr(layer, "keys") and hasattr(layer, "values") and hasattr(layer, "offset"):
        return KVCacheSerializer()
    if hasattr(layer, "state") and isinstance(getattr(layer, "state", None), list):
        return ArraysCacheSerializer()
    raise ValueError(
        f"Unsupported cache layer type: {type(layer).__name__}. "
        f"Supported: {list(SERIALIZER_SUPPORT_MATRIX.keys())}"
    )


class SSDCacheTier:
    """Cold-tier disk cache for KV cache entries.

    Manages a SQLite-indexed on-disk cache directory. Evicted RAM entries
    are spilled here via an async writer thread. Cold-tier fetches reload
    from disk asynchronously with RAM budget reservation.

    Directory layout::

        cache_dir/
          index.db           # SQLite metadata index
          data/              # safetensors files per entry
            {hash}/          # one directory per entry
              layer_0.safetensors
              layer_1.safetensors
              manifest.json  # per-entry layer metadata
    """

    def __init__(self, config: SSDCacheConfig) -> None:
        self._config = config

        if config.cache_dir is None:
            raise ValueError("SSDCacheConfig.cache_dir must be set")

        self._cache_dir = config.cache_dir
        self._data_dir = os.path.join(self._cache_dir, "data")

        # Create directory structure
        os.makedirs(self._cache_dir, mode=config.dir_permissions, exist_ok=True)
        os.makedirs(self._data_dir, mode=config.dir_permissions, exist_ok=True)

        # Open SQLite index
        self._index = SSDIndex(self._cache_dir)

        # Stats
        self._stats = SSDCacheStats()
        self._lock = threading.Lock()
        self._closed = False

        # Spill queue and writer thread
        self._spill_queue: queue.Queue = queue.Queue(maxsize=config.spill_queue_size)
        self._writer_thread: threading.Thread | None = None
        self._writer_stop = threading.Event()

    @staticmethod
    def _entry_hash(tokens: tuple[int, ...]) -> str:
        """Compute deterministic hash for a token sequence."""
        return _tokens_hash(tokens)

    def get_stats(self) -> dict:
        """Return current SSD cache statistics."""
        return self._stats.to_dict()

    def start_writer(self) -> None:
        """Start the background spill writer thread."""
        if self._writer_thread is not None:
            return
        self._writer_stop.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="ssd-cache-writer"
        )
        self._writer_thread.start()
        logger.info("[ssd_cache] writer thread started")

    def _writer_loop(self) -> None:
        """Background loop: drain spill queue and write to disk."""
        while not self._writer_stop.is_set():
            try:
                item = self._spill_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:  # Poison pill for shutdown
                break

            tokens_key, cache_layers, memory_bytes = item
            try:
                self._write_entry(tokens_key, cache_layers, memory_bytes)
            except Exception:
                logger.exception(
                    f"[ssd_cache] failed to write entry " f"({len(tokens_key)} tokens)"
                )

    def enqueue_spill(
        self,
        tokens: tuple[int, ...],
        cache: list[Any],
        memory_bytes: int,
    ) -> bool:
        """Enqueue a cache entry for async spill to SSD.

        Returns True if enqueued, False if queue is full (entry dropped).
        """
        try:
            self._spill_queue.put_nowait((tokens, cache, memory_bytes))
            return True
        except queue.Full:
            logger.warning(
                f"[ssd_cache] spill queue full, dropping entry "
                f"({len(tokens)} tokens, {memory_bytes} bytes)"
            )
            return False

    def _write_entry(
        self,
        tokens_key: tuple[int, ...],
        cache_layers: list[Any],
        memory_bytes: int,
    ) -> None:
        """Write a single cache entry to disk atomically.

        Uses temp-file + rename for crash consistency.
        """
        import shutil

        entry_hash = self._entry_hash(tokens_key)
        entry_dir = os.path.join(self._data_dir, entry_hash)
        tmp_dir = entry_dir + ".tmp"

        # Clean up any leftover tmp dir from a previous crash
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        os.makedirs(tmp_dir, mode=self._config.dir_permissions, exist_ok=True)

        layer_manifests = []
        total_file_bytes = 0

        for i, layer in enumerate(cache_layers):
            serializer = get_serializer_for_layer(layer)
            layer_path = os.path.join(tmp_dir, f"layer_{i}.safetensors")
            metadata = serializer.serialize_layer(layer, i, layer_path)
            layer_manifests.append(metadata)

            # Set file permissions
            os.chmod(layer_path, self._config.file_permissions)
            total_file_bytes += os.path.getsize(layer_path)

        # Write manifest
        manifest = {
            "num_layers": len(cache_layers),
            "layers": layer_manifests,
            "memory_bytes": memory_bytes,
            "num_tokens": len(tokens_key),
        }
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        os.chmod(manifest_path, self._config.file_permissions)

        # Save tokens binary
        tokens_path = os.path.join(tmp_dir, "tokens.bin")
        arr = _array.array("i", tokens_key)
        with open(tokens_path, "wb") as f:
            arr.tofile(f)
        os.chmod(tokens_path, self._config.file_permissions)

        # Atomic rename: tmp_dir -> entry_dir
        if os.path.exists(entry_dir):
            shutil.rmtree(entry_dir)
        os.rename(tmp_dir, entry_dir)

        # Update index
        relative_path = entry_hash
        self._index.insert_entry(
            tokens_key=tokens_key,
            file_path=relative_path,
            memory_bytes=memory_bytes,
            num_tokens=len(tokens_key),
        )

        # Update stats
        with self._lock:
            self._stats.spill_count += 1
            self._stats.spill_bytes += total_file_bytes

        logger.debug(
            f"[ssd_cache] spilled entry: {len(tokens_key)} tokens, "
            f"{total_file_bytes} bytes on disk"
        )

        # Enforce capacity after write
        self._enforce_capacity()

    def lookup_ssd(self, tokens: tuple[int, ...]) -> dict | None:
        """Synchronous check whether tokens exist in SSD tier.

        This is fast (SQLite lookup only, no disk I/O for data).
        Called from synchronous fetch() to report an SSD candidate.

        Returns:
            Dict with entry metadata if found, None otherwise.
        """
        result = self._index.lookup_exact(tokens)
        if result is not None:
            return result
        return None

    def lookup_ssd_prefix(self, tokens: tuple[int, ...]) -> dict | None:
        """Find the longest prefix match in the SSD tier.

        Returns the longest-prefix entry metadata or None.
        """
        results = self._index.lookup_prefix(tokens)
        if results:
            return results[0]  # Already sorted by num_tokens DESC
        return None

    async def async_promote(
        self,
        tokens: tuple[int, ...],
        reserve_budget_fn,
        release_budget_fn,
    ) -> list | None:
        """Promote an entry from SSD to RAM asynchronously.

        CRITICAL: Reserves RAM budget BEFORE the disk read, to avoid
        thrash when multiple promotions race.

        Args:
            tokens: Token sequence to promote.
            reserve_budget_fn: Callable(nbytes) -> bool. Must return True
                if budget is available and reserved, False otherwise.
            release_budget_fn: Callable(nbytes) -> None. Called to release
                budget on failure.

        Returns:
            List of deserialized cache layers, or None if promotion failed.
        """
        import asyncio

        # Step 1: Look up metadata (fast, SQLite)
        meta = self._index.lookup_exact(tokens)
        if meta is None:
            with self._lock:
                self._stats.ssd_misses += 1
            return None

        memory_bytes = meta["memory_bytes"]

        # Step 2: Reserve RAM budget BEFORE disk read
        if not reserve_budget_fn(memory_bytes):
            with self._lock:
                self._stats.promotion_failures += 1
            logger.warning(
                f"[ssd_cache] promotion denied: cannot reserve "
                f"{memory_bytes} bytes RAM budget"
            )
            return None

        # Step 3: Read from disk (in thread pool to avoid blocking event loop)
        t0 = time.time()
        try:
            cache_layers = await asyncio.to_thread(
                self._read_entry, tokens, meta["file_path"]
            )
        except Exception:
            # Release budget on read failure
            release_budget_fn(memory_bytes)
            with self._lock:
                self._stats.promotion_failures += 1
            logger.exception(
                f"[ssd_cache] failed to read entry from disk "
                f"({meta['num_tokens']} tokens)"
            )
            return None

        if cache_layers is None:
            # Corrupted entry — release budget, quarantine entry
            release_budget_fn(memory_bytes)
            with self._lock:
                self._stats.promotion_failures += 1
            return None

        dt = time.time() - t0
        total_read_bytes = sum(
            os.path.getsize(
                os.path.join(
                    self._data_dir, meta["file_path"], f"layer_{i}.safetensors"
                )
            )
            for i in range(len(cache_layers))
            if os.path.exists(
                os.path.join(
                    self._data_dir, meta["file_path"], f"layer_{i}.safetensors"
                )
            )
        )

        with self._lock:
            self._stats.ssd_hits += 1
            self._stats.reload_latency_sum += dt
            self._stats.reload_bytes += total_read_bytes

        # Update access time in index
        self._index.touch(tokens)

        logger.info(
            f"[ssd_cache] promoted entry: {meta['num_tokens']} tokens, "
            f"{total_read_bytes} bytes, {dt*1000:.1f}ms"
        )

        return cache_layers

    def _read_entry(self, tokens: tuple[int, ...], relative_path: str) -> list | None:
        """Read a cache entry from disk. Called from thread pool.

        Returns list of deserialized layer dicts, or None on corruption.
        """
        entry_dir = os.path.join(self._data_dir, relative_path)
        manifest_path = os.path.join(entry_dir, "manifest.json")

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[ssd_cache] corrupt manifest for {relative_path}: {e}")
            self._quarantine_entry(tokens, relative_path)
            return None

        cache_layers = []
        for layer_meta in manifest["layers"]:
            layer_idx = layer_meta["layer_idx"]
            layer_path = os.path.join(entry_dir, f"layer_{layer_idx}.safetensors")
            layer_type = layer_meta["layer_type"]

            try:
                if layer_type in ("KVCache", "RotatingKVCache"):
                    serializer = KVCacheSerializer()
                elif layer_type in ("ArraysCache", "MambaCache"):
                    serializer = ArraysCacheSerializer()
                else:
                    logger.warning(
                        f"[ssd_cache] unknown layer type {layer_type}, skipping"
                    )
                    self._quarantine_entry(tokens, relative_path)
                    return None

                layer_data = serializer.deserialize_layer(layer_path, layer_meta)
                cache_layers.append(layer_data)
            except Exception as e:
                logger.warning(
                    f"[ssd_cache] corrupt layer {layer_idx} in {relative_path}: {e}"
                )
                self._quarantine_entry(tokens, relative_path)
                return None

        return cache_layers

    def _quarantine_entry(self, tokens: tuple[int, ...], relative_path: str) -> None:
        """Move a corrupt entry to quarantine and remove from index."""
        entry_dir = os.path.join(self._data_dir, relative_path)
        quarantine_dir = os.path.join(self._cache_dir, "quarantine", relative_path)

        try:
            if os.path.exists(entry_dir):
                os.makedirs(
                    os.path.dirname(quarantine_dir),
                    mode=self._config.dir_permissions,
                    exist_ok=True,
                )
                os.rename(entry_dir, quarantine_dir)
                logger.warning(
                    f"[ssd_cache] quarantined corrupt entry: {relative_path}"
                )
        except OSError as e:
            logger.warning(f"[ssd_cache] failed to quarantine {relative_path}: {e}")

        self._index.delete_entry(tokens)

    def _enforce_capacity(self) -> None:
        """Evict oldest SSD entries until within capacity limits.

        Called after each spill write. Removes entries by LRU order
        until both entry count and total bytes are within bounds.
        """
        import shutil

        while True:
            entry_count = self._index.get_entry_count()
            total_bytes = self._index.get_total_bytes()

            needs_evict = (
                entry_count > self._config.max_entries
                or total_bytes > self._config.max_size_bytes
            )
            if not needs_evict:
                break

            lru = self._index.get_lru(limit=1)
            if not lru:
                break

            victim = lru[0]
            victim_tokens = _blob_to_tokens(victim["tokens_blob"])
            victim_dir = os.path.join(self._data_dir, victim["file_path"])

            # Delete data files
            if os.path.exists(victim_dir):
                shutil.rmtree(victim_dir)

            # Delete from index
            self._index.delete_entry(victim_tokens)

            logger.debug(
                f"[ssd_cache] disk LRU evicted: {victim['num_tokens']} tokens, "
                f"{victim['memory_bytes']} bytes"
            )

    def reconcile(self) -> int:
        """Reconcile index with files on disk.

        Removes index entries whose data files are missing.
        Removes data directories not in the index.

        Returns number of entries cleaned up.
        """
        import shutil

        cleaned = 0

        # Phase 1: Remove index entries with missing data dirs
        all_entries = self._index.all_entries()
        for entry in all_entries:
            entry_dir = os.path.join(self._data_dir, entry["file_path"])
            manifest_path = os.path.join(entry_dir, "manifest.json")
            if not os.path.isdir(entry_dir) or not os.path.exists(manifest_path):
                tokens = _blob_to_tokens(entry["tokens_blob"])
                self._index.delete_entry(tokens)
                cleaned += 1
                logger.info(
                    f"[ssd_cache] reconcile: removed orphaned index entry "
                    f"({entry['num_tokens']} tokens, path={entry['file_path']})"
                )

        # Phase 2: Remove data directories not in the index
        if os.path.isdir(self._data_dir):
            indexed_hashes = {e["file_path"] for e in self._index.all_entries()}
            for entry_name in os.listdir(self._data_dir):
                entry_path = os.path.join(self._data_dir, entry_name)
                if (
                    os.path.isdir(entry_path)
                    and entry_name not in indexed_hashes
                    and not entry_name.endswith(".tmp")
                ):
                    shutil.rmtree(entry_path)
                    cleaned += 1
                    logger.info(
                        f"[ssd_cache] reconcile: removed orphaned data dir "
                        f"{entry_name}"
                    )

        if cleaned > 0:
            logger.info(f"[ssd_cache] reconciliation cleaned {cleaned} entries")

        return cleaned

    def close(self) -> None:
        """Close the SSD cache tier and release resources."""
        if self._closed:
            return
        self._closed = True

        # Stop writer thread
        self._writer_stop.set()
        if self._writer_thread is not None:
            try:
                self._spill_queue.put_nowait(None)  # Poison pill
            except queue.Full:
                pass
            self._writer_thread.join(timeout=5.0)
            self._writer_thread = None

        self._index.close()
        logger.info("[ssd_cache] SSDCacheTier closed")
