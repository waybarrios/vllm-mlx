# SPDX-License-Identifier: Apache-2.0
"""
Tests for VLM (Vision Language Model) KV cache functionality.

These tests verify the VLMCacheManager for caching image and video
KV states to speed up repeated VLM inference.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from vllm_mlx.vlm_cache import (
    VLMCacheManager,
    VLMCacheStats,
    VLMCacheEntry,
    compute_image_hash,
    compute_images_hash,
)


class TestVLMCacheStats:
    """Tests for VLMCacheStats class."""

    def test_initial_stats(self):
        """Test initial statistics are zero."""
        stats = VLMCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.tokens_saved == 0
        assert stats.image_cache_hits == 0
        assert stats.total_queries == 0
        assert stats.evictions == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = VLMCacheStats(hits=3, misses=7, total_queries=10)
        assert stats.hit_rate == 0.3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = VLMCacheStats(
            hits=5, misses=5, tokens_saved=100,
            image_cache_hits=3, total_queries=10
        )
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["hit_rate"] == 0.5
        assert d["tokens_saved"] == 100
        assert d["image_cache_hits"] == 3


class TestImageHashing:
    """Tests for image hashing functions."""

    def test_hash_local_file(self):
        """Test hashing a local file."""
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image content")
            temp_path = f.name

        hash1 = compute_image_hash(temp_path)
        hash2 = compute_image_hash(temp_path)

        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated to 16 chars

        # Cleanup
        Path(temp_path).unlink()

    def test_hash_different_content(self):
        """Test different content produces different hashes."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"content A")
            path_a = f.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"content B")
            path_b = f.name

        hash_a = compute_image_hash(path_a)
        hash_b = compute_image_hash(path_b)

        assert hash_a != hash_b

        # Cleanup
        Path(path_a).unlink()
        Path(path_b).unlink()

    def test_hash_url_string(self):
        """Test hashing a URL string."""
        url = "https://example.com/image.jpg"
        hash1 = compute_image_hash(url)
        hash2 = compute_image_hash(url)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_combined_images_hash(self):
        """Test hashing multiple images."""
        images = ["image1.jpg", "image2.jpg", "image3.jpg"]
        hash1 = compute_images_hash(images)
        hash2 = compute_images_hash(images)

        assert hash1 == hash2

        # Different order should produce same hash (sorted)
        hash3 = compute_images_hash(["image3.jpg", "image1.jpg", "image2.jpg"])
        assert hash1 == hash3

    def test_empty_images_hash(self):
        """Test hashing empty image list."""
        assert compute_images_hash([]) == "no_images"


class TestVLMCacheManager:
    """Tests for VLMCacheManager class."""

    @pytest.fixture
    def cache_manager(self):
        """Create a cache manager with default settings."""
        return VLMCacheManager(max_entries=10)

    def test_initialization(self):
        """Test cache manager initialization."""
        manager = VLMCacheManager(max_entries=50)
        assert manager.max_size == 50
        assert len(manager) == 0

    def test_fetch_empty_cache(self, cache_manager):
        """Test fetching from empty cache returns miss."""
        images = ["image.jpg"]
        prompt = "Describe this image"

        cache, hit = cache_manager.fetch_cache(images, prompt)

        assert cache is None
        assert hit is False
        assert cache_manager.stats.misses == 1
        assert cache_manager.stats.hits == 0

    def test_store_and_fetch_exact_match(self, cache_manager):
        """Test storing and fetching exact match."""
        images = ["image.jpg"]
        prompt = "Describe this image"
        mock_cache = ["kv_layer_1", "kv_layer_2"]

        # Store cache
        cache_manager.store_cache(images, prompt, mock_cache, num_tokens=100)
        assert len(cache_manager) == 1

        # Fetch exact match
        cache, hit = cache_manager.fetch_cache(images, prompt)

        assert cache is not None
        assert hit is True
        assert cache_manager.stats.hits == 1
        assert cache_manager.stats.tokens_saved == 100
        assert cache_manager.stats.image_cache_hits == 1

    def test_different_prompt_same_image(self, cache_manager):
        """Test different prompts with same image don't share cache."""
        images = ["image.jpg"]
        mock_cache = ["cache"]

        # Store with first prompt
        cache_manager.store_cache(images, "Describe this", mock_cache)

        # Fetch with different prompt - should miss
        cache, hit = cache_manager.fetch_cache(images, "What color is this?")

        assert hit is False
        assert cache_manager.stats.misses == 1

    def test_different_image_same_prompt(self, cache_manager):
        """Test same prompt with different images don't share cache."""
        prompt = "Describe this"
        mock_cache = ["cache"]

        # Store with first image
        cache_manager.store_cache(["image1.jpg"], prompt, mock_cache)

        # Fetch with different image - should miss
        cache, hit = cache_manager.fetch_cache(["image2.jpg"], prompt)

        assert hit is False
        assert cache_manager.stats.misses == 1

    def test_multi_image_caching(self, cache_manager):
        """Test caching with multiple images."""
        images = ["img1.jpg", "img2.jpg", "img3.jpg"]
        prompt = "Compare these images"
        mock_cache = ["multi_image_cache"]

        # Store
        cache_manager.store_cache(images, prompt, mock_cache, num_tokens=500)

        # Fetch - should hit
        cache, hit = cache_manager.fetch_cache(images, prompt)

        assert hit is True
        assert cache_manager.stats.tokens_saved == 500

    def test_video_caching(self, cache_manager):
        """Test caching with video source identifier."""
        # Video sources include fps and max_frames in the key
        video_source = "video:video.mp4:fps2.0:max32"
        prompt = "Describe this video"
        mock_cache = ["video_cache"]

        # Store
        cache_manager.store_cache([video_source], prompt, mock_cache, num_tokens=1000)

        # Fetch - should hit
        cache, hit = cache_manager.fetch_cache([video_source], prompt)

        assert hit is True
        assert cache_manager.stats.tokens_saved == 1000

    def test_video_different_fps_misses(self, cache_manager):
        """Test different video parameters don't share cache."""
        prompt = "Describe this video"
        mock_cache = ["video_cache"]

        # Store with fps=2.0
        cache_manager.store_cache(
            ["video:video.mp4:fps2.0:max32"], prompt, mock_cache
        )

        # Fetch with fps=4.0 - should miss (different key)
        cache, hit = cache_manager.fetch_cache(
            ["video:video.mp4:fps4.0:max32"], prompt
        )

        assert hit is False

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        manager = VLMCacheManager(max_entries=3)

        # Fill cache
        manager.store_cache(["img1.jpg"], "p1", ["cache1"])
        manager.store_cache(["img2.jpg"], "p2", ["cache2"])
        manager.store_cache(["img3.jpg"], "p3", ["cache3"])
        assert len(manager) == 3

        # Add one more - should evict oldest
        manager.store_cache(["img4.jpg"], "p4", ["cache4"])
        assert len(manager) == 3
        assert manager.stats.evictions == 1

        # img1 should be evicted
        cache, hit = manager.fetch_cache(["img1.jpg"], "p1")
        assert hit is False

    def test_lru_touch_on_access(self):
        """Test that accessing a cache updates LRU order."""
        manager = VLMCacheManager(max_entries=3)

        # Fill cache
        manager.store_cache(["img1.jpg"], "p1", ["cache1"])
        manager.store_cache(["img2.jpg"], "p2", ["cache2"])
        manager.store_cache(["img3.jpg"], "p3", ["cache3"])

        # Access img1 to make it most recently used
        manager.fetch_cache(["img1.jpg"], "p1")

        # Add new entry - should evict img2 (oldest untouched)
        manager.store_cache(["img4.jpg"], "p4", ["cache4"])

        # img1 should still be there
        cache, hit = manager.fetch_cache(["img1.jpg"], "p1")
        assert hit is True

        # img2 should be evicted
        cache, hit = manager.fetch_cache(["img2.jpg"], "p2")
        assert hit is False

    def test_clear(self, cache_manager):
        """Test clearing the cache."""
        cache_manager.store_cache(["img1.jpg"], "p1", ["cache1"])
        cache_manager.store_cache(["img2.jpg"], "p2", ["cache2"])
        assert len(cache_manager) == 2

        cache_manager.clear()
        assert len(cache_manager) == 0
        assert cache_manager.stats.hits == 0

    def test_cache_deep_copy(self, cache_manager):
        """Test that fetched cache is a deep copy."""
        original = [[1, 2, 3]]
        cache_manager.store_cache(["img.jpg"], "prompt", original)

        cache, _ = cache_manager.fetch_cache(["img.jpg"], "prompt")

        # Modify returned cache
        cache[0].append(4)

        # Original should be unchanged
        cache2, _ = cache_manager.fetch_cache(["img.jpg"], "prompt")
        assert cache2[0] == [1, 2, 3]

    def test_get_stats(self, cache_manager):
        """Test getting statistics."""
        cache_manager.store_cache(["img.jpg"], "prompt", ["cache"], num_tokens=100)
        cache_manager.fetch_cache(["img.jpg"], "prompt")  # Hit
        cache_manager.fetch_cache(["other.jpg"], "prompt")  # Miss

        stats = cache_manager.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["tokens_saved"] == 100
        assert stats["image_cache_hits"] == 1


class TestMLLMCacheIntegration:
    """Integration tests for MLLM model caching."""

    def test_mllm_cache_initialization(self):
        """Test MLLM model initializes with cache."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        # With cache enabled (default)
        model = MLXMultimodalLM("test-model", enable_cache=True, cache_size=100)
        assert model.enable_cache is True
        assert model._cache_manager is not None
        assert model._cache_manager.max_size == 100

    def test_mllm_cache_disabled(self):
        """Test MLLM model with cache disabled."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", enable_cache=False)
        assert model.enable_cache is False
        assert model._cache_manager is None

    def test_mllm_get_cache_stats(self):
        """Test getting cache stats from MLLM model."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", enable_cache=True)
        stats = model.get_cache_stats()

        assert stats["enabled"] is True
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["cache_entries"] == 0

    def test_mllm_clear_cache(self):
        """Test clearing cache from MLLM model."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", enable_cache=True)
        # Manually add to cache for testing
        model._cache_manager.store_cache(["img.jpg"], "test", ["cache"])
        assert len(model._cache_manager) == 1

        model.clear_cache()
        assert len(model._cache_manager) == 0


if __name__ == "__main__":
    # Quick standalone test with real model
    import asyncio
    import time

    MODEL_NAME = "mlx-community/Qwen3-VL-4B-Instruct-3bit"

    def run_vlm_cache_test():
        from vllm_mlx.models.mllm import MLXMultimodalLM
        import tempfile
        from PIL import Image

        print("=" * 60)
        print("VLM Cache Test")
        print("=" * 60)
        print(f"Model: {MODEL_NAME}")

        # Create a test image
        print("\nCreating test image...")
        img = Image.new('RGB', (224, 224), color='blue')
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name)
            test_image = f.name
        print(f"Test image: {test_image}")

        print("\nLoading model (this may take a moment)...")
        model = MLXMultimodalLM(MODEL_NAME, enable_cache=True, cache_size=50)
        model.load()

        prompt = "What color is this image? Answer in one word."

        print("\n[1] First request (cache miss expected)...")
        start = time.perf_counter()
        output1 = model.generate(prompt=prompt, images=[test_image], max_tokens=20)
        t1 = time.perf_counter() - start
        stats1 = model.get_cache_stats()
        print(f"    Time: {t1:.2f}s")
        print(f"    Output: {output1.text[:50]}...")
        print(f"    Stats: hits={stats1['hits']}, misses={stats1['misses']}")

        print("\n[2] Second request SAME image+prompt (cache hit expected)...")
        start = time.perf_counter()
        output2 = model.generate(prompt=prompt, images=[test_image], max_tokens=20)
        t2 = time.perf_counter() - start
        stats2 = model.get_cache_stats()
        print(f"    Time: {t2:.2f}s")
        print(f"    Output: {output2.text[:50]}...")
        print(f"    Stats: hits={stats2['hits']}, misses={stats2['misses']}, tokens_saved={stats2['tokens_saved']}")

        print("\n[3] Third request DIFFERENT prompt (cache miss expected)...")
        start = time.perf_counter()
        output3 = model.generate(prompt="Describe the contents of this image.", images=[test_image], max_tokens=50)
        t3 = time.perf_counter() - start
        stats3 = model.get_cache_stats()
        print(f"    Time: {t3:.2f}s")
        print(f"    Output: {output3.text[:50]}...")
        print(f"    Stats: hits={stats3['hits']}, misses={stats3['misses']}")

        # Video test (if available)
        print("\n[4] Testing video cache (simulated with same image)...")
        video_source = f"video:{test_image}:fps2.0:max8"
        # Simulate storing video cache
        model._cache_manager.store_cache([video_source], "What happens?", ["fake_cache"], num_tokens=500)
        cache, hit = model._cache_manager.fetch_cache([video_source], "What happens?")
        print(f"    Video cache stored and retrieved: hit={hit}")

        print("\n" + "=" * 60)
        print("Final Cache Stats")
        print("=" * 60)
        final_stats = model.get_cache_stats()
        print(f"Enabled: {final_stats['enabled']}")
        print(f"Hits: {final_stats['hits']}")
        print(f"Misses: {final_stats['misses']}")
        print(f"Hit rate: {final_stats['hit_rate']*100:.1f}%")
        print(f"Tokens saved: {final_stats['tokens_saved']}")
        print(f"Image cache hits: {final_stats['image_cache_hits']}")
        print(f"Cache entries: {final_stats['cache_entries']}")
        print("=" * 60)

        # Cleanup
        Path(test_image).unlink()

    run_vlm_cache_test()
