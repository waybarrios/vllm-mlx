# SPDX-License-Identifier: Apache-2.0
"""
Tests for vllm-mlx hardware detection and system info.

Usage:
    pytest tests/test_optimizations.py -v
"""

import pytest


def test_model_runner_optimizations_reach_hardware_and_compilation(monkeypatch):
    from vllm_mlx import optimizations
    from vllm_mlx.model_runner import MLXModelRunner

    hardware = optimizations.HardwareInfo("Test", 16.0, 100.0, 8)
    monkeypatch.setattr(optimizations, "detect_hardware", lambda: hardware)
    runner = MLXModelRunner.__new__(MLXModelRunner)
    runner._hardware_info = None
    compiled = []
    monkeypatch.setattr(
        runner, "_setup_compiled_forward", lambda: compiled.append(True)
    )

    runner._apply_optimizations()

    assert runner._hardware_info is hardware
    assert compiled == [True]


def test_model_runner_optimization_failure_preserves_fallback(monkeypatch, caplog):
    from vllm_mlx import optimizations
    from vllm_mlx.model_runner import MLXModelRunner

    def fail_detection():
        raise RuntimeError("hardware unavailable")

    monkeypatch.setattr(optimizations, "detect_hardware", fail_detection)
    runner = MLXModelRunner.__new__(MLXModelRunner)
    runner._hardware_info = None
    compiled = []
    monkeypatch.setattr(
        runner, "_setup_compiled_forward", lambda: compiled.append(True)
    )

    runner._apply_optimizations()

    assert runner._hardware_info is None
    assert compiled == []
    assert "Failed to apply optimizations" in caplog.text


def test_model_runner_compiled_forward_matches_mlx_forward():
    from types import SimpleNamespace

    import mlx.core as mx

    from vllm_mlx.model_runner import MLXModelRunner

    class Model:
        def __call__(self, inputs):
            return mx.tanh(inputs @ mx.array([[0.5, -1.0], [1.5, 0.25]]))

    runner = MLXModelRunner.__new__(MLXModelRunner)
    runner.model = Model()
    runner._compiled_forward = None
    runner._hardware_info = None
    runner._loaded = True
    runner._enable_optimizations = True
    runner.model_config = SimpleNamespace(model="tiny-compile")
    runner._apply_optimizations()

    assert runner._compiled_forward is not None
    inputs = mx.array([[1.0, 2.0], [-1.0, 0.5]])
    expected = runner.model(inputs)
    actual = runner._compiled_forward(inputs)
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected).item()
    hardware = runner._hardware_info
    assert hardware is not None
    assert runner.get_model_info()["hardware"] == {
        "chip": hardware.chip_name,
        "memory_gb": hardware.total_memory_gb,
        "bandwidth_gbs": hardware.memory_bandwidth_gbs,
        "gpu_cores": hardware.gpu_cores,
    }


class TestHardwareDetection:
    """Tests for hardware detection functionality."""

    def test_detect_hardware(self):
        """Test that hardware detection works."""
        from vllm_mlx.optimizations import detect_hardware

        hw = detect_hardware()

        assert hw is not None
        assert hw.chip_name is not None
        assert hw.total_memory_gb > 0
        assert hw.memory_bandwidth_gbs > 0
        assert hw.gpu_cores > 0

    def test_get_system_memory(self):
        """Test that system memory detection works."""
        from vllm_mlx.optimizations import get_system_memory_gb

        memory_gb = get_system_memory_gb()

        assert memory_gb > 0
        assert memory_gb < 1024  # Sanity check: less than 1TB

    def test_hardware_profiles_exist(self):
        """Test that hardware profiles are defined."""
        from vllm_mlx.optimizations import HARDWARE_PROFILES

        assert len(HARDWARE_PROFILES) > 0
        assert "M1" in HARDWARE_PROFILES
        assert "M4 Max" in HARDWARE_PROFILES


class TestOptimizationStatus:
    """Tests for optimization status reporting."""

    def test_get_optimization_status(self):
        """Test optimization status reporting."""
        from vllm_mlx.optimizations import get_optimization_status

        status = get_optimization_status()

        assert "hardware" in status
        assert "mlx_memory" in status
        assert "mlx_lm_features" in status

        assert "chip" in status["hardware"]
        assert "device_name" in status["hardware"]


class TestMemoryBandwidth:
    """Tests for memory bandwidth benchmarking."""

    @pytest.mark.slow
    def test_memory_bandwidth_benchmark(self):
        """Test memory bandwidth benchmark."""
        from vllm_mlx.optimizations import benchmark_memory_bandwidth

        results = benchmark_memory_bandwidth()

        assert "1MB" in results
        assert "4MB" in results
        assert "16MB" in results

        print(f"\n{'='*50}")
        print("Memory Bandwidth Benchmark")
        print(f"{'='*50}")
        for size, bandwidth in results.items():
            print(f"{size}: {bandwidth}")
        print(f"{'='*50}")


def run_quick_test():
    """Run a quick test of hardware detection."""
    from vllm_mlx.optimizations import detect_hardware, get_optimization_status

    print("=" * 60)
    print("Quick Hardware Detection Test")
    print("=" * 60)

    hw = detect_hardware()
    print("\nHardware Detection:")
    print(f"  Chip: {hw.chip_name}")
    print(f"  Memory: {hw.total_memory_gb:.1f} GB")
    print(f"  Bandwidth: {hw.memory_bandwidth_gbs} GB/s")
    print(f"  GPU Cores: {hw.gpu_cores}")

    status = get_optimization_status()
    print("\nMLX-LM Features (built-in):")
    for feature, value in status["mlx_lm_features"].items():
        print(f"  {feature}: {value}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_test()
