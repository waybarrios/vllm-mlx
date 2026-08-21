# SPDX-License-Identifier: Apache-2.0
"""
Real-server test for registry-mode idle-unload memory reclaim.

Starts a real `vllm-mlx serve --models-config ...` server subprocess,
drives it over HTTP exactly like a real client would, and reports real
process RSS before load, after load, after idle-unload, and after a
transparent reload -- to demonstrate that mx.clear_cache() in
SimpleEngine.stop() actually releases memory back to the OS instead of
only dropping Python references, and that --auto-unload-idle-seconds now
takes effect in --models-config (registry) mode.

Not a pytest test (see test_paged_cache_real_inference.py for the same
convention): driving real generation through a bare, directly instantiated
ModelManager/SimpleEngine (rather than through the full server process)
hits MLX's worker-thread stream binding in a way that is unrelated to the
idle-unload logic under test here, so this script goes through the real
server process instead -- the same path production traffic uses -- and
must only be invoked directly.

Usage:
    python tests/test_registry_idle_unload_real_model.py
    python tests/test_registry_idle_unload_real_model.py --model mlx-community/Qwen3-0.6B-8bit
"""

import argparse
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Skip if not on Apple Silicon
if sys.platform != "darwin" or platform.machine() != "arm64":
    print("This test requires Apple Silicon")
    sys.exit(0)

PORT = 8971
IDLE_UNLOAD_SECONDS = 10
STARTUP_TIMEOUT_S = 60
UNLOAD_POLL_TIMEOUT_S = 30


def _resolve_local_snapshot(model_name: str) -> str:
    """Return the local HF cache snapshot path for an already-cached model."""
    from vllm_mlx.utils.download import DownloadConfig, ensure_model_downloaded

    return str(ensure_model_downloaded(model_name, config=DownloadConfig()))


def _wait_for_health(base_url: str, timeout_s: float) -> None:
    import requests

    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                return
        except requests.exceptions.RequestException as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"Server did not become healthy in time: {last_error}")


def _model_status(base_url: str) -> dict:
    import requests

    resp = requests.get(f"{base_url}/v1/status", timeout=5)
    resp.raise_for_status()
    return resp.json()["model_manager"]["models"][0]


def _send_chat_completion(base_url: str, content: str) -> None:
    import requests

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "idle-test",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 8,
        },
        timeout=60,
    )
    resp.raise_for_status()


def _rss_mb(pid: int) -> float:
    import psutil

    return psutil.Process(pid).memory_info().rss / (1024**2)


def run_idle_unload_memory_check(model_name: str) -> None:
    print("=" * 70)
    print("  REGISTRY IDLE-UNLOAD - REAL SERVER MEMORY TEST")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print("Resolving local model snapshot (no download expected)...")
    snapshot_path = _resolve_local_snapshot(model_name)
    print(f"  -> {snapshot_path}")

    base_url = f"http://127.0.0.1:{PORT}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "models.yaml"
        config_path.write_text(f"""
manager:
  memory_budget_gb: 32
  idle_unload_seconds: {IDLE_UNLOAD_SECONDS}
models:
  - name: idle-test
    path: {snapshot_path}
    estimated_memory_gb: 1
""")

        print(
            f"\nStarting server on port {PORT} "
            f"(idle_unload_seconds={IDLE_UNLOAD_SECONDS})..."
        )
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm_mlx.cli",
                "serve",
                "--models-config",
                str(config_path),
                "--port",
                str(PORT),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        ok = True
        try:
            _wait_for_health(base_url, STARTUP_TIMEOUT_S)

            rss_before = _rss_mb(proc.pid)
            print(f"\nRSS before load:  {rss_before:.1f} MB")

            print("Sending first chat completion (triggers load)...")
            _send_chat_completion(base_url, "Say hi in one word.")

            status = _model_status(base_url)
            assert (
                status["status"] == "loaded"
            ), f"expected model to be loaded after a request, got {status!r}"
            rss_loaded = _rss_mb(proc.pid)
            print(
                f"RSS after load:   {rss_loaded:.1f} MB  "
                f"(+{rss_loaded - rss_before:.1f} MB)"
            )

            print(
                f"Waiting for idle-unload (timeout={IDLE_UNLOAD_SECONDS}s, "
                f"polling up to {UNLOAD_POLL_TIMEOUT_S}s)..."
            )
            deadline = time.monotonic() + UNLOAD_POLL_TIMEOUT_S
            status = _model_status(base_url)
            while status["status"] != "unloaded" and time.monotonic() < deadline:
                time.sleep(0.5)
                status = _model_status(base_url)

            if status["status"] != "unloaded":
                print(
                    f"\nFAIL: model did not unload within {UNLOAD_POLL_TIMEOUT_S}s "
                    f"(status={status['status']!r})"
                )
                sys.exit(1)

            rss_unloaded = _rss_mb(proc.pid)
            print(
                f"RSS after unload: {rss_unloaded:.1f} MB  "
                f"(-{rss_loaded - rss_unloaded:.1f} MB reclaimed)"
            )

            print("Sending second chat completion (triggers transparent reload)...")
            _send_chat_completion(base_url, "Say hi again.")

            status = _model_status(base_url)
            assert (
                status["status"] == "loaded"
            ), f"expected model to reload transparently, got {status!r}"
            rss_reloaded = _rss_mb(proc.pid)
            print(
                f"RSS after reload: {rss_reloaded:.1f} MB  "
                f"(+{rss_reloaded - rss_unloaded:.1f} MB)"
            )

            loaded_growth = rss_loaded - rss_before
            reclaimed = rss_loaded - rss_unloaded
            reload_growth = rss_reloaded - rss_unloaded

            print("\n" + "-" * 70)
            print("SUMMARY")
            print("-" * 70)
            print(f"  Load grew RSS by:   {loaded_growth:+.1f} MB")
            print(f"  Unload reclaimed:   {reclaimed:+.1f} MB")
            print(f"  Reload grew RSS by: {reload_growth:+.1f} MB")

            if loaded_growth <= 50:
                print("\n  WARNING: loading did not grow RSS as expected (>50MB).")
                ok = False
            if reclaimed <= 0:
                print(
                    "\n  WARNING: unload did not reclaim memory -- "
                    "mx.clear_cache() may not be releasing the Metal buffer cache."
                )
                ok = False
            if reload_growth <= 50:
                print("\n  WARNING: reload did not grow RSS as expected (>50MB).")
                ok = False

            if ok:
                print(
                    "\n  PASS: registry-mode idle-unload measurably releases "
                    "and re-acquires memory."
                )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test registry-mode idle-unload memory reclaim with a real model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-0.6B-8bit",
        help="Model to use for testing",
    )
    args = parser.parse_args()

    run_idle_unload_memory_check(args.model)


if __name__ == "__main__":
    main()
