# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and shared fixtures."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default=None,
        help="URL of the vllm-mlx server for integration tests",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require model loading",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires model loading)"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires running server)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is passed."""
    server_url = config.getoption("--server-url")
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            # Supplying an integration server is an explicit opt-in to the
            # integration suite, including tests that are also marked slow.
            if item.get_closest_marker("slow") is not None and not (
                server_url and item.get_closest_marker("integration") is not None
            ):
                item.add_marker(skip_slow)

    # Skip integration tests unless server URL is explicitly provided
    if not server_url:
        skip_integration = pytest.mark.skip(
            reason="Integration tests require --server-url"
        )
        for item in items:
            if item.get_closest_marker("integration") is not None:
                item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def server_url(request):
    """Get server URL from command line."""
    return request.config.getoption("--server-url")


@pytest.fixture(scope="session")
def anyio_backend():
    """Run anyio-marked tests on asyncio only."""
    return "asyncio"


@pytest.fixture(autouse=True, scope="session")
def _bind_mlx_default_stream():
    """Ensure the main-thread default MLX stream is set once at session start.

    Tests that create worker threads with ``bind_generation_streams`` allocate
    new MLX streams and call ``mx.set_default_stream``.  Because MLX streams
    are thread-local, this is harmless to the main thread.  However, if a test
    *on the main thread* calls ``bind_generation_streams`` (e.g. via
    ``generate_batch_sync`` or a monkeypatch), the default stream changes and
    later tests that assume the original default stream will fail with
    ``RuntimeError: There is no Stream(gpu, N) in current thread``.

    Binding once at session start gives all main-thread tests a known-good
    default stream.
    """
    try:
        import mlx.core as mx

        mx.set_default_stream(mx.new_stream(mx.default_device()))
    except ImportError:
        pass
