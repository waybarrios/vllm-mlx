# Contributing

We welcome contributions to vllm-mlx!

## Getting Started

```bash
# Clone the repository
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

# Install with dev dependencies
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

```bash
# Pure-Python test subset (matches the Ubuntu CI job)
pip install -e ".[dev]"
pytest \
  tests/test_mcp_security.py \
  tests/test_structured_output.py \
  tests/test_reasoning_parser.py \
  tests/test_tool_parsers.py \
  tests/test_streaming_json_encoder.py \
  tests/test_native_tool_format.py \
  tests/test_memory_cache.py \
  tests/test_prefix_cache.py \
  tests/test_mllm_cache.py \
  tests/test_api_models.py \
  tests/test_api_utils.py \
  tests/test_request.py \
  tests/test_anthropic_models.py \
  tests/test_anthropic_adapter.py \
  tests/test_harmony_parsers.py \
  -v --tb=short -m "not slow and not integration"

# Full Apple Silicon suite (requires macOS on Apple Silicon with MLX)
pip install -e ".[dev,vision]"
pytest tests/ -v --tb=short -m "not slow and not integration"

# Run specific test file
pytest tests/test_paged_cache.py -v

# Run with coverage
pytest --cov=vllm_mlx tests/
```

The full suite is intentionally split in CI:

- The Ubuntu matrix runs the pure-Python subset only.
- The Apple Silicon job runs MLX-dependent tests that require macOS on ARM.

If you are running tests locally outside the documented `.[dev]` environment, async tests
will fail because `pytest-asyncio` is a dev dependency rather than a runtime dependency.

### Code Style

```bash
# Format code
black vllm_mlx/
isort vllm_mlx/

# Type checking
mypy vllm_mlx/
```

### Running Benchmarks

```bash
# LLM benchmark
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit

# Image benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Video benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video
```

## Areas for Contribution

- **Bug fixes** - Fix issues and improve stability
- **Performance optimizations** - Improve inference speed
- **New features** - Add functionality
- **Documentation** - Improve docs and examples
- **Benchmarks** - Test on different Apple Silicon chips
- **Model support** - Test and add new models

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure they pass
5. Submit a pull request

## Code Structure

See [Architecture](architecture.md) for details on the codebase structure.

## Testing on Different Hardware

If you have access to different Apple Silicon chips (M1, M2, M3, M4), benchmark results are valuable:

```bash
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --output results_m4.json
```

## Questions?

Open an issue at [GitHub Issues](https://github.com/waybarrios/vllm-mlx/issues).
