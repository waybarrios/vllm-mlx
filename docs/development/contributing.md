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
# Run the full suite on Apple Silicon
pytest tests/

# Run a focused test first
pytest tests/test_paged_cache.py -v
```

MLX-dependent tests require an Apple Silicon environment. Other platforms can
still run supported static checks and non-MLX tests.

### Code Style

```bash
# Lint
ruff check vllm_mlx/ tests/ --select E,F,W --ignore E402,E501,E731,F811,F841

# Check formatting
black --check vllm_mlx/ tests/

# Type-check relevant changes
mypy vllm_mlx/ --ignore-missing-imports --no-error-summary
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

1. Fork the repository.
2. Create a focused feature branch.
3. Make the smallest coherent change that resolves the problem.
4. Add regression coverage when practical.
5. Run the relevant tests and code-quality checks.
6. Submit a pull request describing the impact and verification performed.

## Code Structure

See [Architecture](architecture.md) for details on the codebase structure.

## Testing on Different Hardware

If you have access to different Apple Silicon chips (M1, M2, M3, M4, M5), benchmark results are valuable:

```bash
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --output results_m4.json
```

## Questions?

Open an issue at [GitHub Issues](https://github.com/waybarrios/vllm-mlx/issues).
