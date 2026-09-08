# Installation

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.10+

## Install a release

For a first run, use a fresh environment rather than an editable checkout.
This walkthrough targets [v0.4.1](https://github.com/waybarrios/vllm-mlx/releases/tag/v0.4.1).
This is a pinned reference release, not a promise to install the latest version.
When updating this walkthrough for a release, update the pins and expected
version in the README and this page together, and recheck the quickstart's
CLI flags and first-response request against that release.
Changes merged into `main` after that release are not included just because
their PR is closed. Check the release containing a required fix before choosing
a model or enabling an advanced feature.

Use an Apple Silicon-native Python installation (`arm64`), not a terminal or
Python running under Rosetta. The example uses Python 3.12; install it first
if that command is unavailable.

```bash
python3.12 -c 'import platform; print(platform.system(), platform.machine())'
# Expected: Darwin arm64
mkdir vllm-mlx-demo
cd vllm-mlx-demo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'vllm-mlx==0.4.1'
python -m pip check
python -c 'from importlib.metadata import version; print(version("vllm-mlx"))'
# Expected: 0.4.1
```

The version pin fixes the server release, not every transitive dependency.
After a successful run, retain `python -m pip freeze` output when reporting a
problem or reproducing the environment. Do not install this walkthrough into
an environment managed by another application or running inference service.

Continue to [your first response](quickstart.md#first-response). Model download
and loading time are additional; installation is not a model compatibility or
memory-capacity check.

## Development checkout

Use this path when contributing or when a specific unreleased fix is required.
It follows source, not the release walkthrough above. Record `git rev-parse
HEAD` with any test or issue report. Use a separate environment; do not mix the
editable checkout into the release environment.

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Optional release extras

Run these commands in the release environment, not the development checkout.
Contributors use the corresponding editable extras instead.

### Vision Support

For video processing with transformers:

```bash
python -m pip install 'vllm-mlx[vision]==0.4.1'
```

### Audio Support (STT/TTS)

```bash
python -m pip install 'vllm-mlx[audio]==0.4.1'
```

## What Gets Installed

- `mlx`, `mlx-lm`, `mlx-vlm` - MLX framework and model libraries
- `transformers`, `tokenizers` - HuggingFace libraries
- `opencv-python` - Video processing
- `gradio` - Chat UI
- `psutil` - Resource monitoring
- `mlx-audio` (optional) - Speech-to-Text and Text-to-Speech
- `mlx-embeddings` - Text embeddings

## Verify Installation

```bash
# Check CLI commands
vllm-mlx --help
vllm-mlx-bench --help
vllm-mlx-chat --help
```

Use the [first-response check](quickstart.md#first-response) before benchmarks,
tools, multimodal inputs, or concurrent clients.

## Troubleshooting

### MLX not found

Ensure you're on Apple Silicon:
```bash
uname -m  # Should output "arm64"
```

### Model download fails

Check your internet connection and HuggingFace access. Some models require authentication:
```bash
hf auth login
```

You can inspect and stage models before serving:
```bash
vllm-mlx model inspect mlx-community/Llama-3.2-3B-Instruct-4bit
vllm-mlx model acquire mlx-community/Llama-3.2-3B-Instruct-4bit \
  --target-dir ./models/llama-3b-4bit
```

### Out of memory

Stop the server with Ctrl-C before trying a smaller model. Close other model
servers and reduce the requested context before adding cache or speculative
features. A weight download fitting on disk does not prove the runtime fits
in unified memory. See [artifact and profile choices](quickstart.md#artifact-and-profile-choices).

Use a smaller quantized model:
```bash
vllm-mlx serve mlx-community/Llama-3.2-1B-Instruct-4bit
```

### Server interruptions during long runs (macOS sleep)

Your macOS machine may go to sleep during long-running server sessions. Try using `caffeinate` to prevent sleep:

```bash
caffeinate -dimsu
```
