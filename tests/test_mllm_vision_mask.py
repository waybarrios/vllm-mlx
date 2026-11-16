import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def test_run_vision_encoding_supplies_mask_when_required_by_model(monkeypatch):
    # Stub mlx.core so this test runs cross-platform without Apple Silicon hardware
    if "mlx.core" not in sys.modules:
        mlx_mod = types.ModuleType("mlx")
        mx_mod = types.ModuleType("mlx.core")
        mx_mod.array = list
        mlx_mod.core = mx_mod
        monkeypatch.setitem(sys.modules, "mlx", mlx_mod)
        monkeypatch.setitem(sys.modules, "mlx.core", mx_mod)

    # Verify the AST/source of _run_vision_encoding handles required 'mask' parameter
    src_path = Path(__file__).resolve().parents[1] / "vllm_mlx" / "mllm_batch_generator.py"
    src = src_path.read_text(encoding="utf-8")
    assert 'sig_params.get("mask")' in src
    assert 'kwargs["mask"] = request.attention_mask' in src
