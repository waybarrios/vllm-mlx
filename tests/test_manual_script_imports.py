# SPDX-License-Identifier: Apache-2.0
"""Manual native experiments must be safe to import during pytest discovery."""

import ast
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = (
    "test_paged_cache_benefits.py",
    "test_paged_cache_real_inference.py",
    "test_paged_cache_real_model.py",
    "test_registry_idle_unload_real_model.py",
)
_DEMONSTRATIONS = {
    "demonstrate_benefit_1_shared_system_prompts",
    "demonstrate_benefit_2_memory_efficiency",
    "demonstrate_benefit_3_prefix_sharing",
    "demonstrate_copy_on_write_demo",
}


def _run_python(*args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=_ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("filename", _SCRIPTS)
def test_manual_script_import_does_not_run_experiment(filename):
    code = textwrap.dedent("""
        import asyncio
        import importlib.abc
        import importlib.util
        import subprocess
        import sys

        class NoExperimentImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'vllm_mlx'}:
                    raise AssertionError('Experiment import during discovery: ' + fullname)

        def no_experiment(*args, **kwargs):
            raise AssertionError('Experiment execution during discovery')

        sys.meta_path.insert(0, NoExperimentImports())
        subprocess.Popen = no_experiment
        asyncio.run = no_experiment
        spec = importlib.util.spec_from_file_location('manual_script_import_probe', sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print('IMPORT_COMPLETED_WITHOUT_EXPERIMENT')
        """)
    result = _run_python("-c", code, str(_ROOT / "tests" / filename))
    assert result.returncode == 0, result.stderr
    # A module-level sys.exit(0) also has return code zero; the sentinel must
    # execute after import, and no benchmark output may precede it.
    assert result.stdout == "IMPORT_COMPLETED_WITHOUT_EXPERIMENT\n"
    assert result.stderr == ""


@pytest.mark.parametrize("filename", _SCRIPTS)
def test_manual_script_keeps_direct_cli_hardware_guard(filename):
    script = _ROOT / "tests" / filename
    tree = ast.parse(script.read_text())
    if filename == "test_paged_cache_real_inference.py":
        entry = next(
            node
            for node in tree.body
            if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
        )
    else:
        entry = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
    body = list(entry.body)
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # Ignore an entry-point docstring.
    guard = body[0]
    assert isinstance(guard, ast.If)
    assert "sys.platform" in ast.unparse(guard.test)
    assert "platform.machine()" in ast.unparse(guard.test)
    assert "sys.exit(0)" in ast.unparse(guard)

    # Never run real experiments on an available native host. The static
    # entry-point contract above is checked there too, without platform spoofing.
    if sys.platform != "darwin" or platform.machine() != "arm64":
        result = _run_python(str(script))
        assert result.returncode == 0, result.stderr
        assert result.stdout == "This test requires Apple Silicon\n"
        assert result.stderr == ""


def test_benefit_demonstrations_remain_callable_from_main_not_pytest():
    tree = ast.parse((_ROOT / "tests" / "test_paged_cache_benefits.py").read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert not any(name.startswith("test_") for name in functions)
    assert {name for name in functions if name.startswith("demonstrate_")} == (
        _DEMONSTRATIONS
    )
    main_calls = {
        node.func.id
        for node in ast.walk(functions["main"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert _DEMONSTRATIONS <= main_calls
