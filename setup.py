"""Optional native Metal Context Engine build hooks.

The project metadata remains in ``pyproject.toml``.  This small shim only
adds an extension when the caller is on Apple Silicon macOS, the Metal
compiler is available, and ``VLLM_MLX_BUILD_METAL_CONTEXT=1`` is explicitly
requested.  Normal installs keep this optional extension disabled.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
STRICT = {"1", "true", "yes", "on", "required"}
APPLE_SILICON = {"arm64", "aarch64"}


def _metal_toolchain_available() -> bool:
    if sys.platform != "darwin" or shutil.which("xcrun") is None:
        return False
    return (
        subprocess.run(
            ["xcrun", "--sdk", "macosx", "--find", "metal"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _extensions() -> list[Extension]:
    # The native foundation is never built implicitly.  This keeps ordinary
    # wheels/editable installs portable and makes the toolchain choice
    # auditable in CI and release logs.
    mode = os.environ.get("VLLM_MLX_BUILD_METAL_CONTEXT", "disabled").lower()
    if mode in {"0", "false", "no", "off", "disabled"}:
        return []
    if sys.platform != "darwin":
        if mode in STRICT:
            raise RuntimeError("VLLM_MLX_BUILD_METAL_CONTEXT=1 requires macOS")
        return []
    machine = platform.machine().lower()
    if machine not in APPLE_SILICON:
        if mode in STRICT:
            raise RuntimeError(
                "VLLM_MLX_BUILD_METAL_CONTEXT=1 requires Apple Silicon "
                f"(arm64/aarch64), found {machine or 'unknown'}"
            )
        return []
    if not _metal_toolchain_available():
        if mode in STRICT:
            raise RuntimeError(
                "Metal toolchain unavailable; install Xcode/Command Line Tools "
                "or set VLLM_MLX_BUILD_METAL_CONTEXT=0"
            )
        return []
    return [
        Extension(
            "vllm_mlx._metal_context",
            sources=["native/metal_context/src/python_module.mm"],
            language="objc++",
            extra_compile_args=[
                "-std=c++17",
                "-fobjc-arc",
                "-mmacosx-version-min=11.0",
            ],
            extra_link_args=[
                "-mmacosx-version-min=11.0",
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
            ],
        )
    ]


class MetalBuildExt(build_ext):
    """Build the metallib before the extension and package it beside it."""

    def run(self) -> None:
        if not self.extensions:
            return super().run()
        output = Path(self.build_temp) / "metal_context" / "_metal_context.metallib"
        subprocess.run(
            [
                sys.executable,
                os.fspath(ROOT / "scripts" / "build_metal_context.py"),
                "--output",
                os.fspath(output),
            ],
            check=True,
        )
        super().run()
        destination_dir = (
            ROOT / "vllm_mlx" if self.inplace else Path(self.build_lib) / "vllm_mlx"
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, destination_dir / "_metal_context.metallib")

    def build_extensions(self) -> None:
        # distutils does not list Objective-C++ as a recognized source suffix,
        # even though Apple's clang handles it correctly.  Register it before
        # setuptools asks the compiler to classify extension sources.
        if ".mm" not in self.compiler.src_extensions:
            self.compiler.src_extensions.append(".mm")
        super().build_extensions()


setup(ext_modules=_extensions(), cmdclass={"build_ext": MetalBuildExt})
