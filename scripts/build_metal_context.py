#!/usr/bin/env python3
"""Compile the checked-in Metal Context Engine shaders into a metallib.

This is deliberately a small, explicit workflow instead of a runtime shader
compiler.  Release builders can run it on a pinned macOS/Xcode image and ship
the resulting library beside the optional native extension.  No command is
attempted on non-macOS hosts.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "metal_context" / "kernels" / "paged_decode.metal"
APPLE_SILICON = {"arm64", "aarch64"}


def _xcrun(sdk: str, tool: str) -> str:
    result = subprocess.run(
        ["xcrun", "--sdk", sdk, "--find", tool],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip() or f"xcrun could not find {tool}"
        raise RuntimeError(detail)
    return result.stdout.strip()


def compile_metallib(output: Path, *, sdk: str = "macosx") -> None:
    """Compile the phase-1 shader and atomically publish ``output``."""

    if sys.platform != "darwin":
        raise RuntimeError("Metal shader compilation requires macOS")
    machine = platform.machine().lower()
    if machine not in APPLE_SILICON:
        raise RuntimeError(
            "Metal Context Engine compilation requires Apple Silicon "
            f"(arm64/aarch64), found {machine or 'unknown'}"
        )
    if not SOURCE.is_file():
        raise RuntimeError(f"shader source is missing: {SOURCE}")

    metal = _xcrun(sdk, "metal")
    metallib = _xcrun(sdk, "metallib")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vllm-mlx-metal-") as temp_dir:
        temp = Path(temp_dir)
        air = temp / "paged_decode.air"
        compiled = temp / "_metal_context.metallib"
        subprocess.run(
            [metal, "-c", os.fspath(SOURCE), "-o", os.fspath(air)],
            check=True,
        )
        subprocess.run(
            [metallib, os.fspath(air), "-o", os.fspath(compiled)],
            check=True,
        )
        os.replace(compiled, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "vllm_mlx" / "_metal_context.metallib",
        help="destination metallib path",
    )
    parser.add_argument("--sdk", default="macosx", help="xcrun SDK name")
    args = parser.parse_args()
    compile_metallib(args.output, sdk=args.sdk)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
