"""Dependency compatibility contracts for upstream model support."""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _project_dependencies() -> dict[str, str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = {}
    for raw in pyproject["project"]["dependencies"]:
        name = raw.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].strip()
        dependencies[name] = raw
    return dependencies


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _has_minimum(requirement: str, minimum: str) -> bool:
    needle = ">="
    assert needle in requirement
    version = requirement.split(needle, 1)[1].split(",", 1)[0].strip()
    return _version_tuple(version) >= _version_tuple(minimum)


def test_mlx_vlm_floor_includes_loader_guard_and_step37_flash():
    dependencies = _project_dependencies()

    assert _has_minimum(dependencies["mlx-vlm"], "0.6.5")


def test_mlx_lm_floor_matches_current_mlx_vlm_runtime_requirement():
    dependencies = _project_dependencies()

    assert _has_minimum(dependencies["mlx-lm"], "0.31.3")


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("0.31.2", False),
        ("0.31.3", True),
        ("0.31.4", True),
        ("0.32.0", False),
        ("0.32.1", False),
    ],
)
def test_mlx_lm_version_range_preserves_legacy_cache_api(
    version: str, supported: bool
) -> None:
    requirement = Requirement(_project_dependencies()["mlx-lm"])

    assert requirement.specifier.contains(version) is supported
