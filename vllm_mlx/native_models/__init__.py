# SPDX-License-Identifier: Apache-2.0
"""
Native model implementations for vllm-mlx.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.nn as nn
    from mlx_lm.models.base import BaseModelArgs

_REGISTRY_CACHE: dict[str, tuple[str, str, str]] | None = None


def build_native_model_registry() -> dict[str, tuple[str, str, str]]:
    """
    Auto-discover native model modules in this package and verify no architecture collisions.

    Each model submodule must define:
    - SUPPORTED_ARCHITECTURES: tuple[str, ...]
    - Model / <Name>Model: nn.Module class
    - ModelArgs: BaseModelArgs class

    Returns:
        Mapping from normalized model_type to (module_name, model_cls_name, args_cls_name).

    Raises:
        ValueError: If multiple modules register the same model architecture.
    """
    registry: dict[str, tuple[str, str, str]] = {}
    package_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith("_"):
            continue

        module_name = f"{__name__}.{module_info.name}"
        mod = importlib.import_module(module_name)
        architectures = getattr(mod, "SUPPORTED_ARCHITECTURES", ())
        if not architectures:
            continue

        # Find model class: prefer Model or <Submodule>Model (case-insensitive search)
        model_cls = getattr(mod, "Model", None)
        if model_cls is None:
            for attr_name in dir(mod):
                if attr_name.lower() == f"{module_info.name.lower()}model":
                    model_cls = getattr(mod, attr_name)
                    break

        if model_cls is None:
            raise ValueError(
                f"Module {module_name} defines SUPPORTED_ARCHITECTURES but has no Model class"
            )

        args_cls = getattr(mod, "ModelArgs", None)
        if args_cls is None:
            raise ValueError(
                f"Module {module_name} defines SUPPORTED_ARCHITECTURES but has no ModelArgs class"
            )

        model_cls_name = model_cls.__name__
        args_cls_name = args_cls.__name__

        for arch in architectures:
            normalized_arch = arch.lower()
            if normalized_arch in registry:
                existing_module, _, _ = registry[normalized_arch]
                raise ValueError(
                    f"Architecture collision for '{normalized_arch}': "
                    f"registered by both '{existing_module}' and '{module_name}'"
                )
            registry[normalized_arch] = (module_name, model_cls_name, args_cls_name)

    return registry


def get_registry() -> dict[str, tuple[str, str, str]]:
    """Get the cached native model registry."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = build_native_model_registry()
    return _REGISTRY_CACHE


def has_native_model(model_type: str) -> bool:
    """Check if a native model implementation is available for the given model_type."""
    return model_type.lower() in get_registry()


def get_native_model_class(
    model_type: str,
) -> tuple[type[nn.Module], type[BaseModelArgs]] | None:
    """
    Get the (model_class, model_args_class) tuple for the given model_type.

    Returns:
        (model_class, model_args_class) if registered, else None.
    """
    registry = get_registry()
    key = model_type.lower()
    if key not in registry:
        return None

    module_name, model_cls_name, args_cls_name = registry[key]
    mod = importlib.import_module(module_name)
    return getattr(mod, model_cls_name), getattr(mod, args_cls_name)
