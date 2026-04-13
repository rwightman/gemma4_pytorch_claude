from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class InitContext:
    generator: torch.Generator | None = None


def resolve_residual_init_std(
        init_std: float,
        residual_init_std: float | None,
        use_depth_scaled_residual_init: bool,
        num_layers: int,
) -> float | None:
    if residual_init_std is not None:
        return residual_init_std
    if use_depth_scaled_residual_init:
        return init_std / (2.0 * num_layers) ** 0.5
    return None


def factory_kwargs(
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
) -> dict[str, torch.device | str | torch.dtype]:
    kwargs: dict[str, torch.device | str | torch.dtype] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


class InitModule(nn.Module):
    """Base module with recursive init and runtime-buffer rebuild hooks."""

    def init_weights(
            self,
            ctx: InitContext | None = None,
            _visited: set[int] | None = None,
    ) -> None:
        if ctx is None:
            ctx = InitContext()
        if _visited is None:
            _visited = set()
        _init_module_tree(self, ctx, _visited)

    def _init_weights(self, ctx: InitContext) -> None:
        """Initialize local state only; recursion is handled by init_weights()."""
        return

    def init_non_persistent_buffers(self, _visited: set[int] | None = None) -> None:
        if _visited is None:
            _visited = set()
        _rebuild_non_persistent_buffers(self, _visited)

    def _init_non_persistent_buffers(self) -> None:
        """Rebuild local runtime-only buffers."""
        return


def _init_module_tree(module: nn.Module, ctx: InitContext, visited: set[int]) -> None:
    if id(module) in visited:
        return
    visited.add(id(module))
    for child in module.children():
        _init_module_tree(child, ctx, visited)
    if isinstance(module, InitModule):
        module._init_weights(ctx)


def _rebuild_non_persistent_buffers(module: nn.Module, visited: set[int]) -> None:
    if id(module) in visited:
        return
    visited.add(id(module))
    for child in module.children():
        _rebuild_non_persistent_buffers(child, visited)
    init_buffers = type(module).__dict__.get("_init_non_persistent_buffers")
    if init_buffers is not None:
        init_buffers(module)
