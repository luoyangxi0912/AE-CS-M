from __future__ import annotations

import math
from typing import Iterable, Sequence

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None

    class _FallbackModule:
        pass

    class _FallbackNN:
        Module = _FallbackModule
        Sequential = object

    nn = _FallbackNN()


class TorchDependencyError(ImportError):
    pass


def require_torch():
    if torch is None:
        raise TorchDependencyError(
            "Mentor AE-family baselines require PyTorch. Install torch in the active Python environment before using these baselines."
        )
    return torch, nn


def default_device(device: str | torch.device | None = None) -> torch.device:
    torch_mod, _ = require_torch()
    if device is not None:
        return torch_mod.device(device)
    return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")


def resolve_struct(struct: Sequence[int | str]) -> list[int]:
    resolved: list[int] = []
    for item in struct:
        if isinstance(item, int):
            resolved.append(item)
            continue
        if not resolved:
            raise ValueError(f"Relative layer size {item!r} cannot be the first struct entry.")
        prev = resolved[-1]
        if item.startswith("/"):
            resolved.append(max(1, int(prev / float(item[1:]))))
        elif item.startswith("*"):
            resolved.append(max(1, int(prev * float(item[1:]))))
        else:
            resolved.append(int(item))
    return resolved


def activation(name: str | None) -> nn.Module | None:
    _, nn_mod = require_torch()
    if name is None or name in {"", "linear", "none"}:
        return None
    if name in {"lr", "leaky_relu"}:
        return nn_mod.LeakyReLU(negative_slope=0.2)
    if name in {"r", "relu"}:
        return nn_mod.ReLU()
    if name in {"t", "tanh"}:
        return nn_mod.Tanh()
    if name in {"s", "sigmoid"}:
        return nn_mod.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


def make_mlp(
    layer_sizes: Sequence[int],
    hidden_activation: str | None = "lr",
    output_activation: str | None = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    _, nn_mod = require_torch()
    layers: list[nn.Module] = []
    for idx in range(len(layer_sizes) - 1):
        layers.append(nn_mod.Linear(layer_sizes[idx], layer_sizes[idx + 1]))
        is_last = idx == len(layer_sizes) - 2
        act = activation(output_activation if is_last else hidden_activation)
        if act is not None:
            layers.append(act)
        if not is_last and dropout > 0:
            layers.append(nn_mod.Dropout(p=dropout))
    return nn_mod.Sequential(*layers)


def bounded_hidden_dim(input_dim: int, hidden_dim: int | None = None) -> int:
    if hidden_dim is not None:
        return max(1, int(hidden_dim))
    return max(1, int(math.ceil(input_dim / 2)))


class MentorModule(nn.Module):
    def __init__(self, device: str | torch.device | None = None):
        super().__init__()
        self.dvc = default_device(device)

    def move_to_device(self):
        self.to(self.dvc)
        return self

    def trainable_parameters(self) -> Iterable:
        return (p for p in self.parameters() if p.requires_grad)
