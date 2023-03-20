import torch
from torch import nn
from typing import Dict, Tuple


class Changeling(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.active = True
        self.frozen = False
        self.unfreeze()

    def freeze(self) -> None:
        self._freeze_weights(self.module, True)
        self.frozen = True

    def unfreeze(self) -> None:
        self._freeze_weights(self.module, False)
        self.frozen = False

    def activate(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.active = False

    def _freeze_weights(self, module: nn.Module, freeze: bool) -> None:
        for param in module.parameters():
            param.requires_grad = not freeze

    def _get_layer_shape(self, layer: nn.Module, in_or_out: str) -> Tuple[int, ...]:
        if isinstance(layer, nn.Linear):
            shape = (getattr(layer, f"{in_or_out}_features"), )
            return shape
        elif isinstance(layer, nn.Conv2d):
            channels = getattr(layer, f"{in_or_out}_channels")
            kernel_size = layer.kernel_size
            kernel_height, kernel_width = (
                kernel_size
                if isinstance(kernel_size, tuple)
                else (kernel_size, kernel_size)
            )
            shape = (int(channels), int(kernel_height), int(kernel_width))
            return shape

        raise ValueError("Unsupported module type")

    def _first_layer(self) -> nn.Module:
        if isinstance(self.module, nn.Sequential):
            return self.module[0]
        return self.module

    def _last_layer(self) -> nn.Module:
        if isinstance(self.module, nn.Sequential):
            return self.module[-1]
        return self.module

    @property
    def in_shape(self) -> Tuple[int, ...]:
        first_layer = self._first_layer()
        return self._get_layer_shape(first_layer, "in")

    @property
    def out_shape(self) -> Tuple[int, ...]:
        last_layer = self._last_layer()
        return self._get_layer_shape(last_layer, "out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return x
        return self.module(x)


class ChangelingDict(nn.ModuleDict):
    def __init__(self, layers: Dict[str, Changeling]):
        super().__init__(layers)
