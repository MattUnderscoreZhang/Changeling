import torch
from torch import nn


class Branch(nn.Sequential):
    def __init__(self, *args: nn.Module):
        super().__init__(*args)

        # Apply Xavier initialization
        for module in [m for m in self if self._is_supported_layer(m)]:
            assert isinstance(module.weight, torch.Tensor)
            assert isinstance(module.bias, torch.Tensor)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        self.active = True
        self.frozen = False
        self.unfreeze()

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        self.frozen = False

    def activate(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.active = False

    def _get_n_features(self, layer: nn.Module, in_or_out: str) -> int:
        if isinstance(layer, nn.Linear):
            return getattr(layer, f"{in_or_out}_features")
        elif isinstance(layer, nn.Conv2d):
            return getattr(layer, f"{in_or_out}_channels")
        raise ValueError("Unsupported module type")

    def _is_supported_layer(self, layer: nn.Module) -> bool:
        return (
            isinstance(layer, nn.Linear) or
            isinstance(layer, nn.Conv2d)
        )

    @property
    def in_features(self) -> int:
        first_layer = [l for l in self if self._is_supported_layer(l)][0]
        return self._get_n_features(first_layer, "in")

    @property
    def out_features(self) -> int:
        last_layer = [l for l in self if self._is_supported_layer(l)][-1]
        return self._get_n_features(last_layer, "out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            super().forward(x)
            if self.active
            else torch.zeros((x.shape[0], self.out_features))
        )
