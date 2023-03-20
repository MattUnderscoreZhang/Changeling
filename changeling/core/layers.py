import torch
from torch import nn, Tensor
from typing import Dict, List


class SumInputLayer(nn.Module):
    def forward(self, x: List[Tensor]) -> Tensor:
        # Check if all inputs have consistent sizes
        input_shapes = [tensor.shape for tensor in x]
        assert all([input_shapes[0][1:] == shape[1:] for shape in input_shapes])

        input_sum = torch.sum(torch.stack(x, dim=0), dim=0)
        return input_sum


class ConcatInputLayer(nn.Module):
    def forward(self, x: List[Tensor]) -> Tensor:
        # Check if all inputs have the same dimension
        input_shapes = [tensor.shape for tensor in x]
        assert all([input_shapes[0][1:] == shape[1:] for shape in input_shapes])

        return torch.cat(x, dim=1)


class SplitOutputLayer(nn.Module):
    def __init__(self, out_features: Dict[str, int]):
        super().__init__()
        self.out_features = out_features

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out_tensors = {}
        start = 0
        for name, out_features in self.out_features.items():
            end = start + out_features
            out_tensors[name] = x[:, start:end]
            start = end
        return out_tensors
