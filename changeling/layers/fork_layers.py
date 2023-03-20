import torch
from torch import nn, Tensor
from typing import List


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


# TODO: create SplitOutputLayer which takes a Dict[str, shape] init and a Tensor forward
# splits Tensor into Dict[str, Tensor] according to shapes
