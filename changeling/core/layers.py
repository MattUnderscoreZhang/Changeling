import torch
from torch import nn, Tensor
from typing import Any

from changeling.core.branch import Branch


class SumInputLayer(nn.Module):
    def forward(self, x: list[Tensor]) -> Tensor:
        # Check if all inputs have consistent sizes
        input_shapes = [tensor.shape for tensor in x]
        assert all([input_shapes[0][1:] == shape[1:] for shape in input_shapes])

        input_sum = torch.sum(torch.stack(x, dim=0), dim=0)
        return input_sum


class ConcatInputLayer(nn.Module):
    def forward(self, x: list[Tensor]) -> Tensor:
        # Check if all inputs have the same dimension
        input_shapes = [tensor.shape for tensor in x]
        assert all([input_shapes[0][1:] == shape[1:] for shape in input_shapes])

        return torch.cat(x, dim=1)


class SplitOutputLayer(nn.Module):
    def __init__(self, out_branches: dict[Any, Branch]):
        super().__init__()
        self.out_branches = out_branches

    def forward(self, x: Tensor) -> dict[Any, Tensor]:
        out_tensors = {}
        start = 0
        for name, out_branch in self.out_branches.items():
            end = start + out_branch.in_features
            out_tensors[name] = (
                out_branch(x[:, start:end])
                if out_branch.active
                else torch.Tensor([-1])
            )
            start = end
        return out_tensors


# TODO: write ExpandableLinearLayer
