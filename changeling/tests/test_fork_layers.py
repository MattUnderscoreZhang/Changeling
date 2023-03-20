from typing import List
import pytest
import torch
from torch import Tensor
from changeling.layers import SumInputLayer, ConcatInputLayer


# Dummy input tensors
dummy_inputs = [torch.rand(16, 8, 8) for _ in range(3)]


@pytest.mark.parametrize("x", [dummy_inputs])
def test_sum_input_layer(x: List[Tensor]):
    sum_input_layer = SumInputLayer()
    output = sum_input_layer.forward(x)

    assert type(output) == torch.Tensor
    assert output.shape == x[0].shape
    assert torch.allclose(output, sum(x))


@pytest.mark.parametrize("x", [dummy_inputs])
def test_concat_input_layer(x: List[Tensor]):
    concat_input_layer = ConcatInputLayer()
    output = concat_input_layer.forward(x)

    assert type(output) == torch.Tensor
    assert output.shape == torch.Size((x[0].shape[0], x[0].shape[1]*len(x), x[0].shape[2]))
    assert torch.allclose(output, torch.cat(x, dim=1))
