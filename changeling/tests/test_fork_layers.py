import pytest
import torch
from torch import Tensor
from typing import Any, Dict

from changeling.layers import SumInputLayer, ConcatInputLayer, SplitOutputLayer


@pytest.fixture(params=[1, 2])
def inputs_and_output(request) -> Dict[str, Any]:
    dim = request.param
    total_features = 10
    inputs = (
        [torch.zeros(16, total_features) + i for i in range(3)]
        if dim == 1
        else [torch.zeros(16, total_features, 8, 8) + i for i in range(3)]
    )
    sum_output = (
        torch.zeros(16, total_features) + 3
        if dim == 1
        else torch.zeros(16, total_features, 8, 8) + 3
    )
    concat_output = (
        torch.cat([torch.zeros(16, total_features) + i for i in range(3)], dim=1)
        if dim == 1
        else torch.cat([torch.zeros(16, total_features, 8, 8) + i for i in range(3)], dim=1)
    )
    concat_output_shape = (
        (16, total_features * 3)
        if dim == 1
        else (16, total_features * 3, 8, 8)
    )
    return {
        "inputs": inputs,
        "sum_output": sum_output,
        "concat_output": concat_output,
        "concat_output_shape": concat_output_shape,
    }


def test_sum_input_layer(inputs_and_output: Dict[str, Any]):
    sum_input_layer = SumInputLayer()
    output = sum_input_layer.forward(inputs_and_output["inputs"])
    assert isinstance(output, Tensor)
    assert torch.allclose(output, inputs_and_output["sum_output"])


def test_concat_input_layer(inputs_and_output: Dict[str, Any]):
    concat_input_layer = ConcatInputLayer()
    output = concat_input_layer.forward(inputs_and_output["inputs"])
    assert isinstance(output, Tensor)
    assert output.shape == inputs_and_output["concat_output_shape"]
    assert torch.allclose(output, inputs_and_output["concat_output"])


@pytest.fixture(params=[1, 2])
def input_and_features_dict(request) -> Dict[str, Any]:
    dim = request.param
    total_features = 10
    input_tensor = (
        torch.randn(16, total_features)
        if dim == 1
        else torch.randn(16, total_features, 8, 8)
    )
    output_features_dict = {
        f"out_branch_{i}": 2
        for i in range(total_features // 2)
    }
    return {
        "input": input_tensor,
        "features_dict": output_features_dict,
    }


def test_split_output_layer(input_and_features_dict: Dict[str, Any]):
    x = input_and_features_dict["input"]
    features_dict = input_and_features_dict["features_dict"]
    layer = SplitOutputLayer(features_dict)
    output = layer(x)

    assert isinstance(output, Dict)
    assert len(output) == len(features_dict)

    for key, out_features in features_dict.items():
        assert key in output
        assert output[key].shape == (x.shape[0], out_features, *x.shape[2:])
    concat_input_layer = ConcatInputLayer()
    assert torch.allclose(concat_input_layer.forward(list(output.values())), x)
