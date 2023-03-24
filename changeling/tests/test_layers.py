import pytest
import torch
from torch import nn, Tensor
from typing import Any, Dict

from changeling.core.branch import Branch
from changeling.core.layers import (
    MeanInputLayer,
    SumInputLayer,
    ConcatInputLayer,
    SplitOutputLayer,
    BroadcastOutputLayer,
)


@pytest.fixture(params=[1, 2])
def inputs_and_output(request) -> Dict[str, Any]:
    dim = request.param
    batch_size = 16
    total_features = 10
    inputs = (
        [torch.zeros(batch_size, total_features) + i for i in range(3)]
        if dim == 1
        else [torch.zeros(batch_size, total_features, 8, 8) + i for i in range(3)]
    )
    sum_output = (
        torch.zeros(batch_size, total_features) + 3
        if dim == 1
        else torch.zeros(batch_size, total_features, 8, 8) + 3
    )
    concat_output = (
        torch.cat([torch.zeros(batch_size, total_features) + i for i in range(3)], dim=1)
        if dim == 1
        else torch.cat([torch.zeros(batch_size, total_features, 8, 8) + i for i in range(3)], dim=1)
    )
    concat_output_shape = (
        (batch_size, total_features * 3)
        if dim == 1
        else (batch_size, total_features * 3, 8, 8)
    )
    return {
        "inputs": inputs,
        "sum_output": sum_output,
        "mean_output": sum_output / 3,
        "concat_output": concat_output,
        "concat_output_shape": concat_output_shape,
    }


def test_sum_input_layer(inputs_and_output: Dict[str, Any]):
    sum_input_layer = SumInputLayer()
    output = sum_input_layer.forward(inputs_and_output["inputs"])
    assert isinstance(output, Tensor)
    assert torch.allclose(output, inputs_and_output["sum_output"])


def test_mean_input_layer(inputs_and_output: Dict[str, Any]):
    mean_input_layer = MeanInputLayer()
    output = mean_input_layer.forward(inputs_and_output["inputs"])
    assert isinstance(output, Tensor)
    assert torch.allclose(output, inputs_and_output["mean_output"])


def test_concat_input_layer(inputs_and_output: Dict[str, Any]):
    concat_input_layer = ConcatInputLayer()
    output = concat_input_layer.forward(inputs_and_output["inputs"])
    assert isinstance(output, Tensor)
    assert output.shape == inputs_and_output["concat_output_shape"]
    assert torch.allclose(output, inputs_and_output["concat_output"])


@pytest.fixture(params=[1, 2])
def split_output_data(request) -> Dict[str, Any]:
    dim = request.param
    batch_size = 16
    in_features = 16
    out_branch_features = {
        "out_0": 8,
        "out_1": 4,
        "out_2": 2,
        "out_3": 2,
    }
    input = (
        torch.zeros(batch_size, in_features)
        if dim == 1
        else torch.zeros(batch_size, in_features, 8, 8)
    )
    out_branches = (
        {
            name: Branch(nn.Linear(in_features, 1))
            for name, in_features in out_branch_features.items()
        }
        if dim == 1
        else {
            name: Branch(nn.Conv2d(in_features, 1, 3, 3, padding=1))
            for name, in_features in out_branch_features.items()
        }
    )
    out_branches["out_0"].deactivate()
    out_branches["out_2"].deactivate()
    return {"out_branches": out_branches, "input": input}


def test_split_output_layer(split_output_data: Dict[str, Any]):
    input = split_output_data["input"]
    out_branches = split_output_data["out_branches"]
    layer = SplitOutputLayer(out_branches)
    output = layer(input)

    assert isinstance(output, Dict)
    assert len(output) == len(out_branches)
    assert output.keys() == out_branches.keys()
    assert output["out_0"] == Tensor([-1])
    assert output["out_1"].shape[:2] == (16, 1)
    assert output["out_2"] == Tensor([-1])
    assert output["out_3"].shape[:2] == (16, 1)


@pytest.fixture(params=[1, 2])
def broadcast_output_data(request) -> Dict[str, Any]:
    dim = request.param
    batch_size = 16
    in_features = 16
    out_branch_features = {
        "out_0": 8,
        "out_1": 4,
        "out_2": 2,
        "out_3": 2,
    }
    input = (
        torch.zeros(batch_size, in_features)
        if dim == 1
        else torch.zeros(batch_size, in_features, 8, 8)
    )
    out_branches = (
        {
            name: Branch(nn.Linear(in_features, out_features))
            for name, out_features in out_branch_features.items()
        }
        if dim == 1
        else {
            name: Branch(nn.Conv2d(in_features, out_features, 3, 3, padding=1))
            for name, out_features in out_branch_features.items()
        }
    )
    out_branches["out_0"].deactivate()
    out_branches["out_2"].deactivate()
    return {"out_branches": out_branches, "input": input}


def test_broadcast_output_layer(broadcast_output_data: Dict[str, Any]):
    input = broadcast_output_data["input"]
    out_branches = broadcast_output_data["out_branches"]
    layer = BroadcastOutputLayer(out_branches)
    output = layer(input)

    assert isinstance(output, Dict)
    assert len(output) == len(out_branches)
    assert output.keys() == out_branches.keys()
    assert output["out_0"] == Tensor([-1])
    assert output["out_1"].shape[:2] == (16, 4)
    assert output["out_2"] == Tensor([-1])
    assert output["out_3"].shape[:2] == (16, 2)
