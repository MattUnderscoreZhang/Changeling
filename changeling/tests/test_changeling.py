from typing import Dict
import pytest
import torch
import torch.nn as nn

from changeling.layers import Changeling


@pytest.fixture(params=["sequential", "linear", "conv2d"])
def changeling_data(request):
    if request.param == "sequential":
        seq_model = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        in_shape_expected= (10,)
        out_shape_expected= (4,)
    elif request.param == "linear":
        seq_model = nn.Linear(10, 4)
        in_shape_expected= (10,)
        out_shape_expected= (4,)
    elif request.param == "conv2d":
        seq_model = nn.Conv2d(3, 6, kernel_size=(3, 3))
        in_shape_expected= (3, 3, 3)
        out_shape_expected= (6, 3, 3)
    else:
        raise ValueError("Unsupported module type")

    c = Changeling(seq_model)
    request.param = {
        "instance": c,
        "in_shape_expected": in_shape_expected,
        "out_shape_expected": out_shape_expected,
    }
    return request.param


def test_freeze(changeling_data: Dict):
    changeling = changeling_data["instance"]
    changeling.freeze()
    assert all([not p.requires_grad for p in changeling.parameters()])


def test_unfreeze(changeling_data: Dict):
    changeling = changeling_data["instance"]
    changeling.freeze()
    changeling.unfreeze()
    assert all([p.requires_grad for p in changeling.parameters()])


def test_deactivate(changeling_data: Dict):
    changeling = changeling_data["instance"]
    changeling.deactivate()
    assert not changeling.active


def test_activate(changeling_data: Dict):
    changeling = changeling_data["instance"]
    changeling.deactivate()
    changeling.activate()
    assert changeling.active


def test_in_shape(changeling_data: Dict):
    changeling = changeling_data["instance"]
    assert changeling.in_shape == changeling_data["in_shape_expected"]


def test_out_shape(changeling_data: Dict):
    changeling = changeling_data["instance"]
    assert changeling.out_shape == changeling_data["out_shape_expected"]


def test_forward(changeling_data: Dict):
    changeling = changeling_data["instance"]
    if isinstance(changeling.module, nn.Sequential):
        batch_size = 32
        x = torch.randn(batch_size, 10)
        assert changeling(x).shape == (batch_size, 4)
    elif isinstance(changeling.module, nn.Linear):
        batch_size = 32
        x = torch.randn(batch_size, 10)
        assert changeling(x).shape == (batch_size, 4)
    elif isinstance(changeling.module, nn.Conv2d):
        batch_size = 32
        x = torch.randn(batch_size, 3, 32, 32)
        assert changeling(x).shape == (batch_size, 6, 30, 30)
