from typing import Dict
import pytest
import torch
import torch.nn as nn

from changeling.core.branch import Branch


@pytest.fixture(params=["linear", "conv2d"])
def branch_data(request):
    if request.param == "linear":
        seq_model = Branch(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        in_features_expected = 10
        out_features_expected = 4
        input = torch.rand(16, 10)
        expected_output_size = torch.Size([16, 4])
    elif request.param == "conv2d":
        seq_model = nn.Conv2d(3, 6, kernel_size=(3, 3))
        seq_model = Branch(
            nn.Conv2d(10, 6, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(6, 4, kernel_size=(3, 3)),
            nn.ReLU(),
        )
        in_features_expected = 10
        out_features_expected = 4
        input = torch.rand(16, 10, 8, 8)
        expected_output_size = torch.Size([16, 4, 2, 2])
    else:
        raise ValueError("Unsupported module type")

    return {
        "branch": seq_model,
        "in_features_expected": in_features_expected,
        "out_features_expected": out_features_expected,
        "input": input,
        "expected_output_size": expected_output_size,
    }


def test_freeze(branch_data: Dict):
    branch = branch_data["branch"]
    branch.freeze()
    assert all([not p.requires_grad for p in branch.parameters()])


def test_unfreeze(branch_data: Dict):
    branch = branch_data["branch"]
    branch.freeze()
    branch.unfreeze()
    assert all([p.requires_grad for p in branch.parameters()])


def test_deactivate(branch_data: Dict):
    branch = branch_data["branch"]
    branch.deactivate()
    assert not branch.active


def test_activate(branch_data: Dict):
    branch = branch_data["branch"]
    branch.deactivate()
    branch.activate()
    assert branch.active


def test_in_features(branch_data: Dict):
    branch = branch_data["branch"]
    assert branch.in_features == branch_data["in_features_expected"]


def test_out_features(branch_data: Dict):
    branch = branch_data["branch"]
    assert branch.out_features == branch_data["out_features_expected"]


def test_forward(branch_data: Dict):
    branch = branch_data["branch"]
    assert branch(branch_data["input"]).size() == branch_data["expected_output_size"]
