import torch.nn as nn

from changeling.layers import (
    ForkLayer,
)
from changeling.layers.changeling import Changeling, ChangelingDict


def test_fork_layer():
    # define InputForkLayer with mocked inputs
    mock_input1 = Changeling(nn.Linear(10, 3))
    mock_input2 = Changeling(
        nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
        )
    )
    mock_output1 = Changeling(nn.Linear(3, 1))
    mock_output2 = Changeling(
        nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )
    )
    layer = ForkLayer(
        inputs = ChangelingDict({
            "input1": mock_input1,
            "input2": mock_input2,
        }),
        outputs = ChangelingDict({
            "output1": mock_output1,
            "output2": mock_output2,
        }),
    )

    # add and remove inputs
    assert set(layer.inputs.keys()) == {"input1", "input2"}
    layer.add_input("input3", Changeling(nn.Linear(10, 1)))
    assert set(layer.inputs.keys()) == {"input1", "input2", "input3"}
    layer.remove_input("input2")
    assert set(layer.inputs.keys()) == {"input1", "input3"}

    # add and remove outputs
    assert set(layer.outputs.keys()) == {"output1", "output2"}
    layer.add_output("output3", Changeling(nn.Linear(3, 1)))
    assert set(layer.outputs.keys()) == {"output1", "output2", "output3"}
    layer.remove_output("output2")
    assert set(layer.outputs.keys()) == {"output1", "output3"}
