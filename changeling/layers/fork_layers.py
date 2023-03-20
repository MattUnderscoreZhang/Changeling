import sys
import torch
import torch.nn as nn
from typing import Dict

from changeling.layers import Changeling, ChangelingDict


class ForkLayer(nn.Module):
    def __init__(self, inputs: ChangelingDict, outputs: ChangelingDict):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

    def add_input(self, name: str, input: Changeling):
        self.inputs[name] = input

    def remove_input(self, name: str):
        if name in self.inputs:
            self.inputs.pop(name)

    def add_output(self, name: str, output: Changeling):
        self.outputs[name] = output

    def remove_output(self, name: str):
        if name in self.outputs:
            self.outputs.pop(name)


class SumInputLayer(ForkLayer):
    def __init__(self, inputs: ChangelingDict):
        super().__init__(inputs, ChangelingDict({}))

        # assert all inputs have consistent sizes
        out_features = [sequence.out_features for sequence in self.inputs.values()]
        assert all([out_features[0] == features for features in out_features])

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = [
            self.inputs[name](x[name])
            for name in x
            if self.inputs[name].active
        ]
        if len(x) != len(inputs):
            print(
                "Warning: inputs to SumInputLayer did not match expected active inputs",
                file=sys.stderr
            )
        input_sum = torch.sum(torch.stack(inputs, dim=0), dim=0)
        return input_sum


class ConcatInputLayer(ForkLayer):
    def __init__(self, inputs: ChangelingDict):
        super().__init__(inputs, ChangelingDict({}))

        # assert all inputs have the same dimension
        out_features = [sequence.out_features for sequence in self.inputs.values()]
        assert all([
            out_features[0].shape[2:] == features.shape[2:]
            for features in out_features
        ])

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = [
            self.inputs[name](x[name])
            for name in x
            if name not in self.inactive
        ]
        if len(x) != len(inputs):
            print(
                "Warning: inputs to SumInputLayer did not match expected active inputs",
                file=sys.stderr
            )
        return torch.cat(inputs, dim=1)


class BroadcastOutputLayer(ForkLayer):
    def __init__(self, outputs: ChangelingDict):
        super().__init__(ChangelingDict({}), outputs)

        # assert all outputs have consistent sizes
        in_features = [sequence.in_features for sequence in self.outputs.values()]
        assert all([in_features[0] == features for features in in_features])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        layers = {
            name: self.outputs[name](x)
            for name in self.outputs
        }
        return layers
