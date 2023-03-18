import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from typing import Dict, List


class Changeling(nn.Module):
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.frozen = False
        self.active = True
        self.in_features = layers[0].in_features
        self.out_features = 0
        activation_layers = [nn.ReLU, nn.Sigmoid, nn.Tanh]
        for layer in reversed(layers):
            if not any([isinstance(layer, activation) for activation in activation_layers]):
                self.out_features = layer.out_features
                break
        self.layers = layers

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.frozen = False


SequenceDict = Dict[str, Changeling]


class ForkLayer(nn.Module):
    def __init__(self, input: SequenceDict, output: SequenceDict):
        super().__init__()
        self.input = input
        self.output = output
        self.inactive_outputs = List[str]

        # assert all inputs and outputs have consistent sizes
        out_features = [sequence.out_features for sequence in self.input.values()]
        assert all([out_features[0] == features for features in out_features])
        in_features = [sequence.in_features for sequence in self.output.values()]
        assert all([in_features[0] == features for features in in_features])

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = [self.input[name](x[name]) for name in x]
        input_sum = torch.sum(torch.stack(inputs, dim=0), dim=0)
        outputs = {
            name: self.output[name](input_sum)
            for name in self.output if not self.output[name].frozen
        }
        return outputs

    def add_input(self, name: str, input: Changeling):
        self.input[name] = input

    def add_output(self, name: str, output: Changeling):
        self.output[name] = output

    def remove_input(self, name: str):
        if name in self.input:
            self.input.pop(name)

    def remove_output(self, name: str):
        if name in self.output:
            self.output.pop(name)


def xavier_init(layer: nn.Linear) -> nn.Linear:
    # Initialize layer weights using Xavier initialization
    init.xavier_uniform_(layer.weight.data)
    init.zeros_(layer.bias.data)
    return layer
