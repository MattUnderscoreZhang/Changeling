from torch import nn


def xavier_init(layer: nn.Linear) -> nn.Linear:
    # Initialize layer weights using Xavier initialization
    nn.init.xavier_uniform_(layer.weight.data)
    nn.init.zeros_(layer.bias.data)
    return layer
