import torch
import torch.nn as nn
from changeling import Changeling, ForkLayer, xavier_init


def test_changeling():
    # define Changeling network
    net = Changeling([
        xavier_init(nn.Linear(10, 16)),
        nn.ReLU(),
        xavier_init(nn.Linear(16, 8)),
        nn.ReLU(),
        xavier_init(nn.Linear(8, 4)),
        nn.ReLU(),
        xavier_init(nn.Linear(4, 3)),
    ])

    # test inputs and outputs
    assert net.in_features == 10
    assert net.out_features == 3

    # test freezing and unfreezing
    assert net.frozen == False
    net.freeze()
    assert net.frozen == True
    net.unfreeze()
    assert net.frozen == False


def test_forklayer():
    # define Changeling network
    input_1 = Changeling([
        xavier_init(nn.Linear(10, 16)),
        nn.ReLU(),
        xavier_init(nn.Linear(16, 8)),
        nn.ReLU(),
        xavier_init(nn.Linear(8, 4)),
        nn.ReLU(),
        xavier_init(nn.Linear(4, 3)),
    ])
    input_2 = Changeling([
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 3),
        nn.Tanh(),
    ])
    output_1 = Changeling([
        nn.Linear(3, 2),
        nn.Sigmoid(),
    ])
    output_2 = Changeling([
        nn.Linear(3, 5),
        nn.Sigmoid(),
    ])

    # test net setup
    net = ForkLayer(
        input={'input_1': input_1, 'input_2': input_2},
        output={'output_1': output_1, 'output_2': output_2},
    )
    assert set(net.input.keys()) == {'input_1', 'input_2'}
    assert set(net.output.keys()) == {'output_1', 'output_2'}
    assert net.input['input_1'].in_features == 10
    assert net.input['input_2'].in_features == 3
    assert net.output['output_1'].out_features == 2
    assert net.output['output_2'].out_features == 5
    output = net({'input_1': torch.randn(1, 10), 'input_2': torch.randn(1, 3)})
    assert set(output.keys()) == {'output_1', 'output_2'}
    assert output['output_1'].shape == (1, 2)
    assert output['output_2'].shape == (1, 5)

    # test removing inputs and outputs
    net.remove_input('input_1')
    net.remove_output('output_1')
    assert set(net.input.keys()) == {'input_2'}
    assert set(net.output.keys()) == {'output_2'}
    output = net({'input_2': torch.randn(1, 3)})
    assert set(output.keys()) == {'output_2'}
    assert output['output_2'].shape == (1, 5)

    # test adding inputs and outputs
    net.add_input('input_1', input_1)
    net.add_output('output_1', output_1)
    assert set(net.input.keys()) == {'input_1', 'input_2'}
    assert set(net.output.keys()) == {'output_1', 'output_2'}
    output = net({'input_1': torch.randn(1, 10), 'input_2': torch.randn(1, 3)})
    assert set(output.keys()) == {'output_1', 'output_2'}
    assert output['output_1'].shape == (1, 2)
    assert output['output_2'].shape == (1, 5)
