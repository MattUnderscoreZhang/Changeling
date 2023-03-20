import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms

"""
from layers import BroadcastOutputLayer, Changeling, SumInputLayer


def test_mnist_training():
    # load MNIST dataset
    data = datasets.MNIST(
        'data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    # define Changeling network
    def digit_output_layer() -> Changeling:
        return Changeling([nn.Linear(128, 1)])

    mnist_image_input = Changeling([nn.Linear(784, 128)])
    digit_ouput_layer = BroadcastOutputLayer({})
    digit_output_layers = {
        str(digit): digit_output_layer()
        for digit in range(10)
    }
    digit_output_layer.add_output('0', digit_output_layers['0'])
    digit_softmax_layer = SumInputLayer(
        input=digit_output_layers,
        output=
    )

    # train, adding more digits as the net learns to recognize them
    for max_digit in range(1, 10):
        # change training data
        filtered_data = [d for d in data if d[1] in range(max_digit)]
        filtered_data = TensorDataset(
            *[d[0] for d in filtered_data],
            *[d[1] for d in filtered_data],
        )
        train_data, test_data = random_split(filtered_data, [50000, 10000])
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(image_input_layer.parameters(), lr=0.001)

        # change net architecture
        image_input_layer.add_output(str(max_digit), digit_output_layer())

        # train until loss is less than 0.1
        avg_loss = np.inf
        running_loss = 0
        while avg_loss > 0.1:
            running_loss = 0
            for _, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.view(inputs.shape[0], -1)
                optimizer.zero_grad()
                outputs = image_input_layer({'mnist_image': inputs})
                loss = criterion(outputs['0'].squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_data)
            print(f"Max digit: {max_digit}, Loss: {avg_loss}")

        # evaluate on test_data after each epoch
        with torch.no_grad():
            test_loss = 0
            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs = inputs.view(inputs.shape[0], -1)
                labels = labels <= max_digit
                outputs = image_input_layer({'mnist_image': inputs})
                test_loss += criterion(outputs['0'].squeeze(), labels.float()).item()

        test_loss = test_loss / len(test_data)
        print(f"Max digit: {max_digit}, Test Loss: {test_loss}") 

"""
