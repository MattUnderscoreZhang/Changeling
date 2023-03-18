import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from changeling import Changeling

def test_mnist_training():
    # Transform images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_data = datasets.MNIST('data', train=True, transform=transform, download=True)

    # Define data loader
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Use only digits 1 and 2 for binary classification
    filtered_train_data = [d for d in train_data if d[1] in [1, 2]]

    # Define Changeling network
    net = Changeling(input_dims=28*28, output_dims=2, hidden_dims=[64, 64])

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Train for five epochs
    for epoch in range(5):
        running_loss = 0
        for i, data in enumerate(train_loader):
            # Separate inputs and labels
            inputs, labels = data
            # Use only digits 1 and 2 for binary classification
            inputs = inputs[labels <= 2].view(inputs[labels <= 2].shape[0], -1)
            labels = labels[labels <= 2]
            # Zero gradients
            optimizer.zero_grad()
            # Pass inputs through network
            outputs = net(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Backpropagate and optimize weights
            loss.backward()
            optimizer.step()
            # Update running loss
            running_loss += loss.item()

        # Print average loss for epoch
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(filtered_train_data)))
