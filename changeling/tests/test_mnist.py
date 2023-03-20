from typing import List, Tuple
import torch
from torch import device, nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import QMNIST
import torchvision.transforms as transforms


class MNISTSubset(Dataset):
    def __init__(self, mnist_data: Dataset, labels_to_include: List[int]):
        self.data = []
        for img, label in mnist_data:
            if label in labels_to_include:
                self.data.append((img, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: device
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


def test(model: nn.Module, test_loader: DataLoader, device: device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def get_data_loaders(
    mnist_train: Dataset,
    mnist_test: Dataset,
    batch_size: int,
    n_labels: int,
) -> Tuple[DataLoader, DataLoader]:
    max_labels = 10
    labels = list(range(max_labels))
    mnist_cur_train = MNISTSubset(mnist_train, labels[:n_labels])
    mnist_cur_test = MNISTSubset(mnist_test, labels[:n_labels])
    train_loader = DataLoader(mnist_cur_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_cur_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_mnist() -> Tuple[int, float]:
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train = QMNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = QMNIST(root='./data', train=False, download=True, transform=transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    max_epochs = 100
    batch_size = 128
    n_labels = 2
    accuracies = []
    accuracy_threshold = 0.95
    consecutive_epochs_threshold = 3
    n_consecutive_epochs = 0
    train_loader, test_loader = get_data_loaders(mnist_train, mnist_test, batch_size, n_labels)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch} - ", end = "")
        train_loss = train(model, train_loader, loss_function, optimizer, device)
        test_accuracy = test(model, test_loader, device)
        accuracies.append(test_accuracy)
        print(f"Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        n_consecutive_epochs = (
            n_consecutive_epochs + 1
            if test_accuracy >= accuracy_threshold
            else 0
        )
        if n_consecutive_epochs >= consecutive_epochs_threshold:
            if n_labels + 1 > 10:
                break
            n_labels += 1
            print(f"Adding label {n_labels} to training set.")
            train_loader, test_loader = get_data_loaders(
                mnist_train, mnist_test, batch_size, n_labels
            )
    print("Training complete")
    final_accuracy = accuracies[-1]
    print(f"Final accuracy: {final_accuracy:.4f}")
    return n_labels, final_accuracy


def test_mnist():
    n_labels, final_accuracy = train_mnist()
    assert n_labels == 10
    assert final_accuracy >= 0.95
