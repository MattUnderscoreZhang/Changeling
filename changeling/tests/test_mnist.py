from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import QMNIST
import torchvision.transforms as transforms
from typing import List, Tuple

from changeling.core.teacher import Teacher, Lesson


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


def get_dataloaders(
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


def test_mnist():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train = QMNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = QMNIST(root='./data', train=False, download=True, transform=transform)

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
    )
    curriculum = [
        Lesson(
            name=f"First {n} digits",
            get_dataloaders=lambda: get_dataloaders(
                mnist_train, mnist_test, batch_size=128, n_labels=n
            )
        )
        for n in range(1, 11)
    ]
    teacher = Teacher(model, curriculum)
    assert teacher.teach(max_epochs=100)
