from torch import cuda, nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import QMNIST
import torchvision.transforms as transforms

from changeling.core.changeling import Changeling
from changeling.core.learning_criteria import AccuracyThresholdAchieved
from changeling.core.teacher import Teacher, Lesson


class MNISTSubset(Dataset):
    def __init__(self, mnist_data: Dataset, labels_to_include: list[int]):
        self.data = [
            (img, label)
            for img, label in mnist_data
            if label in labels_to_include
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, int]:
        return self.data[idx]


def get_dataloaders(
    mnist_train: Dataset,
    mnist_test: Dataset,
    batch_size: int,
    n_labels: int,
) -> tuple[DataLoader, DataLoader]:
    max_labels = 10
    labels = list(range(max_labels))
    mnist_cur_train = MNISTSubset(mnist_train, labels[:n_labels])
    mnist_cur_test = MNISTSubset(mnist_test, labels[:n_labels])
    train_loader = DataLoader(
        mnist_cur_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if cuda.is_available() else 0,
        pin_memory=cuda.is_available(),
    )
    test_loader = DataLoader(
        mnist_cur_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if cuda.is_available() else 0,
        pin_memory=cuda.is_available(),
    )
    return train_loader, test_loader


class MyModel(Changeling):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
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
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def prep_lesson(self, name: str) -> None:
        pass

    def loss(self, outputs: Tensor, labels: Tensor) -> Tensor:
        return self.loss_function(outputs, labels)


def main():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train = QMNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = QMNIST(root='./data', train=False, download=True, transform=transform)

    model = MyModel()
    curriculum = [
        Lesson(
            name=f"First {n} digits",
            get_dataloaders=lambda: get_dataloaders(
                mnist_train, mnist_test, batch_size=128, n_labels=n
            ),
            lesson_complete=AccuracyThresholdAchieved(0.95),
        )
        for n in range(1, 11)
    ]
    teacher = Teacher(model, curriculum)
    assert teacher.teach(max_epochs=100)


if __name__ == "__main__":
    main()
