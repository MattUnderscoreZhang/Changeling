import torch
from torch import cuda, nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from changeling.core.branch import Branch
from changeling.core.changeling import Changeling
from changeling.core.layers import ConcatInputLayer, MeanInputLayer
from changeling.core.teacher import Teacher, Lesson


class CIFARSubset(Dataset):
    def __init__(
        self,
        cifar_data: Dataset,
        labels_to_include: list[int],
    ):
        self.data = [
            (img, label)
            for img, label in cifar_data
            if label in labels_to_include
        ]
        self.grayscale_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[dict[str, Tensor], int]:
        color_img, label = self.data[idx]
        gray_img = self.grayscale_transform(color_img)
        return {"gray_branch": gray_img, "color_branch": color_img}, label


def get_dataloaders(
    cifar_train: Dataset,
    cifar_test: Dataset,
    batch_size: int,
    labels_to_include: list[int],
) -> tuple[DataLoader, DataLoader]:
    cifar_cur_train = CIFARSubset(
        cifar_train,
        labels_to_include,
    )
    cifar_cur_test = CIFARSubset(
        cifar_test,
        labels_to_include,
    )
    train_loader = DataLoader(
        cifar_cur_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if cuda.is_available() else 0,
        pin_memory=cuda.is_available(),
    )
    test_loader = DataLoader(
        cifar_cur_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if cuda.is_available() else 0,
        pin_memory=cuda.is_available(),
    )
    return train_loader, test_loader


class MyModel(Changeling):
    def __init__(self):
        super().__init__()
        self.input_branches = {
            "gray_branch": Branch(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            "color_branch": Branch(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
        }
        self.mean_layer = MeanInputLayer()
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.output_branches = {
            i: Branch(
                nn.Linear(128, 1)
            )
            for i in range(10)
        }
        self.concat_layer = ConcatInputLayer()

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        inputs = [
            self.input_branches[branch](x[branch])
            for branch in x.keys()
            if self.input_branches[branch].active
        ]
        mean_inputs = self.mean_layer(inputs)
        hidden_out = self.hidden_layers(mean_inputs)
        out_labels = [
            self.output_branches[i](hidden_out)
            if self.output_branches[i].active
            else torch.zeros(hidden_out.shape[0], 1)
            for i in range(10)
        ]
        return self.concat_layer(out_labels)

    def prep_lesson(self, name: str) -> None:
        if name.startswith("Grayscale Input"):
            self.input_branches["gray_branch"].activate()
            self.input_branches["gray_branch"].unfreeze()
            self.input_branches["color_branch"].deactivate()
        elif name.startswith("Dual Input"):
            self.input_branches["gray_branch"].activate()
            self.input_branches["gray_branch"].freeze()
            self.input_branches["color_branch"].activate()
        elif name.startswith("Color Input"):
            self.input_branches["gray_branch"].deactivate()
            self.input_branches["color_branch"].activate()
        labels = [
            int(i.strip())
            for i in name.split("[")[1].split("]")[0].split(",")
        ]
        for i in range(10):
            self.output_branches[i].deactivate()
        for i in labels:
            self.output_branches[i].activate()


def main():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_test = CIFAR10(root='./data', train=False, download=True, transform=transform)

    model = MyModel()
    curriculum_labels = [
        list(range(0, 2)),
        list(range(2, 4)),
        list(range(0, 4)),
        list(range(4, 6)),
        list(range(0, 6)),
        list(range(6, 8)),
        list(range(0, 8)),
        list(range(8, 10)),
        list(range(0, 10)),
    ]
    curriculum = [
        lesson
        for labels in curriculum_labels
        for lesson in
        [
            Lesson(
                name=f"Grayscale Input - {labels}",
                get_dataloaders=lambda labels=labels: get_dataloaders(
                    cifar_train, cifar_test, batch_size=128,
                    labels_to_include=labels,
                ),
                accuracy_threshold=0.7,
            ),
            Lesson(
                name=f"Dual Input - {labels}",
                get_dataloaders=lambda: get_dataloaders(
                    cifar_train, cifar_test, batch_size=128,
                    labels_to_include=labels,
                ),
                accuracy_threshold=0.8,
            ),
            Lesson(
                name=f"Color Input - {labels}",
                get_dataloaders=lambda: get_dataloaders(
                    cifar_train, cifar_test, batch_size=128,
                    labels_to_include=labels,
                ),
                accuracy_threshold=0.9,
            ),
        ]
    ]
    teacher = Teacher(model, curriculum)
    assert teacher.teach(max_epochs=100)

if __name__ == "__main__":
    main()
