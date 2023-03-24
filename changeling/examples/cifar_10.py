from torch import cuda, nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from changeling.core.branch import Branch
from changeling.core.changeling import Changeling
from changeling.core.layers import ConcatInputLayer, SumInputLayer
from changeling.core.teacher import Teacher, Lesson


class CIFARSubset(Dataset):
    def __init__(
        self,
        cifar_data: Dataset,
        labels_to_include: list[int],
        labels_to_boost: list[int],
    ):
        self.data = [
            (img, label)
            for img, label in cifar_data
            if label in labels_to_include
            for _ in range(1 + 4 * (label in labels_to_boost))
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
    labels_to_boost: list[int],
) -> tuple[DataLoader, DataLoader]:
    cifar_cur_train = CIFARSubset(
        cifar_train,
        labels_to_include,
        labels_to_boost,
    )
    cifar_cur_test = CIFARSubset(
        cifar_test,
        labels_to_include,
        labels_to_boost,
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
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
        }
        self.sum_layer = SumInputLayer()
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.output_branches = {
            i: Branch(
                nn.Linear(512, 1)
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
        sum_inputs = self.sum_layer(inputs)
        hidden_out = self.hidden_layers(sum_inputs)
        out_labels = [
            self.output_branches[i](hidden_out)
            for i in range(10)
            if self.output_branches[i].active
        ]
        return self.concat_layer(out_labels)

    def prep_lesson(self, name: str) -> None:
        if name.startswith("Grayscale Input"):
            self.input_branches["gray_branch"].activate()
            self.input_branches["color_branch"].deactivate()
            n_labels = int(name.split(" ")[-2])
            for i in range(10):
                if i < n_labels:
                    self.output_branches[i].activate()
                else:
                    self.output_branches[i].deactivate()
        elif name == "Dual Input":
            self.input_branches["gray_branch"].activate()
            self.input_branches["color_branch"].activate()
            for i in range(10):
                self.output_branches[i].activate()
        elif name == "Color Input":
            self.input_branches["gray_branch"].deactivate()
            self.input_branches["color_branch"].activate()
            for i in range(10):
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
    curriculum = [
        Lesson(
            name=f"Grayscale Input - {i} Labels",
            get_dataloaders=lambda i=i: get_dataloaders(
                cifar_train, cifar_test, batch_size=128,
                labels_to_include=list(range(10))[:i],
                labels_to_boost=list(range(10))[i-2:i],
            ),
            accuracy_threshold=0.8,
        )
        for i in range(2, 11, 2)
    ] + [
        Lesson(
            name="Dual Input",
            get_dataloaders=lambda: get_dataloaders(
                cifar_train, cifar_test, batch_size=128,
                labels_to_include=list(range(10)),
                labels_to_boost=[],
            ),
            accuracy_threshold=0.8,
        )
    ] + [
        Lesson(
            name="Color Input",
            get_dataloaders=lambda: get_dataloaders(
                cifar_train, cifar_test, batch_size=128,
                labels_to_include=list(range(10)),
                labels_to_boost=[],
            ),
            accuracy_threshold=0.95,
        )
    ]
    teacher = Teacher(model, curriculum)
    assert teacher.teach(max_epochs=100)

if __name__ == "__main__":
    main()
