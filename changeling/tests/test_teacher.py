from copy import deepcopy
import torch
from torch import cuda, nn, Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from changeling.core.branch import Branch
from changeling.core.changeling import Changeling
from changeling.core.layers import MeanInputLayer
from changeling.core.learning_criteria import NEpochsComplete
from changeling.core.teacher import Lesson, Teacher


class MyDataset(Dataset):
    def __init__(self):
        n_samples = 160
        self.data = torch.randn((n_samples, 3, 32, 32))
        self.labels = torch.randint(0, 10, (n_samples,))
        self.grayscale_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[dict[str, Tensor], Tensor]:
        color_img = self.data[idx]
        gray_img = self.grayscale_transform(color_img)
        return {"gray_branch": gray_img, "color_branch": color_img}, self.labels[idx]


def get_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if cuda.is_available() else 0,
        pin_memory=cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
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
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.MaxPool2d(2),
                nn.ReLU(),
            ),
            "color_branch": Branch(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.MaxPool2d(2),
                nn.ReLU(),
            ),
        }
        for name, branch in self.input_branches.items():
            self.add_module(name, branch)
        self.mean_layer = MeanInputLayer()
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(256, 10),
            nn.Softmax(dim=1),
        )
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        input_tensors = [
            branch(x[input_name])
            for input_name, branch
            in self.input_branches.items()
            if branch.active
        ]
        return self.hidden_layers(self.mean_layer(input_tensors))

    def prep_lesson(self, name: str) -> None:
        if name == "Lesson 0":
            self.input_branches["gray_branch"].activate()
            self.input_branches["color_branch"].deactivate()
        if name == "Lesson 1":
            self.input_branches["gray_branch"].freeze()
            self.input_branches["color_branch"].activate()
        if name == "Lesson 2":
            self.input_branches["gray_branch"].deactivate()

    def loss(self, output, labels) -> Tensor:
        return self.loss_function(output, labels)


def test_teacher():
    curriculum = [
        Lesson(
            name=f"Lesson {i}",
            get_dataloaders=lambda: get_dataloaders(MyDataset(), MyDataset(), 16),
            lesson_complete=NEpochsComplete(10),
        )
        for i in range(3)
    ]
    model = MyModel()
    teacher = Teacher(model, curriculum)

    # First lesson
    weights_before = {
        branch: [deepcopy(p) for p in model.input_branches[branch][0].parameters()]
        for branch in model.input_branches.keys()
    }
    teacher.teach_lesson(0)
    weights_after = {
        branch: [deepcopy(p) for p in model.input_branches[branch][0].parameters()]
        for branch in model.input_branches.keys()
    }
    assert model.input_branches["gray_branch"].active
    assert not model.input_branches["color_branch"].active
    assert not any(
        torch.equal(w1, w2)
        for w1, w2
        in zip(weights_before["gray_branch"], weights_after["gray_branch"])
    )
    assert all(
        torch.equal(w1, w2)
        for w1, w2
        in zip(weights_before["color_branch"], weights_after["color_branch"])
    )

    # Second lesson
    weights_before = weights_after
    teacher.teach_lesson(1)
    weights_after = {
        branch: [deepcopy(p) for p in model.input_branches[branch][0].parameters()]
        for branch in model.input_branches.keys()
    }
    assert model.input_branches["gray_branch"].active
    assert model.input_branches["color_branch"].active
    assert all(
        torch.equal(w1, w2)
        for w1, w2
        in zip(weights_before["gray_branch"], weights_after["gray_branch"])
    )
    assert not any(
        torch.equal(w1, w2)
        for w1, w2
        in zip(weights_before["color_branch"], weights_after["color_branch"])
    )

    # Third lesson
    weights_before = weights_after
    teacher.teach_lesson(2)
    weights_after = {
        branch: [deepcopy(p) for p in model.input_branches[branch][0].parameters()]
        for branch in model.input_branches.keys()
    }
    assert not model.input_branches["gray_branch"].active
    assert model.input_branches["color_branch"].active
    assert all(
        torch.equal(w1, w2)
        for w1, w2
        in zip(weights_before["gray_branch"], weights_after["gray_branch"])
    )
    assert not any(
        torch.equal(w1, w2)
        for w1, w2
        in zip(weights_before["color_branch"], weights_after["color_branch"])
    )
