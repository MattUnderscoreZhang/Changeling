# Changeling

Changeling is a neural network framework that allows for easy construction and manipulation of networks with multiple inputs and outputs. Parts of the network can be activated, deactivated, or frozen during training, allowing the user to train parts of the net individually. An example use case is training an RL agent that learns on progressively harder tasks. The agent can begin with a small neural net, as well as a reduced input and action space. As the agent learns tasks, its neural net can be expanded with more inputs, action outputs, and hidden neurons.

## Usage

To use, first import the necessary classes and methods and define your neural net architecture using the `Changeling` class. Changeling inherits from torch.nn.Module, and has the same rules for parameter registration.

```
from torch import nn
from changeling.core.branch import Branch
from changeling.core.changeling import Changeling
from changeling.core.layers import MeanInputLayer


class MyModel(Changeling):
    def __init__(self):
        super().__init__()
        self.input_branches = {
            "gray_image": Branch(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            "color_image": Branch(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
        }
        for name, branch in self.input_branches.items():
            self.add_module(name, branch)
        self.mean_input_layer = MeanInputLayer()
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
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
            for i in range(10)
        }
        for name, branch in self.output_branches.items():
            self.add_module(name, branch)

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        x = [
            self.input_branches[branch](inputs[branch])
            for branch in inputs.keys()
            if self.input_branches[branch].active
        ]
        x = self.mean_input_layer(x)
        x = self.hidden_layers(x)
        x = [
            self.output_branches[i](x)
            if self.output_branches[i].active
            else torch.zeros(hidden_out.shape[0], 1)
            for i in range(10)
        ]
        x = self.concat_layer(x)
        return x
```

This example neural net has two input branches, one processing grayscale images and the other color images, and 10 output branches for a classification task. The Branch class has activate, deactivate, freeze, and unfreeze functions, which can be used to control which parts of the net are active during training. Inactive branches are not used in the forward function.

Next, define the `prep_lesson` and `loss` methods for your neural net. The `prep_lesson` method specifies which parts of the net will be active or inactive during a given lesson, and the `loss` method defines the loss function for the net.

```
class MyModel(Changeling):
    # ... 
    
    def prep_lesson(self, name: str) -> None:
        if name == "Lesson 1":
            self.input_branches["gray_image"].activate()
            self.input_branches["color_image"].deactivate()
            for i in range(10):
                self.output_branches[i].deactivate()
            self.output_branches[0].activate()
        elif name == "Lesson 2":
            self.input_branches["gray_image"].activate()
            self.input_branches["color_image"].activate()
            for i in range(10):
                self.output_branches[i].deactivate()
            self.output_branches[0].activate()
            self.output_branches[1].activate()
        # ...

    def loss(self, outputs: Tensor, labels: Tensor) -> Tensor:
        return nn.CrossEntropyLoss(outputs, labels)
```

Once the neural net is defined, you can create a training `Curriculum` and use the `Teacher` class to train and evaluate the net. The `Teacher` expects a list of `Lesson` objects, each specifying the training data, lesson name, and lesson completion criteria. The `Teacher` will use the `lesson_complete` criteria to determine when to move on to the next lesson in the curriculum.

```
from torch.utils.data import DataLoader
from changeling.core.learning_criteria import AccuracyThresholdAchieved, NEpochsComplete


class MyDataset(Dataset):
    # define your dataset here


train_loader = DataLoader(MyDataset(train=True), batch_size=32, shuffle=True)
test_loader = DataLoader(MyDataset(train=False), batch_size=32, shuffle=False)


def get_dataloader_subset(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    output_labels_to_include: list[int],
) -> tuple[DataLoader, DataLoader]:
    ...


model = MyModel()
curriculum = [
    Lesson(
        name="Lesson 1",
        get_dataloaders=lambda: get_dataloader_subset(
            train_dataset,
            test_dataset,
            batch_size=32,
            output_labels_to_include=[0, 1],
        ),
        lesson_complete=NEpochsComplete(5),
    ),
    # ...    
    Lesson(
        name="Lesson N",
        get_dataloaders=lambda: get_dataloader_subset(
            train_dataset,
            test_dataset,
            batch_size=32,
            output_labels_to_include=list(range(10)),
        ),
        lesson_complete=AccuracyThresholdAchieved(0.95),
    ),
]


teacher = Teacher(model, curriculum)
teacher.teach()
```

## Contributing

Contributions to Changeling are welcome!

## License

TODO: fill this out
