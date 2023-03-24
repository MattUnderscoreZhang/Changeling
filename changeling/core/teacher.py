from dataclasses import dataclass
from termcolor import colored
import torch
from torch import optim
from torch.utils.data import DataLoader
from typing import Callable

from changeling.core.changeling import Changeling


@dataclass
class Lesson:
    name: str
    get_dataloaders: Callable[[], tuple[DataLoader, DataLoader]]
    accuracy_threshold: float


Curriculum = list[Lesson]


class Teacher:
    def __init__(
        self,
        model: Changeling,
        curriculum: Curriculum,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.curriculum = curriculum
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.consecutive_epochs_threshold = 3
        self.debug_print = True

    def train(self, train_loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = (
                {k: v.to(self.device) for k, v in inputs.items()}
                if type(inputs) == dict
                else inputs.to(self.device)
            )
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(train_loader)

    def test(self, test_loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = (
                    {k: v.to(self.device) for k, v in inputs.items()}
                    if type(inputs) == dict
                    else inputs.to(self.device)
                )
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                if self.debug_print:
                    self.print_predictions(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def print_predictions(self, outputs, labels) -> None:
        label = labels[0]
        output_list = outputs[0].tolist()
        prediction_correct = label == output_list.index(max(output_list))
        output_string = (
            colored('✓', 'green')
            if prediction_correct
            else colored('✗', 'red')
        )
        output_string += ' ['  # ]
        for i, output in enumerate(output_list):
            output_string += (
                colored("{:.3f}".format(output), "cyan")
                if i == label
                else colored("{:.3f}".format(output), "blue")
            )
            if i < len(output_list) - 1:
                output_string += ', '
        output_string += ']'
        print(output_string)

    def refresh_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def prep_lesson(self, lesson_n: int) -> tuple[DataLoader, DataLoader, float]:
        lesson = self.curriculum[lesson_n]
        print(colored(f"Prepping lesson - {lesson.name}.", "yellow"))
        train_loader, test_loader = lesson.get_dataloaders()
        self.model.prep_lesson(lesson.name)
        accuracy_threshold = lesson.accuracy_threshold
        self.refresh_optimizer()
        return train_loader, test_loader, accuracy_threshold

    def teach(self, max_epochs: int = -1) -> bool:
        lesson_n = 0
        train_loader, test_loader, accuracy_threshold = self.prep_lesson(lesson_n)

        n_consecutive_epochs = 0
        if max_epochs == -1:
            max_epochs = int('inf')
        for epoch in range(max_epochs):
            if self.debug_print:
                print(f"Epoch {epoch}:")
            else:
                print(f"Epoch {epoch} - ", end="")
            train_loss = self.train(train_loader)
            test_accuracy = self.test(test_loader)
            print(f"Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            n_consecutive_epochs = (
                n_consecutive_epochs + 1
                if test_accuracy >= accuracy_threshold
                else 0
            )
            if n_consecutive_epochs >= self.consecutive_epochs_threshold:
                if lesson_n + 1 >= len(self.curriculum):
                    print("Training complete")
                    return True
                lesson_n += 1
                train_loader, test_loader, accuracy_threshold = self.prep_lesson(lesson_n)

        print("Reached max epochs without finishing training.")
        return False
