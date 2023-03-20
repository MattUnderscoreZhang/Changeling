from dataclasses import dataclass
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Callable, List, Tuple


@dataclass
class Lesson:
    name: str
    get_dataloaders: Callable[[], Tuple[DataLoader, DataLoader]]


Curriculum = List[Lesson]


class Teacher:
    def __init__(
        self,
        model: nn.Module,
        curriculum: Curriculum,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.curriculum = curriculum
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.accuracy_threshold = 0.95
        self.consecutive_epochs_threshold = 3

    def train(self, train_loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
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
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def prep_lesson(self, lesson_n: int) -> Tuple[DataLoader, DataLoader]:
        lesson = self.curriculum[lesson_n]
        print(f"Prepping lesson - {lesson.name}.")
        train_loader, test_loader = lesson.get_dataloaders()
        return train_loader, test_loader

    def teach(self, max_epochs: int = -1) -> bool:
        lesson_n = 0
        train_loader, test_loader = self.prep_lesson(lesson_n)

        n_consecutive_epochs = 0
        if max_epochs == -1:
            max_epochs = int('inf')
        for epoch in range(max_epochs):
            print(f"Epoch {epoch} - ", end="")
            train_loss = self.train(train_loader)
            test_accuracy = self.test(test_loader)
            print(f"Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            n_consecutive_epochs = (
                n_consecutive_epochs + 1
                if test_accuracy >= self.accuracy_threshold
                else 0
            )
            if n_consecutive_epochs >= self.consecutive_epochs_threshold:
                if lesson_n + 1 >= len(self.curriculum):
                    print("Training complete")
                    return True
                lesson_n += 1
                train_loader, test_loader = self.prep_lesson(lesson_n)

        print("Reached max epochs without finishing training.")
        return False
