import torch
from torch.utils.data import DataLoader, TensorDataset

from changeling.core.learning_criteria import NEpochsComplete
from changeling.core.teacher import Lesson


def test_lesson():
    train_dataset = TensorDataset(
        torch.randn(10, 3, 32, 32),
        torch.randint(0, 10, (10,))
    )
    test_dataset = TensorDataset(
        torch.randn(5, 3, 32, 32),
        torch.randint(0, 10, (5,))
    )
    lesson_name = "Lesson 1"
    lesson_complete = NEpochsComplete(3)
    lesson = Lesson(lesson_name, lambda: (DataLoader(train_dataset), DataLoader(test_dataset)), lesson_complete)
    assert lesson.name == lesson_name
    train_loader, test_loader = lesson.get_dataloaders()
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert lesson.lesson_complete(0.5, 0.8) == False
    assert lesson.lesson_complete(0.5, 0.8) == False
    assert lesson.lesson_complete(0.5, 0.8) == True


def test_curriculum():
    train_dataset_1 = TensorDataset(
        torch.randn(3, 3, 32, 32),
        torch.randint(0, 10, (3,))
    )
    test_dataset_1 = TensorDataset(
        torch.randn(4, 3, 32, 32),
        torch.randint(0, 10, (4,))
    )
    lesson_1_name = "Lesson 1"
    lesson_1_complete = NEpochsComplete(2)
    lesson_1 = Lesson(lesson_1_name, lambda: (DataLoader(train_dataset_1), DataLoader(test_dataset_1)), lesson_1_complete)
    train_dataset_2 = TensorDataset(
        torch.randn(5, 3, 32, 32),
        torch.randint(0, 10, (5,))
    )
    test_dataset_2 = TensorDataset(
        torch.randn(2, 3, 32, 32),
        torch.randint(0, 10, (2,))
    )
    lesson_2_name = "Lesson 2"
    lesson_2_complete = NEpochsComplete(1)
    lesson_2 = Lesson(lesson_2_name, lambda: (DataLoader(train_dataset_2), DataLoader(test_dataset_2)), lesson_2_complete)

    curriculum = [lesson_1, lesson_2]
    assert len(curriculum) == 2
    assert curriculum[0].name == lesson_1_name
    assert curriculum[1].name == lesson_2_name
