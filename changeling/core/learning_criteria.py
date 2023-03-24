class NEpochsComplete:
    def __init__(self, n_epochs_threshold: int) -> None:
        self.n_epochs = 0
        self.n_epochs_threshold = n_epochs_threshold

    def __call__(self, train_loss: float, test_acc: float) -> bool:
        self.n_epochs += 1
        if self.n_epochs >= self.n_epochs_threshold:
            self.n_epochs = 0
            return True
        return False


class AccuracyThresholdAchieved:
    def __init__(self, acc_threshold: float) -> None:
        self.n_consecutive_epochs = 0
        self.consecutive_epochs_threshold = 3
        self.acc_threshold = acc_threshold

    def __call__(self, train_loss: float, test_acc: float) -> bool:
        self.n_consecutive_epochs = (
            self.n_consecutive_epochs + 1
            if test_acc >= self.acc_threshold
            else 0
        )
        if self.n_consecutive_epochs >= self.consecutive_epochs_threshold:
            self.n_consecutive_epochs = 0
            return True
        return False
