from changeling.core.learning_criteria import NEpochsComplete, AccuracyThresholdAchieved


def test_n_epochs_complete():
    learning_criteria = NEpochsComplete(3)
    assert learning_criteria(0.5, 0.8) == False
    assert learning_criteria(0.5, 0.8) == False
    assert learning_criteria(0.5, 0.8) == True
    assert learning_criteria(0.5, 0.8) == False


def test_accuracy_threshold_achieved():
    learning_criteria = AccuracyThresholdAchieved(0.9)
    assert learning_criteria(0.5, 0.8) == False
    assert learning_criteria(0.5, 0.85) == False
    assert learning_criteria(0.5, 0.9) == False
    assert learning_criteria(0.5, 0.9) == False
    assert learning_criteria(0.5, 0.9) == True
    assert learning_criteria(0.5, 0.9) == False
    assert learning_criteria(0.5, 0.9) == False
    assert learning_criteria(0.5, 0.9) == True
