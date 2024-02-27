"""Contains evaluation implementation"""

import torch


def diceCoef(y_hat, y_true):
    """
    Args:
        y_hat: Predicted
        y_true: True labels
    """
    return 2 * (y_hat * y_true).sum() / (y_hat.sum() + y_true.sum())

def averageDiceCoef(y_hat, y_true, classes):
    dice = 0
    for label in classes:
        dice += diceCoef(y_hat == label, y_true == label)
    return dice/len(classes)

def evaluate(model, X, y, classes):
    output = model(X)
    pred = torch.argmax(output, dim=1)
    return averageDiceCoef(pred, y, classes)