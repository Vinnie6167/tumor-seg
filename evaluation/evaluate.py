"""Contains evaluation implementation"""

import torch


def diceCoef(y_hat, y_true):
    """
    Computes the dice coefficient for 1 class.

    Args:
        y_hat: ndarray/tensor - predicted (filled with 1's or 0's)
        y_true: ndarray/tensor - true labels (filled with 1's or 0's)

    Returns: dice coefficient
    """
    return 2 * (y_hat * y_true).sum() / (y_hat.sum() + y_true.sum())


def averageDiceCoef(y_hat, y_true, classes):
    """
    Computes average dice coefficient over a set of classes.

    Args:
        y_hat: ndarray/tensor - predicted
        y_true: ndarray/tensor - true labels
        classes: iterable - potential class labels

    Returns: average dice coefficient

    # TODO: Should we adjust how we are weighting the classes?
    """
    dice = 0
    for label in classes:
        dice += diceCoef(y_hat == label, y_true == label)
    return dice/len(classes)


def diceForEach(y_hat, y_true, classes):
    """
    Computes the dice coefficient for each class.

    Args:
        y_hat: ndarray/tensor - predicted
        y_true: ndarray/tensor - true labels
        classes: iterable - potential class labels

    Returns: dict[int, float] - dice coefficient for each class

    """
    diceForEach = {}
    for label in classes:
        diceForEach[label] = diceCoef(y_hat == label, y_true == label)
    return diceForEach


def construct_confusion(y_hat, y_true, classes):
    """
    Computes the confusion matrix.

    Args:
        pred: tensor/ndarray of predicted classes
        y: tensor/ndarray of true classes
        classes: iterable of potential classes

    Returns: tensor - confusion matrix
    """

    cm = torch.zeros((len(classes), len(classes)))

    y_hat = y_hat.flatten()
    y_true = y_true.flatten()

    for i in range(y_hat.shape[0]):
        cm[y_hat[i], y_true[i]] += 1

    return cm
