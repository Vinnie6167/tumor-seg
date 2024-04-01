"""Implements NearestMean Model"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

# from evaluate import evaluate
from tumor_dataset import TumorDataset

class NearestMean(nn.Module):
    def __init__(self, means, labels):
        """
        Initializes NearestMean Model from means and corresponding labels.

        Args:
            means: List[float] - mean[i] = mean of all values with label of labels[i]
            labels: List[int] - corresponding labels (labels[i] cannot be -1)

        Returns:
            Initialized NearestMean Model
        """
        assert -1 not in labels

        super().__init__()

        zipped = list(zip(means, labels))
        zipped.sort()

        # Defines intervals for each class
        self.thresholds = [(zipped[i][0] + zipped[i + 1][0]) / 2 for i in range(len(zipped) - 1)]
        self.labels = [z[1] for z in zipped]

    def forward(self, X):
        """
        Makes predictions depending on intervals.

        Args:
            X: torch.Tensor - values to predict on

        Returns:
            torch.Tensor of equivalent shape to X with class prediction for each value in X
        """
        original_shape = X.shape
        X_flat = X.flatten()
        y = torch.full(X_flat.shape, -1)

        for i in range(len(self.labels) - 1):
            y[(y == -1) * (X_flat < self.thresholds[i])] = self.labels[i]
        y[y == -1] = self.labels[-1]

        return y.reshape(original_shape)


class TumorSimpleBaseline(nn.Module):
    def __init__(self, dataloader, classes):
        print('Initializing TumorSimpleBaseline')

        super().__init__()

        means = {label : 0 for label in classes}
        counts = {label : 0 for label in classes}
        print('Collecting means')
        for X, y in tqdm(dataloader):
            X = X[:, 0].flatten() # Only care about t1 image
            y = y.flatten()
            for label in classes:
                means[label] += torch.sum(X[y == label])
                counts[label] += torch.sum(y == label)
        for label in classes:
            means[label] /= counts[label]

        self.nearestMean = NearestMean(means, classes)

    def forward(self, X):
        X = X[:, 0] # Only care about t1 image

        return self.nearestMean(X)

def main():
    """Trains and evaluates threshold model"""

    print('Loading Data')

    dataset = TumorDataset('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
    train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=16, shuffle=True)

    classes = [0, 1, 2, 3]

    print('Training Model')

    model = TumorSimpleBaseline(train_dataloader, classes)

    print('Evaluating Model')

    # dice_score = 0
    # tumor_classes = list(filter(lambda x : x != 0, classes))
    a = 0
    b = 0
    for X, y in tqdm(train_dataloader):
        pred = model(X)
        pred = pred.flatten()
        y = y.flatten()
        a += ((pred == 0) * (y != 0)).sum()
        b += (y != 0).sum()
    print(a, b)
    print(f'False positive rate: {a/b}')

    # print(f'Threshold model received dice score of {dice_score}')


if __name__ == '__main__':
    main()