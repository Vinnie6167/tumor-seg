import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from evaluation.evaluate import construct_confusion, diceForEach

from tumor_dataset import TumorDataset

import matplotlib.pyplot as plt

import os

class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=4, out_features=8)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=8, out_features=4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        b, imgs, h, w, d = X.shape
        X = X.reshape((-1, imgs))

        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        y_hat = self.softmax(X).reshape((b, imgs, h, w, d))

        return y_hat


def train_one_epoch(dataloader, model, optimizer, loss_fn):
    for X, y in dataloader:
        optimizer.zero_grad()

        output = model(X)

        loss = loss_fn(output, y)
        loss.backward()

        optimizer.step()


def compute_loss(dataloader, model, loss_fn):
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            output = model(X)
            total_loss += loss_fn(output, y).item()
    return total_loss/len(dataloader)

def main(args):
    """Trains and evaluates simpled conv3d model"""

    print('Loading Data')

    dataset = TumorDataset('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
    train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=8, shuffle=True)

    classes = [0, 1, 2, 3]

    print('Training Model')

    if os.path.isfile(f'{args.name}.pt'):
        print(f'Loading {args.name}.pt')
        model = torch.load(f'{args.name}.pt')
    else:
        model = SimpleLinear()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 0

    # losses = [compute_loss(train_dataloader, model, loss_fn)]
    for epoch in tqdm(range(1, num_epochs + 1)):
        train_one_epoch(train_dataloader, model, optimizer, loss_fn)
        torch.save(model, f'{args.name}_{epoch}.pt')

        # if epoch % 5 == 0:
        #     losses.append(compute_loss(train_dataloader, model, loss_fn))

    # plt.plot(range(0, num_epochs // 5 + 1), losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.savefig('loss.png')

    print('Evaluating Model')

    tumor_classes = list(filter(lambda x : x != 0, classes))
    dice_scores = {label : 0 for label in tumor_classes}
    for X, y in tqdm(val_dataloader):
        output = model(X)
        pred = torch.argmax(output, dim=1)

        batch_scores = diceForEach(pred, y, tumor_classes)
        for label in tumor_classes:
            dice_scores[label] += batch_scores[label]
    for label in tumor_classes:
        dice_scores[label] /= len(val_dataloader)

    # TODO: Confusion Matrix incorrect
    # cm = construct_confusion(pred, y, classes)

    print(f'Model received dice scores of {dice_scores}.')
    # print('Confusion Matrix:')
    # print(cm)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--name',
        default='model',
        type=str,
        help='Name of the model save file',
    )

    args = parser.parse_args()

    main(args)