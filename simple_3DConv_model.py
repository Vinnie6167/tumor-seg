import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from evaluate import evaluate

from tumor_dataset import TumorDataset

import matplotlib.pyplot as plt

class Simple3DConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, padding='same')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        Z = self.conv(X)
        y_hat = self.softmax(Z)
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
    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=1, shuffle=True)

    classes = [0, 1, 2, 3]

    print('Training Model')

    model = Simple3DConv()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 0

    losses = [compute_loss(train_dataloader, model, loss_fn)]
    for epoch in tqdm(range(1, num_epochs + 1)):
        train_one_epoch(train_dataloader, model, optimizer, loss_fn)

        if epoch % 5 == 0:
            losses.append(compute_loss(train_dataloader, model, loss_fn))

    plt.plot(range(0, num_epochs // 5 + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')

    print('Evaluating Model')

    dice_score = 0
    tumor_classes = list(filter(lambda x : x != 0, classes))
    for X, y in tqdm(val_dataloader):
        dice_score += evaluate(model, X, y, tumor_classes)
    dice_score /= len(val_dataloader)

    print(f'3DConv model received dice score of {dice_score}')

    torch.save(model, f'{args.name}.pt')

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