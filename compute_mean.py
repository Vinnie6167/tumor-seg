import torch
from tumor_dataset import TumorDataset
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset = TumorDataset('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=8, shuffle=True)

pixel_mean = 0
count = 0
for X, y in tqdm(train_dataloader):
    pixel_mean = pixel_mean * (count / (count + X.shape[0])) + torch.mean(X) * (X.shape[0] / (count + X.shape[0]))
    count += X.shape[0]

pixel_var = 0
count = 0
for X, y in tqdm(train_dataloader):
    pixel_var = pixel_var * (count / (count + X.shape[0])) + torch.var(X) * (X.shape[0] / (count + X.shape[0]))
    count += X.shape[0]

print(pixel_mean, pixel_var) # 100.9358, 233677.6719