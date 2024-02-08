import torch.nn as nn

class SimpleConvolution(nn.Module):
    def __init__(self, kernel_size=3):
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              stride=1, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.conv(x)
        y_hat = self.sigmoid(z)
        return y_hat