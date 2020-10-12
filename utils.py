import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, conv_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3,
            stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(conv_channels)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3,
            stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(conv_channels)

    def forward(self, x):

        identity = x
        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.in2(self.conv2(x)))
        return F.relu(x + identity)
