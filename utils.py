import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
        

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1),
            nn.InstanceNorm2d(filters),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1),
            nn.InstanceNorm2d(filters)
        )

    def forward(self, x):
        return F.relu(x + self.res_block(x))
