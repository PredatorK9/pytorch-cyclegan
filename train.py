import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from models import PatchGAN, Generator


def build_models(num_res_block, )