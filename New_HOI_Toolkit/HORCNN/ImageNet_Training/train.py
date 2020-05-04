import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
import torch.optim as optim

class HO_RCNN(nn.Module):

    def __init__(self):
        super(HO_RCNN, self).__init__()

        # Human Stream Layers:
        self.human_cnn_layers = nn.Sequential(
                # First convLayer:
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
                # Second convLayer:
                nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
                # Third ConvLayer:
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Fourth ConvLayer:
                nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
                nn.ReLU(inplace=True),
                # Fifth ConvLayer:
                nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                )
        self.human_linear_layers = nn.Sequential(
                # FC-Layer 1:
                nn.Linear(6*6*256, 4096),
                #nn.ReLU(inplace=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                # FC-Layer 2:
                nn.Linear(4096, 4096),
                #nn.ReLU(inplace=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                # Final Class Score layer:
                nn.Linear(4096, 600)
                )

    # Forward Pass Function:
    def forward(self, img_human):

        # Human Stream Pass:
        human_stream = self.human_cnn_layers(img_human)
        # Flatten for linear layers
        human_stream = torch.flatten(human_stream, 1)
        human_stream = self.human_linear_layers(human_stream)

        return human_stream


trainset = torchvision.datasets.ImageNet(root='./imgnet', train=True, download=True)
valset = torchvision.datasets.ImageNet(root='./imgnet', train=False, download=True)

train_loader = torch.utils.data.DataLoader(trainset, batch_sz=4, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(valset, batch_sz=1, shuffle=True, num_workers=1)






