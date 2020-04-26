import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys
import torch.optim as optim

# Dataset stuff
sys.path.append('../Dataset')
from data_loader import HICODET_train, HICODET_test

# Set anomaly tracking:
torch.autograd.set_detect_anomaly(True)

class HO_RCNN(nn.Module):

    def __init__(self):
        super(HO_RCNN, self).__init__()

        # Object Stream Layers:
        self.object_cnn_layers = nn.Sequential(
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
        self.object_linear_layers = nn.Sequential(
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
    def forward(self, img_human, img_object, img_pairwise):

        # Object Stream Pass:
        object_stream = self.object_cnn_layers(img_object)
        object_stream = torch.flatten(object_stream, 1)
        object_stream = self.object_linear_layers(object_stream)

        return object_stream