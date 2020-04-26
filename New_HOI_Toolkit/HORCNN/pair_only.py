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

        # Pairwise Stream Layers:
        self.pairwise_cnn_layers = nn.Sequential(
                # Conv Layer 1:
                nn.Conv2d(2, 64, kernel_size=5),
                #nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                # Conv layer 2:
                nn.Conv2d(64, 32, kernel_size=5),
                #nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
                )
        self.pairwise_linear_layers = nn.Sequential(
                # FC 1:
                nn.Linear(32*60*60, 256),
                #nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.Linear(256, 600)
                )

    # Forward Pass Function:
    def forward(self, img_human, img_object, img_pairwise):

        # Pairwise Stream Pass:
        pairwise_stream = self.pairwise_cnn_layers(img_pairwise)
        pairwise_stream = torch.flatten(pairwise_stream, 1)
        pairwise_stream = self.pairwise_linear_layers(pairwise_stream)

        return pairwise_stream