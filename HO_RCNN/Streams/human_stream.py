import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

class Human_net(nn.Module):

    def __init__(self):
        super(Human_net, self).__init__()

        self.cnn_layers = nn.Sequential(
                # First convLayer:
                nn.Conv2d(3, 96, kernel_size=11, stride=4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(5),
                # Second convLayer:
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(5),
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
        self.linear_layers = nn.Sequential(
                # FC-Layer 1:
                nn.Linear(256, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(inplace=True),
                # FC-Layer 2:
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(inplace=True),
                # Final Class Score layer:
                nn.Linear(4096, 600)
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model = Human_net()

img = Image.open('plane.jpg')
img = img.resize((256,256))
#img = img.convert('RGB')
img = np.asarray(img).transpose(-1,0,1)
img = torch.from_numpy(np.asarray(img))


print(img.shape)
imgs = []
imgs.append(img)

r_img = torch.randn(1, 3, 256, 256)
print(r_img.shape)

with torch.no_grad():
    res = model(r_img)

print(res.shape)
