import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys

sys.path.append('../../Dataset/Tools')
from dataset_load import HICO_DET_Dataloader

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
                nn.Linear(9216, 1),
                nn.ReLU(inplace=True),
                nn.Dropout(inplace=True),
                # FC-Layer 2:
                nn.Linear(1, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(inplace=True),
                # Final Class Score layer:
                nn.Linear(4096, 600)
                )
        '''
        self.cnn_layers = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)#,
                #nn.ReLU(inplace=True),
                #nn.MaxPool2d(kernel_size=3)
                )
        '''

    def forward(self, x):
        x = self.cnn_layers(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print('FLATTEN RES:')
        print(x.shape)
        x = self.linear_layers(x)
        return x

model = Human_net()
model.cuda()
data = HICO_DET_Dataloader('/Documents/hico/images/train2015', '../../Dataset/Tools/test2015', '../../Dataset/anno_bbox.mat')
img = data.__get_human_crop__(1, 'test')
img = img.resize((256,256))
#img = img.convert('RGB')
img = np.asarray(img).transpose(-1,0,1)
img = torch.from_numpy(img)
img = img.unsqueeze(0).cuda()
#img2 = torch.randn(3,1,11,11)

with torch.no_grad():
    res = model(img.float())

print(res.shape)
print(res)
