import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys

sys.path.append('../Dataset/Tools')
from dataset_load import HICO_DET_Dataloader, get_interaction_pattern

class HO_RCNN(nn.Module):

    def __init__(self):
        super(HO_RCNN, self).__init__()

        self.human_cnn_layers = nn.Sequential(
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
        self.human_linear_layers = nn.Sequential(
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

        self.object_cnn_layers = nn.Sequential(
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
        self.object_linear_layers = nn.Sequential(
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

        self.pairwise_cnn_layers = nn.Sequential(
                # Conv Layer 1:
                nn.Conv2d(3, 64, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                # Conv layer 2:
                nn.Conv2d(64, 32, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
                )
        self.pairwise_linear_layers = nn.Sequential(
                # FC 1:
                nn.Linear(4096, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 600)
                )

    def forward(self, img_human, img_object, img_pairwise):

        human_stream = self.human_cnn_layers(img_human)
        print('Human ConvOut Shape: ' + str(human_stream.shape))
        human_stream = human_stream.view(human_stream.size(0), -1)
        print('Human Flatten Shape:' + str(human_stream.shape))
        human_stream = self.human_linear_layers(human_stream)
        print('Human Output Shape:' + str(human_stream.shape))

        object_stream = self.object_cnn_layers(img_object)
        print('Object ConvOut Shape: ' + str(object_stream.shape))
        object_stream = object_stream.view(object_stream.size(0), -1)
        print('Object Flatten Shape:' + str(object_stream.shape))
        object_stream = self.object_linear_layers(object_stream)
        print('Object Output Shape:' + str(object_stream.shape))

        pairwise_stream = self.pairwise_cnn_layers(img_pairwise)
        print('pairwise ConvOut Shape: ' + str(pairwise_stream.shape))
        pairwise_stream = pairwise_stream.view(pairwise_stream.size(0), -1)
        print('pairwise Flatten Shape:' + str(pairwise_stream.shape))
        pairwise_stream = self.pairwise_linear_layers(pairwise_stream)
        print('pairwise Output Shape:' + str(pairwise_stream.shape))


print('Initializing Model')
model = HO_RCNN()
print('Done Initializing Model')
print('Moving Model to GPU')
model.cuda()
print('Model on GPU')

data = HICO_DET_Dataloader('/Documents/hico/images/train2015', '../Dataset/Tools/test2015', '../Dataset/anno_bbox.mat')

print('Getting Human Image: ')
img_h, bbox_h = data.__get_human_crop__(0, 'test')
print('Done Getting Human Image: ')
print('Getting Object Image: ')
img_o, bbox_o = data.__get_object_crop__(0, 'test')
print('Done Getting Object Image: ')

print('Getting Interaction Image: ')
w, h = data.__get_img_dims__(0, 'test')
img_p = get_interaction_pattern(w, h, bbox_h, bbox_o)
print('Done Getting Interaction Image: ')

img_h = img_h.resize((256,256))
img_h = np.asarray(img_h).transpose(-1,0,1)
img_h = torch.from_numpy(img_h)
img_h = img_h.unsqueeze(0).cuda()

img_o = img_o.resize((256,256))
img_o = np.asarray(img_o).transpose(-1,0,1)
img_o = torch.from_numpy(img_o)
img_o = img_o.unsqueeze(0).cuda()

img_p.unsqueeze(0).cuda()

print('Running Infrence')
with torch.no_grad():
    res = model(img_h.float(), img_o.float(), img_p.float())

print(res.shape)
print(res)
