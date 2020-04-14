import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys
import torch.optim as optim

# Dataset stuff
sys.path.append('../Dataset/Tools')
from dataset_load import HICO_DET_Dataloader, get_interaction_pattern

# Set anomaly tracking:
torch.autograd.set_detect_anomaly(True)

# Dataset loader class. Loads data from matfiles and creates lists of annotations
class HO_RCNN(nn.Module):

    def __init__(self):
        super(HO_RCNN, self).__init__()

        # Human Stream Layers:
        self.human_cnn_layers = nn.Sequential(
                # First convLayer:
                nn.Conv2d(3, 96, kernel_size=11, stride=4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(5),
                # Second convLayer:
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(5),
                # Third ConvLayer:
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(),
                # Fourth ConvLayer:
                nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
                nn.ReLU(),
                # Fifth ConvLayer:
                nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                )
        self.human_linear_layers = nn.Sequential(
                # FC-Layer 1:
                nn.Linear(6*6*256, 4096),
                #nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.Dropout(inplace=True),
                # FC-Layer 2:
                nn.Linear(4096, 4096),
                #nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.Dropout(inplace=True),
                # Final Class Score layer:
                nn.Linear(4096, 600)
                )

        # Object Stream Layers:
        self.object_cnn_layers = nn.Sequential(
                # First convLayer:
                nn.Conv2d(3, 96, kernel_size=11, stride=4),
                #nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(5),
                # Second convLayer:
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
                #nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(5),
                # Third ConvLayer:
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
               # nn.ReLU(inplace=True),
                nn.ReLU(),
                # Fourth ConvLayer:
                nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
                #nn.ReLU(inplace=True),
                nn.ReLU(),
                # Fifth ConvLayer:
                nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
                #nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                )
        self.object_linear_layers = nn.Sequential(
                # FC-Layer 1:
                nn.Linear(6*6*256, 4096),
                #nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.Dropout(inplace=True),
                # FC-Layer 2:
                nn.Linear(4096, 4096),
                #nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.Dropout(inplace=True),
                # Final Class Score layer:
                nn.Linear(4096, 600)
                )

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

        # Human Stream Pass:
        human_stream = self.human_cnn_layers(img_human)
        # Flatten for linear layers
        human_stream = torch.flatten(human_stream, 1)
        human_stream = self.human_linear_layers(human_stream)

        # Object Stream Pass:
        object_stream = self.object_cnn_layers(img_object)
        object_stream = torch.flatten(object_stream, 1)
        object_stream = self.object_linear_layers(object_stream)

        # Pairwise Stream Pass:
        pairwise_stream = self.pairwise_cnn_layers(img_pairwise)
        pairwise_stream = torch.flatten(pairwise_stream, 1)
        pairwise_stream = self.pairwise_linear_layers(pairwise_stream)

        # Elementwise summation for final class score
        class_score = human_stream.add(object_stream.add(pairwise_stream))
        return class_score

def train_model(model, num_epochs, learn_rate):
    print('=============== Training Model With Parameters =================')
    print('num_epochs = ' + str(num_epochs))
    print('learning rate = ' + str(learn_rate))
    print('================================================================')



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

print('Getting Outputs')
outs = data.__get_output__(0,'test')
outs = torch.from_numpy(outs)
outs = outs.unsqueeze(0)
outs = torch.tensor(outs, dtype=torch.long).cuda()
print('Got OUtput')

img_h = img_h.resize((256,256))
img_h = np.asarray(img_h).transpose(-1,0,1)
img_h = torch.from_numpy(img_h)
img_h = img_h.unsqueeze(0).cuda()

img_o = img_o.resize((256,256))
img_o = np.asarray(img_o).transpose(-1,0,1)
img_o = torch.from_numpy(img_o)
img_o = img_o.unsqueeze(0).cuda()

img_p = img_p.unsqueeze(0).cuda()

print('Loss and optim')
#loss, optimizer = loss_optim(model, 0.001)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


for param in model.parameters():
    param.requires_grad = True

print('Running Infrence')
with torch.enable_grad():
    res = model(img_h.float(), img_o.float(), img_p.float())
    #res = model(img_p.float())

print(res.shape)
print(res)
print(outs.shape)
print(outs)


with torch.enable_grad():
    print('Getting Loss')
    loss_size = criterion(res, torch.max(outs, 1)[1])
    print('Loss = ' + str(loss_size))
    print('Backwards')
    loss_size.backward()
    print('optimizer')
    optimizer.step()
