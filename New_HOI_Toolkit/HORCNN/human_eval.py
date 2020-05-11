import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torchvision import models
from PIL import Image
import numpy as np
import sys
import torch.optim as optim
from scipy.io import loadmat
import random
from sklearn import metrics

# Dataset stuff
sys.path.append('../Dataset')
from data_loader import HICODET_train, HICODET_test

# Set anomaly tracking:
torch.autograd.set_detect_anomaly(True)



def compute_loss(preds, targets, loss_fn):
	total_loss = torch.tensor(0.).cuda()
	#loss_list = torch.from_numpy(targets)
	batch_loss = []
	for i in range(len(targets)):
		prop = targets[i]
		print(prop)
		prop_loss = []
		for j in range(len(prop)):
			label = np.zeros(600)
			idx = prop[j] - 1
			print(idx)
			label[idx] = 1.0
			prop_loss.append(loss_fn(preds[i], torch.from_numpy(label).cuda()))
		batch_loss.append(sum(prop_loss))
	return(torch.tensor(sum(batch_loss)).cuda())

epochs = 30
learn_rate = .001

bbox_mat = loadmat('../Dataset/images/anno_bbox.mat')

class HO_RCNN_P(nn.Module):

    def __init__(self):
        super(HO_RCNN_P, self).__init__()

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
    def forward(self,img_pairwise):

        # Pairwise Stream Pass:
        pairwise_stream = self.pairwise_cnn_layers(img_pairwise)
        pairwise_stream = torch.flatten(pairwise_stream, 1)
        pairwise_stream = self.pairwise_linear_layers(pairwise_stream)

        return pairwise_stream


#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.CrossEntropyLoss()
#all_params = list(human_model.parameters()) + list(object_model.parameters()) + list(pairwise_model.parameters())
#optimizer = optim.SGD(all_params, lr = learn_rate)

#test_data = HICODET_test('../Dataset/images/test2015', bbox_mat, props_file='../Dataset/images/pkl_files/fullTest.pkl')
#test_data_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=1, shuffle=True)
train_data = HICODET_train('../Dataset/images/train2015', bbox_mat, props_file='../Dataset/images/pkl_files/fullTrain.pkl', props_list = '../Dataset/images/pkl_files/fullTrain_proposals.pkl')
#train_data = HICODET_train('../Dataset/images/train2015', bbox_mat, props_file='../Dataset/images/pkl_files/fullTrain.pkl')
n_total = len(train_data)
n_valid = int(n_total * 0.2)
n_train = int(n_total - n_valid) 

print('Length of training set: ' + str(n_train))
print('Length of validation set: ' + str(n_valid))

#print(n_total)
#print(n_valid)
#print(n_train)

train_set, valid_set = torch.utils.data.random_split(train_data, (n_train, n_valid))

train_data_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=1, shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(dataset = valid_set, batch_size=1, shuffle=False)

human_model = torch.load('./saved_models/human.pth')
object_model = torch.load('./saved_models/object.pth')
pairwise_model = torch.load('./saved_models/pairwise.pth')

num_zeros = 0
num_goods = 0
itr = 0

for h,o,p, outs in train_data_loader:
	
	if np.sum(outs) == 0.:
		num_zeros += 1
	else:
		num_goods += 1
	itr += 1
	if itr % 1000 == 0:
		print(num_zeros)
		print(num_goods)
		print('---------')

print('Num Zeros in trainset: ' + str(num_zeros))
print('Num good samples in trainset: ' + str(num_goods))

'''
threshold = 0.7

with torch.no_grad():
	ap = 0.
	for h,o,p, labels in valid_data_loader:
		total_pred = torch.add(torch.add(human_model(h.float().cuda()), object_model(o.float().cuda())), pairwise_model(p.float().cuda()))
		total_pred = nn.Sigmoid(total_pred)
		total_pred = total_pred > threshold
		total_pred = total_pred.int()
		total_pred = total_pred.cpu().detach().numpy()

		#ap += metrics.precision_score(labels.numpy(),total_pred)
	print('AP on validation set:')
	print(ap/n_valid)
'''				
