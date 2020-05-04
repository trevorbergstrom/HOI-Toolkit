import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import torchvision
from torchvision import models
from PIL import Image
import numpy as np
import sys
import torch.optim as optim
from scipy.io import loadmat

# Dataset stuff
sys.path.append('../Dataset')
from data_loader import HICODET_train, HICODET_test

# Set anomaly tracking:
torch.autograd.set_detect_anomaly(True)

# These two are just caffe net, only have alexnet but they are the same
human_stream = models.alexnet(pretrained=True)
object_stream = models.alexnet(pretrained=True)

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
    def forward(self, img_human, img_object, img_pairwise):

        # Pairwise Stream Pass:
        pairwise_stream = self.pairwise_cnn_layers(img_pairwise)
        pairwise_stream = torch.flatten(pairwise_stream, 1)
        pairwise_stream = self.pairwise_linear_layers(pairwise_stream)

        return pairwise_stream

pairwise_stream = HO_RCNN_P()

human_stream.cuda()
object_stream.cuda()
pairwise_stream.cuda()

params = list(human_stream.parameters()) + list(object_stream.parameters()) + list(pairwise_stream.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params, lr = 0.0001)

test_data = HICODET_test('../Dataset/images/test2015', bbox_mat, props_file='../Dataset/images/pkl_files/fullTest.pkl')
test_data_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=1, shuffle=True)
train_data = HICODET_train('../Dataset/images/train2015', bbox_mat, props_file='../Dataset/images/pkl_files/fullTrain.pkl')
train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=1, shuffle=False)

# For each epoch:
for epoch in range(1):
	# iterate through dataloader
	losses = []
	batch_sz = 8
	img_count = 0
	batch_loss = 0

	cur_img_list = []
	cur_i = 1
	for in_proposals, output_labels in train_data_loader:
		# Since batch size is set at 1 we need to loop through 8 images before updating
		cur_img_list.append(cur_i)
		cur_i += 1

		predictions = []
		num_props = 0
		
		for i in range(len(in_proposals)):
			print('Going through proposal #' + str(i))
			
			# For each proposal in the image
			for hop_prop in in_proposals:

				human = hop_prop[0].cuda()
				obj  = hop_prop[1].cuda()
				pair = hop_prop[2].cuda()

				with torch.enable_grad():
					predictions.append( torch.div(torch.add(human_stream(human.float()), torch.add(obj.float(), pair.float())), 8.) )
				num_props += 1
				del human

		print('Memalloc after forward: ' + str(torch.cuda.memory_allocated(0) / 1073741824) + "GB")
		# Now average all the predictions in this image:
		avg_pred = torch.zeros([1,600], dtype=torch.float64).cuda()
		for p in predictions:
			avg_pred = torch.add(avg_pred, p)
		avg_pred = torch.div(avg_pred, num_props)
		print('Memalloc after avg: ' + str(torch.cuda.memory_allocated(0) / 1073741824) + "GB")
		# Compute loss over all the classes here.
		batch_loss += compute_loss(avg_pred, output_labels, criterion)
		del avg_pred
		del predictions
		del output_labels
		del in_proposals

		img_count += 1

		if img_count == batch_sz:
			print('Now update')
			img_count = 0
			try:
				with autograd.detect_anomaly():
					batch_loss.backward(retain_graph=True)
				optimizer.step()
				del batch_loss
				batch_loss = 0
			except RuntimeError:
				print('Error occured in batch:')
				print(cur_img_list)
				continue
				