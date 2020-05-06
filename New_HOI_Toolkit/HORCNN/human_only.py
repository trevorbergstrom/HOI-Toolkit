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

# Dataset stuff
sys.path.append('../Dataset')
from data_loader import HICODET_train, HICODET_test

# Set anomaly tracking:
torch.autograd.set_detect_anomaly(True)



def compute_loss(preds, targets, loss_fn):
	total_loss = torch.tensor(0.).cuda()
	
	for i in range(len(targets)):
		total_loss += loss_fn(preds, targets[0][i].unsqueeze(0).type(torch.long).cuda())
	print('Memalloc after loss: ' + str(torch.cuda.memory_allocated(0) / 1073741824) + "GB")
	print(total_loss.is_cuda)
	return total_loss / len(targets)

epochs = 30
learn_rate = .001

bbox_mat = loadmat('../Dataset/images/anno_bbox.mat')

model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096,600)
model = model.train()
model.cuda()

for param in model.parameters():
	param.requires_grad = True

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr = learn_rate)

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
	optimizer.zero_grad()


	for in_proposals, output_labels in train_data_loader:
		# Since batch size is set at 1 we need to loop through 8 images before updating
		cur_img_list.append(cur_i)
		cur_i += 1

		predictions = []
		num_props = 0
		
		for i in range(len(in_proposals)):
			print('Going through proposal #' + str(i))
			
			#human = torch.from_numpy(in_proposals[i][0]).cuda()
			human = in_proposals[i][0].cuda()
			
			with torch.enable_grad():
				pred = model(human.float())
			#print(output_labels)
			#losses.append(criterion(pred, torch.from_numpy(output_labels[i]) ))
			loss = criterion(pred, output_labels[i].float().cuda())

			print('Updating...')
			img_count = 0
			total_loss = loss
			print('Loss for batch: ' + str(total_loss))
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()
			losses.clear()
	'''
	for in_proposals, output_labels in train_data_loader:
		# Since batch size is set at 1 we need to loop through 8 images before updating
		cur_img_list.append(cur_i)
		cur_i += 1

		predictions = []
		num_props = 0
		
		for i in range(len(in_proposals)):
			print('Going through proposal #' + str(i))
			
			#human = torch.from_numpy(in_proposals[i][0]).cuda()
			human = in_proposals[i][0].cuda()
			
			with torch.enable_grad():
				pred = model(human.float())
			#print(output_labels)
			#losses.append(criterion(pred, torch.from_numpy(output_labels[i]) ))
			losses.append(criterion(pred, output_labels[i].float().cuda() ))

		img_count += 1

		if img_count == batch_sz:
			print('Updating...')
			img_count = 0
			total_loss = sum(losses)
			print('Loss for batch: ' + str(total_loss))
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()
			losses.clear()
	'''	

				