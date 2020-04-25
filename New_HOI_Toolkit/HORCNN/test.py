import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys
import torch.optim as optim
from scipy.io import loadmat
# Dataset stuff
sys.path.append('../Dataset')
from data_loader import HICODET_train, HICODET_test
from horcnn_model import HO_RCNN

def compute_loss(preds, targets, loss_fn):
	total_loss = torch.tensor(0.)

	for i in range(len(targets)):
		total_loss += loss_fn(preds, torch.tensor(targets[i]).cuda())

	return total_loss / len(targets)

epochs = 30
learn_rate = .001

bbox_mat = loadmat('../Dataset/images/anno_bbox.mat')

model = HO_RCNN()
model.cuda()

for param in model.parameters():
	param.requires_grad = True

criterion = nn.CrossEntropyLoss()
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

	for in_proposals, output_labels in train_data_loader:
		# Since batch size is set at 1 we need to loop through 8 images before updating
		
		for i in range(len(in_proposals)):
			print('Going through proposal #' + str(i))
			predictions = []
			num_props = 0
			# For each proposal in the image
			for hop_prop in in_proposals:

				human = hop_prop[0].cuda()
				obj = hop_prop[1].cuda()
				pair = hop_prop[2].cuda()

				with torch.enable_grad():
					predictions.append(model(human.float(), obj.float(), pair.float()))
				num_props += 1

		# Now average all the predictions in this image:
		avg_pred = torch.zeros([1,600], dtype=torch.float64).cuda()
		for p in predictions:
			avg_pred = torch.add(avg_pred, p)
		avg_pred = torch.div(avg_pred, num_props)

		# Compute loss over all the classes here.
		batch_loss += compute_loss(avg_pred, output_labels)

		img_count += 1

		if img_count == batch_sz:
			print('Now update')
			img_count = 0
			batch_loss.backward()
			optimizer.step()
		











