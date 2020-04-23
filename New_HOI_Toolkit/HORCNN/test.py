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
train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=8, shuffle=True)

# For each epoch:
for epoch in range(1):
	# iterate through dataloader
	for batch, output_labels in train_data_loader:
		# Create a list to save individual pair predictions
		# For each set of pairs (hop)
		print('here I am')
		losses = []
		for i in range(len(batch)):
			print(len(batch))
			predictions = []
			num_props = 0
			for hop_prop in batch[i]:
				if hop_prop[0].shape != torch.Size([1]):
					human = hop_prop[0].unsqueeze(0).cuda()
					obj = hop_prop[1].unsqueeze(0).cuda()
					pair = hop_prop[2].unsqueeze(0).cuda()
					print(human.shape)
					print(obj.shape)
					print(pair.shape)
					print(len(batch[i]))
					print(len(hop_prop))
					print(hop_prop.shape)
					print(len(hop_prop[0]))

					with torch.enable_grad():
						predictions.append(model(human.float(), obj.float(), pair.float()))
					num_props += 1

			# Now average all the predictions in this image:
			avg_pred = torch.zeros([1,600], dtype=torch.float64).cuda()
			for i in predictions:
				avg_pred = torch.add(avg_pred, predictions[i])
			avg_pred = torch.div(avg_pred, num_props)

			losses.append(criterion(avg_pred, outpus[i].cuda()))

		avg_loss = sum(losses) / len(losses)
		avg_loss.backward()
		optimizer.step()











