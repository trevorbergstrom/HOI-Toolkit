from data_loader import HICODET_test, HICODET_train
import convert_hico_mat as tools
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

def compute_loss2(preds, correct_hoi, loss_fn):
	total_loss = torch.tensor(0.).cuda()
	print(correct_hoi)
	for i in correct_hoi:
		gt = np.zeros(600)
		gt[i-1] = 1.
		total_loss = total_loss + loss_fn(preds, torch.from_numpy(gt).unsqueeze(0).cuda())
		del gt
	return total_loss

def compute_loss(preds, targets, loss_fn):
	total_loss = torch.tensor(0.).cuda()
	
	for i in range(len(targets)):
		tartar = torch.from_numpy(targets).type(torch.double).unsqueeze(0).cuda()
		#tartar = torch.max(tartar, 1)[1]
		#print(preds.shape)
		#print(tartar.shape)
		total_loss += loss_fn(preds, tartar)
	return torch.div(total_loss, float(len(targets)))

def pickle_proposals(props, file_name):
	with open(file_name, 'wb') as f:
		pickle.dump(props, f)

bbox_mat = tools.load_mat('images/anno_bbox.mat')
test_data = HICODET_test('images/test2015', bbox_mat, props_file='images/pkl_files/fullTest.pkl')

# Set anomaly tracking:
torch.autograd.set_detect_anomaly(True)

model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096,600)

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9)

model.cuda()

for epoch in range(1):

	img_count = 0
	batch_loss = torch.tensor(0.).cuda()

	for i in range(38000):
		print('Img# ' + str(i))
		proposals, labels = test_data[i]

		preds = []
		num_props = 0

		for j in proposals:
			human = j[0].cuda()
			with torch.enable_grad():
				preds.append(model(human.float().unsqueeze(0)))
			num_props += 1
			del j

		print("Averaging")
		avg_pred = torch.zeros([1,600], dtype=torch.float64).cuda()

		for p in preds:
			avg_pred = torch.add(avg_pred, p)
		avg_pred = torch.div(avg_pred, num_props)
		
		print("Compute loss")
		batch_loss = batch_loss + compute_loss2(avg_pred, labels, criterion)

		del avg_pred
		del preds
		del labels
		del proposals

		img_count += 1 
		
		if img_count == 8:
			print("batch_loss:")
			print(batch_loss)

			print("\nbackwards pass")
			img_count = 0
			batch_loss.backward(retain_graph = True)
			print("weight update")
			optimizer.step()
			batch_loss = torch.tensor(0.).cuda()



