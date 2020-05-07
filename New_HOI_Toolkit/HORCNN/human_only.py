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
	
	for i in range(len(targets)):
		total_loss += loss_fn(preds, targets[0][i].unsqueeze(0).type(torch.long).cuda())
	print('Memalloc after loss: ' + str(torch.cuda.memory_allocated(0) / 1073741824) + "GB")
	print(total_loss.is_cuda)
	return total_loss / len(targets)

epochs = 30
learn_rate = .001

bbox_mat = loadmat('../Dataset/images/anno_bbox.mat')

human_model = models.alexnet(pretrained=True)
human_model.classifier[6] = nn.Linear(4096,600)
human_model = human_model.train()
human_model.cuda()

object_model = models.alexnet(pretrained=True)
object_model.classifier[6] = nn.Linear(4096,600)
object_model = object_model.train()
object_model.cuda()

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

pairwise_model = HO_RCNN_P()
pairwise_model = pairwise_model.train()
pairwise_model.cuda()

for param in object_model.parameters():
	param.requires_grad = True
for param in human_model.parameters():
	param.requires_grad = True
for param in pairwise_model.parameters():
	param.requires_grad = True

criterion = nn.BCEWithLogitsLoss()
all_params = list(human_model.parameters()) + list(object_model.parameters()) + list(pairwise_model.parameters())
optimizer = optim.SGD(all_params, lr = learn_rate)

#test_data = HICODET_test('../Dataset/images/test2015', bbox_mat, props_file='../Dataset/images/pkl_files/fullTest.pkl')
#test_data_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=1, shuffle=True)
train_data = HICODET_train('../Dataset/images/train2015', bbox_mat, props_file='../Dataset/images/pkl_files/fullTrain.pkl', props_list = '../Dataset/images/pkl_files/fullTrain_proposals.pkl')
n_total = len(train_data)
n_valid = int(n_total * 0.2)
n_train = int(n_total - n_valid) 

print('Length of training set: ' + str(n_train))
print('Length of validation set: ' + str(n_valid))

#print(n_total)
#print(n_valid)
#print(n_train)

train_set, valid_set = torch.utils.data.random_split(train_data, (n_train, n_valid))

train_data_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=8, shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(dataset = valid_set, batch_size=1, shuffle=False)

# For each epoch:
for epoch in range(1):
	# iterate through dataloader
	losses = []
	batch_count = 0
	img_count = 0


	#optimizer.zero_grad()

	for human_crop, object_crop, int_pattern, outputs in train_data_loader:
		batch_count += 1
		human_crop = human_crop.float().cuda()
		object_crop = object_crop.float().cuda()
		int_pattern = int_pattern.float().cuda()
		outputs = outputs.float().cuda()

		with torch.enable_grad():
			human_pred = human_model(human_crop)
			object_pred = object_model(object_crop)
			pairwise_pred = pairwise_model(int_pattern)
		
		total_pred = torch.add(torch.add(human_pred, object_pred), pairwise_pred)

		batch_loss = criterion(total_pred, outputs)
		
		if batch_count % 100 == 0:
			print('Batch# ' + str(batch_count))
			print(batch_loss.item())
			#if batch_count == 100:
				#print()
				#print(outputs)

		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()

model_dir = './saved_models'

torch.save(human_model.state_dict(), 'human.pth')
torch.save(object_model.state_dict(), 'object.pth')
torch.save(pairwise_model.state_dict(), 'pairwise.pth')

threshold = 0.7

with torch.no_grad():
	ap = 0.
	for h,o,p, labels in valid_data_loader:
		total_pred = torch.add(torch.add(human_model(h.float().cuda()), object_model(o.float().cuda())), pairwise_model(p.float().cuda()))
		total_pred = nn.Sigmoid(total_pred)
		total_pred = total_pred > threshold
		total_pred = total_pred.int()
		total_pred = total_pred.cup().detach().numpy()

		ap += metrics.precision_score(labels.numpy(),total_pred)
	print('AP on validation set:')
	print(ap/n_valid)

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
			loss = criterion(pred, output_labels[i].float().cuda())

			print('Updating...')
			img_count = 0
			total_loss = loss
			print('Loss for batch: ' + str(total_loss))
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()
			losses.clear()

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

				