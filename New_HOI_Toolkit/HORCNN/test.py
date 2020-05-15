import argparse
import os
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
from sklearn.metrics import average_precision_score
import pathlib
from collections import namedtuple

# Dataset stuff
sys.path.append('../Dataset')
from data_loader import HICODET_train, HICODET_test
from pair_only import HO_RCNN_Pair

def main():
	parser = argparse.ArgumentParser(description="Training the HORCNN Model!")
	parser.add_argument("bbmatfile", help="Path to the HICO-DET bounding box matfile", default='../Dataset/images/anno_bbox.mat', nargs='?')
	#parser.add_argument("train_path", help="Path to the file containing training images", default='../Dataset/images/train2015', nargs='?')
	parser.add_argument("test_path", help="Path to the file containing Testing images", default='../Dataset/images/test2015', nargs='?')
	#parser.add_argument("det_prop_file", help="Path of the object detection proposals pickle file. This file should contian detected objects from the images. If none is specified this program will run the FastRCNN Object detector over the training images to create the file", 
	#	default='../Dataset/images/pkl_files/fullTrain.pkl', nargs='?')
	parser.add_argument("det_prop_file", help="Path of the object detection proposals pickle file. This file should contian detected objects from the images. If none is specified this program will run the FastRCNN Object detector over the training images to create the file", 
		default='../Dataset/images/pkl_files/full_test2015.pkl', nargs='?')
	parser.add_argument("batch_size", help="batch_size for training", type=int, default=4, nargs='?')
	parser.add_argument("gpu", help="Runninng on GPU?", type=bool, default=True, nargs='?')
	parser.add_argument("model_path", help="Model Path To Evaluate", default='./saved_models/Final_Trained', nargs='?')
	parser.add_argument("threshold", help="prediction threshold", default=0.001, nargs='?', type=float)
	args = parser.parse_args()

	bbox_mat = loadmat('../Dataset/images/anno_bbox.mat')

	test_data = HICODET_test(args.test_path, args.bbmatfile, props_file=args.det_prop_file, proposal_count=1)
	print('Length of Testing dataset: ' + str(len(test_data)))
	print('Num props ' + str(test_data.proposal_count))
	test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

	print('------------DATASET SETUP COMPLETE-------------')
	print('\n')
	print('------------Setting Up Models------------------')
	human_model = models.alexnet(pretrained=True)
	human_model.classifier[6] = nn.Linear(4096,600)
	human_model = human_model.eval()
	
	object_model = models.alexnet(pretrained=True)
	object_model.classifier[6] = nn.Linear(4096,600)
	object_model = object_model.eval()
	
	pairwise_model = HO_RCNN_Pair()
	pairwise_model = pairwise_model.eval()

	human_model.load_state_dict(torch.load(os.path.join(args.model_path, 'human.pth')))
	object_model.load_state_dict(torch.load(os.path.join(args.model_path, 'object.pth')))
	pairwise_model.load_state_dict(torch.load(os.path.join(args.model_path, 'pairwise.pth')))

	if args.gpu == True:
		human_model.cuda()
		object_model.cuda()
		pairwise_model.cuda()

	criterion = nn.BCEWithLogitsLoss()
	print('--------------MODEL SETUP COMPLETE-------------')
	print('---------------TESTING MODEL-------------------')

	predictions = []
	outs = []

	batch_count = 0
	losses = []
	for human_crop, object_crop, int_pattern, outputs in test_data_loader:
		batch_count += 1
		human_crop = torch.stack([j for i in human_crop for j in i])
		object_crop = torch.stack([j for i in object_crop for j in i])
		int_pattern = torch.stack([j for i in int_pattern for j in i])
		outputs = torch.stack([j for i in outputs for j in i])
	
		human_crop = human_crop.float().cuda()
		object_crop = object_crop.float().cuda()
		int_pattern = int_pattern.float().cuda()
	
		with torch.no_grad():
			human_pred = human_model(human_crop)
			object_pred = object_model(object_crop)
			pairwise_pred = pairwise_model(int_pattern)
	
		total_pred = torch.add(torch.add(human_pred, object_pred), pairwise_pred)
		#total_pred_cpu = total_pred.cpu()
		#predictions.append(total_pred_cpu.numpy().astype(float))
		predictions.append(total_pred)
		#outs.append(outputs.numpy().astype(int))
		outs.append(outputs.cuda())
		batch_loss = criterion(total_pred, outputs.float().cuda())
		losses.append(batch_loss.item())
	
		if batch_count % 100 == 0:
			print('EVAL : Batch #' + str(batch_count) + ': Loss = ' + str(batch_loss.item()))

	final_loss = sum(losses) / len(test_data)
	print('<-------------------Final Validation Loss = '+str(final_loss) + '-------------------->')
	#outs = np.asarray(outs)
	sig = nn.Sigmoid()
	cm_item = namedtuple('cm_item',['hoi_id', 'tp','fp','tn','fn', 'num_pos', 'num_neg'])
	confusion_matrix = []
	for i in range(600):
		confusion_matrix.append([0,0,0,0,0,0])

	for i in range(len(predictions)): #<---- NUmber of batches passed to the network
		batch_pred = torch.unbind(predictions[i])
		batch_labels = torch.unbind(outs[i])
		for j in range(len(batch_pred)): #<------ Images per batch
			pred = batch_pred[j]

			pred = sig(pred)
			pred = (pred>args.threshold).int()
			pred = pred.cpu().numpy()
			labels = batch_labels[j].cpu().numpy().astype(int)

			for k in range(len(pred)):
				p = pred[k]
				l = labels[k]
				tp = 0
				fp = 0
				tn = 0
				fn = 0
				num_pos = 0
				num_neg = 0

				if p == 1 and l == 1: #True Positive
					confusion_matrix[k][0] += 1
					confusion_matrix[k][4] += 1
				elif p == 1 and l == 0: #False Positive
					confusion_matrix[k][1] += 1
					confusion_matrix[k][4] += 1
				elif p == 0 and l == 0: #True Negative
					confusion_matrix[k][2] += 1
					confusion_matrix[k][5] += 1
				elif p == 0 and l == 1: # False Negative
					confusion_matrix[k][3] += 1
					confusion_matrix[k][5] += 1
				else:
					print('WE HAVE A PROBLEM')

	# COMPUTE APs for EACH CLASS
	aps = []
	for i in range(600):
		if confusion_matrix[i][4] == 0:
			aps.append(0)
		else:
			aps.append(float(confusion_matrix[i][0] / confusion_matrix[i][4]))


	print('<------------MEAN AP = ' +str(sum(aps)/600) + '------------------->')
	for i in range(600):
		print('HOI_Class: ' + str(i+1) + ' AP= '+ str(aps[i]) + 'TruePos: ' + str(confusion_matrix[i][0]) + 'FalsePos: ' + str(confusion_matrix[i][1]) + 'TrueNeg: ' + str(confusion_matrix[i][2]) + 'FalseNeg: ' + str(confusion_matrix[i][3]) + 'NumPos: ' + str(confusion_matrix[i][4]) + 'NumNeg: ' + str(confusion_matrix[i][5]))
	print('<------------MEAN AP = ' +str(sum(aps)/600) + '------------------->')

	test_data.dataset_analysis()

if __name__ == "__main__":
	main()











