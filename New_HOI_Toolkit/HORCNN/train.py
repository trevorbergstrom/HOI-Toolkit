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
from sklearn import metrics
import pathlib

# Dataset stuff
sys.path.append('../Dataset')
from data_loader import HICODET_train, HICODET_test
from pair_only import HO_RCNN_Pair

# Set anomaly tracking:
torch.autograd.set_detect_anomaly(True)

# Steps that need to happen: 
# Create Dataset:
# 	Parameters: Detection Proposals file, fully constructed proposal file, image locations Batch size
# Create Models:
# Parameters none
def main():
	parser = argparse.ArgumentParser(description="Training the HORCNN Model!")
	parser.add_argument("bbmatfile", help="Path to the HICO-DET bounding box matfile", default='../Dataset/images/anno_bbox.mat', nargs='?')
	#parser.add_argument("train_path", help="Path to the file containing training images", default='../Dataset/images/train2015', nargs='?')
	
	parser.add_argument("train_path", help="Path to the file containing training images", default='../Dataset/images/train2015', nargs='?')
	#parser.add_argument("det_prop_file", help="Path of the object detection proposals pickle file. This file should contian detected objects from the images. If none is specified this program will run the FastRCNN Object detector over the training images to create the file", 
	#	default='../Dataset/images/pkl_files/fullTrain.pkl', nargs='?')
	
	parser.add_argument("det_prop_file", help="Path of the object detection proposals pickle file. This file should contian detected objects from the images. If none is specified this program will run the FastRCNN Object detector over the training images to create the file", 
		default='../Dataset/images/pkl_files/full_train2015.pkl', nargs='?')

	parser.add_argument("ho_prop_file", help="Path to the human-object proposals pickle file. This file contains the paired human-object proposals from the object proposals. If none is specified, will generate and save this file.",
		default='../Dataset/images/pkl_files/fullTrain_proposals.pkl', nargs='?')
	parser.add_argument("batch_size", help="batch_size for training", type=int, default=64, nargs='?')
	parser.add_argument("epochs", help="Number of training epochs", type=int, default=20, nargs='?')
	parser.add_argument("lr", help="Learning rate", type=float, default=0.0001, nargs='?')
	parser.add_argument("val_split", help="Percentage of training data to use for validation. e.g 20% = .2", type=float, default=0.2, nargs='?')
	parser.add_argument("gpu", help="Runninng on GPU?", type=bool, default=True, nargs='?')
	parser.add_argument("model_save_dir", help="Directory where saved models reside", default='./saved_models', nargs='?')
	parser.add_argument("save_each_epoch", help="Save every epoch?", type=bool, default=False, nargs='?')

	args = parser.parse_args()

	print(args)
	train_data = HICODET_train(args.train_path, args.bbmatfile, props_file=args.det_prop_file, props_list=args.ho_prop_file, proposal_count=4)

	x = train_data.get_img_props(train_data.proposals[0], train_data.annotations[0], 8)
	#print(x)
	n_total = len(train_data)
	n_valid = int(n_total * args.val_split)
	n_train = int(n_total - n_valid)

	n_train_batches = n_train / args.batch_size

	print('Length of training set: ' + str(n_train))
	print('Length of validation set: ' + str(n_valid))

	train_set, valid_set = torch.utils.data.random_split(train_data, (n_train, n_valid))

	train_data_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=4, shuffle=True)
	valid_data_loader = torch.utils.data.DataLoader(dataset = valid_set, batch_size=1, shuffle=False)

	'''
	for h,o,p,l in train_data_loader:
		h=torch.stack([j for i in h for j in i])
		o=torch.stack([j for i in o for j in i])
		p=torch.stack([j for i in p for j in i])
		l=torch.stack([j for i in l for j in i])
		print(h.shape)
		print(o.shape)
		print(p.shape)
		print(l.shape)

	exit()
	'''
	print('------------DATASET SETUP COMPLETE-------------')
	print('\n')
	print('------------Setting Up Models------------------')

	human_model = models.alexnet(pretrained=True)
	human_model.classifier[6] = nn.Linear(4096,600)
	human_model = human_model.train()
	
	object_model = models.alexnet(pretrained=True)
	object_model.classifier[6] = nn.Linear(4096,600)
	object_model = object_model.train()
	
	pairwise_model = HO_RCNN_Pair()
	pairwise_model = pairwise_model.train()

	if args.gpu == True:
		human_model.cuda()
		object_model.cuda()
		pairwise_model.cuda()

	for param in object_model.parameters():
		param.requires_grad = True
	for param in human_model.parameters():
		param.requires_grad = True
	for param in pairwise_model.parameters():
		param.requires_grad = True

	criterion = nn.BCEWithLogitsLoss()
	all_params = list(human_model.parameters()) + list(object_model.parameters()) + list(pairwise_model.parameters())
	optimizer = optim.SGD(all_params, lr = args.lr)

	print('--------------MODEL SETUP COMPLETE-------------')
	print('\n')

	path_list = []
	if args.save_each_epoch == True:
		print('--------------Setting Up Saves-----------------')
		for i in range(args.epochs):
			folder = 'epoch' + str(i+1)
			path = os.path.join(args.model_save_dir, folder)
			pathlib.Path(path).mkdir(parents=True, exist_ok=True)
			path_list.append(path)
		print('------CREATING SAVE DIRECTORY COMPLETE---------')


	print('------------------Training---------------------')

	for epoch in range(args.epochs):
		losses = []
		batch_count = 0
		final_batch_loss = 0.
		for human_crop, object_crop, int_pattern, outputs in train_data_loader:
			batch_count += 1
			human_crop = torch.stack([j for i in human_crop for j in i])
			object_crop = torch.stack([j for i in object_crop for j in i])
			int_pattern = torch.stack([j for i in int_pattern for j in i])
			outputs = torch.stack([j for i in outputs for j in i])

			human_crop = human_crop.float().cuda()
			object_crop = object_crop.float().cuda()
			int_pattern = int_pattern.float().cuda()

			with torch.enable_grad():
				human_pred = human_model(human_crop)
				object_pred = object_model(object_crop)
				pairwise_pred = pairwise_model(int_pattern)

			total_pred = torch.add(torch.add(human_pred, object_pred), pairwise_pred)
			batch_loss = criterion(total_pred, outputs.float().cuda())

			if batch_count % 100 == 0:
				print('Epoch #' + str(epoch) + ': Batch #' + str(batch_count) + '/' + str(n_train_batches) + ': Loss = ' + str(batch_loss.item()))

			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			final_batch_loss = batch_loss

		print('Epoch #'+str(epoch)+': Loss = '+str(final_batch_loss.item()))

		if args.save_each_epoch==True:
			print('-------------Saving Model----------------')
			h_path = os.path.join(path_list[epoch], 'human.pth')
			o_path = os.path.join(path_list[epoch], 'object.pth')
			p_path = os.path.join(path_list[epoch], 'pairwise.pth')
			torch.save(human_model.state_dict(), h_path)
			torch.save(object_model.state_dict(), o_path)
			torch.save(pairwise_model.state_dict(), p_path)
			print('-----------SAVE MODEL COMPLETE-----------')

		# Probably need to do validation evaluation here.

	if args.save_each_epoch==False:
		folder = os.path.join(args.model_save_dir, 'Final_Trained')
		os.mkdir(folder)
		h_path = os.path.join(folder, 'human.pth')
		o_path = os.path.join(folder, 'object.pth')
		p_path = os.path.join(folder, 'pairwise.pth')
		torch.save(human_model.state_dict(), h_path)
		torch.save(object_model.state_dict(), o_path)
		torch.save(pairwise_model.state_dict(), p_path)




if __name__ == "__main__":
	main()