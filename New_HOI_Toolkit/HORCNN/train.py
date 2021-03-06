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
	parser.add_argument("--bbmatfile", help="Path to the HICO-DET bounding box matfile", default='../Dataset/images/anno_bbox.mat', nargs='?')
	#parser.add_argument("train_path", help="Path to the file containing training images", default='../Dataset/images/train2015', nargs='?')
	
	parser.add_argument("--train_path", help="Path to the file containing training images", default='../Dataset/images/train2015', nargs='?')
	#parser.add_argument("det_prop_file", help="Path of the object detection proposals pickle file. This file should contian detected objects from the images. If none is specified this program will run the FastRCNN Object detector over the training images to create the file", 
	#	default='../Dataset/images/pkl_files/fullTrain.pkl', nargs='?')
	
	parser.add_argument("--det_prop_file", help="Path of the object detection proposals pickle file. This file should contian detected objects from the images. If none is specified this program will run the FastRCNN Object detector over the training images to create the file", 
		default='../Dataset/images/pkl_files/full_train2015.pkl', nargs='?')

	parser.add_argument("--batch_size", help="batch_size for training", type=int, default=4, nargs='?')
	parser.add_argument("--epochs", help="Number of training epochs", type=int, default=2, nargs='?')
	parser.add_argument("--lr", help="Learning rate", type=float, default=0.001, nargs='?')
	parser.add_argument("--val_split", help="Percentage of training data to use for validation. e.g 20% = .2", type=float, default=0.2, nargs='?')
	parser.add_argument("--multi_gpu", help="Run on multiple GPUs? (max=2)", type=bool, default=False, nargs='?')
	parser.add_argument("--gpu", help="Runninng on GPU?", type=bool, default=True, nargs='?')
	parser.add_argument("--model_save_dir", help="Directory where saved models reside", default='./saved_models', nargs='?')
	parser.add_argument("--save_each_epoch", help="Save every epoch?", type=bool, default=False, nargs='?')
	parser.add_argument("--cp_path", help="Checkpoint Model Path", default='none', nargs='?')

	args = parser.parse_args()

	if args.batch_size < 4:
		print('Batch Size for training must be greater than 4. Changing')
		args.batch_size = 4

	print('<-------------- Training Properties-------------->')
	print('Bounding Box File: ' + args.bbmatfile)
	print('Path to training images: ' + args.train_path)
	print('Path for pre-computed detection proposals: ' + args.det_prop_file)
	print('Batch_size: ' + str(args.batch_size))
	print('Epochs: ' + str(args.epochs))
	print('Learning Rate: ' + str(args.lr))
	print('Validation set split: ' + str(args.val_split))
	print('Training on GPU?: ' + str(args.gpu))
	print('Training with multiple GPUs?: ' + str(args.multi_gpu))
	print('Directory to save models: ' + args.model_save_dir)
	print('Saving After Every Epoch?: ' + str(args.save_each_epoch))
	print('Checkpoint File to Resume Training: ' + args.cp_path)
	print('<------------------------------------------------->')


	train_data = HICODET_train(args.train_path, args.bbmatfile, props_file=args.det_prop_file, proposal_count=args.batch_size)

	x = train_data.get_img_props(train_data.proposals[0], train_data.annotations[0], 8)
	
	n_total = len(train_data)
	n_valid = int(n_total * args.val_split)
	n_train = int(n_total - n_valid)

	n_train_batches = int(n_train / args.batch_size)

	print('Length of training set: ' + str(n_train))
	print('Length of validation set: ' + str(n_valid))

	train_set, valid_set = torch.utils.data.random_split(train_data, (n_train, n_valid))

	train_data.dataset_analysis()
	exit()

	train_data_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=args.batch_size, shuffle=True)
	valid_data_loader = torch.utils.data.DataLoader(dataset = valid_set, batch_size=1, shuffle=False)

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

	if args.cp_path != 'none':
		print('Training from Checkpoint: ' + args.cp_path)
		human_model.load_state_dict(torch.load(os.path.join(args.cp_path, 'human.pth')))
		object_model.load_state_dict(torch.load(os.path.join(args.cp_path, 'object.pth')))
		pairwise_model.load_state_dict(torch.load(os.path.join(args.cp_path, 'pairwise.pth')))

	dev_gpu_1 = 0
	dev_gpu_2 = 0
	
	if args.multi_gpu == True:
		dev_gpu_2 = 1

	if args.gpu == True:
		human_model.cuda(device=dev_gpu_1)
		object_model.cuda(device=dev_gpu_2)
		pairwise_model.cuda(device=dev_gpu_2)

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
	final_folder =''
	if args.save_each_epoch == True:
		print('--------------Setting Up Saves-----------------')
		for i in range(args.epochs):
			folder = 'epoch' + str(i+1)
			path = os.path.join(args.model_save_dir, folder)
			pathlib.Path(path).mkdir(parents=True, exist_ok=True)
			path_list.append(path)

		final_folder = os.path.join(args.model_save_dir, 'Final_Trained')
		pathlib.Path(final_folder).mkdir(parents=True, exist_ok=True)
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

			human_crop = human_crop.float().cuda(device=dev_gpu_1)
			object_crop = object_crop.float().cuda(device=dev_gpu_2)
			int_pattern = int_pattern.float().cuda(device=dev_gpu_2)

			with torch.enable_grad():
				human_pred = human_model(human_crop)
				object_pred = object_model(object_crop)
				pairwise_pred = pairwise_model(int_pattern)

			total_pred = torch.add(torch.add(human_pred, object_pred.cuda(device=dev_gpu_1)), pairwise_pred.cuda(device=dev_gpu_1))
			batch_loss = criterion(total_pred, outputs.float().cuda(device=dev_gpu_1))
			losses.append(batch_loss.item())

			if batch_count % 100 == 0:
				print('Epoch #' + str(epoch) + ': Batch #' + str(batch_count) + '/' + str(n_train_batches) + ': Loss = ' + str(batch_loss.item()))

			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

		final_loss = sum(losses) / n_train
		print('<-------------------Epoch #'+str(epoch)+': Loss = '+str(final_loss) + '-------------------->')

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

	print('-------------------------TRAINING COMPLETE----------------')
	print('-------------Saving FINAL Model----------------')
	if args.save_each_epoch==False:
		h_path = os.path.join(folder, 'human.pth')
		o_path = os.path.join(folder, 'object.pth')
		p_path = os.path.join(folder, 'pairwise.pth')
		torch.save(human_model.state_dict(), h_path)
		torch.save(object_model.state_dict(), o_path)
		torch.save(pairwise_model.state_dict(), p_path)
	print('-----------SAVE FINAL MODEL COMPLETE-----------')

	#------------------------------- EVAL ON VALID SET-----------------------------
	print('-----------Performing Evaluation On Validation Set-----------')
	object_model.cuda(device=dev_gpu_1)
	pairwise_model.cuda(device=dev_gpu_1)
	human_model.eval()
	object_model.eval()
	pairwise_model.eval()

	batch_count = 0
	losses = []
	for human_crop, object_crop, int_pattern, outputs in valid_data_loader:
		batch_count += 1
		human_crop = torch.stack([j for i in human_crop for j in i])
		object_crop = torch.stack([j for i in object_crop for j in i])
		int_pattern = torch.stack([j for i in int_pattern for j in i])
		outputs = torch.stack([j for i in outputs for j in i])
	
		human_crop = human_crop.float().cuda(device=dev_gpu_1)
		object_crop = object_crop.float().cuda(device=dev_gpu_1)
		int_pattern = int_pattern.float().cuda(device=dev_gpu_1)
	
		with torch.no_grad():
			human_pred = human_model(human_crop)
			object_pred = object_model(object_crop)
			pairwise_pred = pairwise_model(int_pattern)
	
		total_pred = torch.add(torch.add(human_pred, object_pred), pairwise_pred)
		batch_loss = criterion(total_pred, outputs.float(device=dev_gpu_1).cuda())
		losses.append(batch_loss.item())
	
		if batch_count % 100 == 0:
			print('EVAL : Batch #' + str(batch_count) + ': Loss = ' + str(batch_loss.item()))

	final_loss = sum(losses) / n_valid
	print('<-------------------Final Validation Loss = '+str(final_loss) + '-------------------->')
if __name__ == "__main__":
	main()