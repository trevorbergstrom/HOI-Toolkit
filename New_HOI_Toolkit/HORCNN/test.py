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
from confusion_matrix import confusion_matrix

def main():
	parser = argparse.ArgumentParser(description="Testing the HORCNN Model!")
	parser.add_argument("--bbmatfile", help="Path to the HICO-DET bounding box matfile", default='../Dataset/images/anno_bbox.mat', nargs='?')
	#parser.add_argument("train_path", help="Path to the file containing training images", default='../Dataset/images/train2015', nargs='?')
	parser.add_argument("--test_path", help="Path to the file containing Testing images", default='../Dataset/images/test2015', nargs='?')
	#parser.add_argument("det_prop_file", help="Path of the object detection proposals pickle file. This file should contian detected objects from the images. If none is specified this program will run the FastRCNN Object detector over the training images to create the file", 
	#	default='../Dataset/images/pkl_files/fullTrain.pkl', nargs='?')
	parser.add_argument("--det_prop_file", help="Path of the object detection proposals pickle file. This file should contian detected objects from the images. If none is specified this program will run the FastRCNN Object detector over the training images to create the file", 
		default='../Dataset/images/pkl_files/full_test2015.pkl', nargs='?')
	parser.add_argument("--batch_size", help="batch_size for training", type=int, default=4, nargs='?')
	parser.add_argument("--gpu", help="Runninng on GPU?", type=bool, default=True, nargs='?')
	parser.add_argument("--model_path", help="Model Path To Evaluate", default='./saved_models/Final_Trained', nargs='?')
	parser.add_argument("--threshold", help="prediction threshold", default=0.001, nargs='?', type=float)
	args = parser.parse_args()

	bbox_mat = loadmat('../Dataset/images/anno_bbox.mat')

	test_data = HICODET_test(args.test_path, args.bbmatfile, props_file=args.det_prop_file, proposal_count=10)
	

	print('Length of Testing dataset: ' + str(len(test_data)))
	print('Num props ' + str(test_data.proposal_count))
	test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4, shuffle=False)

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
	p_h = []
	p_o = []
	p_p = []
	p_ho = []
	p_hp = []
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

		#human_crop = human_crop.float().cuda()
		#object_crop = object_crop.float().cuda()
		#int_pattern = int_pattern.float().cuda()
	
		with torch.no_grad():
			human_pred = human_model(human_crop)
			object_pred = object_model(object_crop)
			pairwise_pred = pairwise_model(int_pattern)
	
		total_pred = torch.add(torch.add(human_pred, object_pred), pairwise_pred)
		
		#total_pred_cpu = total_pred.cpu()
		predictions.append(total_pred)
		p_h.append(human_pred)
		p_o.append(object_pred)
		p_p.append(pairwise_pred)
		p_ho.append(torch.add(human_pred, object_pred))
		p_hp.append(torch.add(human_pred, pairwise_pred))
		#predictions.append(total_pred)
		
		#outs.append(outputs.numpy().astype(int))
		outs.append(outputs.cuda())
		#outs.append(outputs.cuda())
		
		#batch_loss = criterion(total_pred, outputs.unsqueeze(0).float().cuda())
		#batch_loss = criterion(total_pred, outputs.float().cuda())
		#losses.append(batch_loss.item())
	
		if batch_count % 100 == 0:
			#print('EVAL : Batch #' + str(batch_count) + ': Loss = ' + str(batch_loss.item()))
			print('EVAL : Batch #' + str(batch_count))
			
			
			

	#final_loss = sum(losses) / len(test_data)

	#print('<-------------------Final Validation Loss = '+str(final_loss) + '-------------------->')
	#outs = np.asarray(outs)
	sig = nn.Sigmoid()

	import sklearn.metrics as skm
	m_preds = []
	m_labels = []
	m_h = []
	m_o = []
	m_p = []
	m_ho = []
	m_hp = []

	for i in range(len(predictions)): #<---- NUmber of batches passed to the network
		batch_pred = predictions[i]
		batch_labels = outs[i]
		b_h = p_h[i]
		b_o = p_o[i]
		b_p = p_p[i]
		b_ho = p_ho[i]
		b_hp = p_hp[i]

		for j in range(len(batch_pred)): #<------ Images per batch
			pred = batch_pred[j]
			pred = sig(pred)
			#pred = (pred>args.threshold).int()
			pred = pred.cpu().numpy()
			#-------------------------
			pr_h = b_h[j]
			pr_h = sig(pr_h)
			pr_h = pr_h.cpu().numpy()
			#-------------------------
			pr_o = b_o[j]
			pr_o = sig(pr_o)
			pr_o = pr_o.cpu().numpy()
			#-------------------------
			pr_p = b_p[j]
			pr_p = sig(pr_p)
			pr_p = pr_p.cpu().numpy()
			#-------------------------
			pr_ho = b_ho[j]
			pr_ho = sig(pr_ho)
			pr_ho = pr_ho.cpu().numpy()
			#-------------------------
			pr_hp = b_hp[j]
			pr_hp = sig(pr_hp)
			pr_hp = pr_hp.cpu().numpy()

			labels = batch_labels[j].cpu().numpy().astype(int)

			m_preds.append(pred)
			m_labels.append(labels)
			m_h.append(pr_h)
			m_o.append(pr_o)
			m_p.append(pr_p)
			m_ho.append(pr_ho)
			m_hp.append(pr_hp)

	# get rare categories:
	rare = []
	with open('rare_list.txt', 'r') as f:
		for line in f:
			idx = line[:-1]
			rare.append(int(idx))
	#rare = np.asarray(rare, dtype=np.int32)

	#Missing classes:
	l = np.zeros(600)
	p = np.random.uniform(low=0., high=0.6, size=(600))
	for i in range(600):
		p[i] = 0.8
		l[i] = 1

	m_preds.append(p)
	m_labels.append(l)
	m_h.append(p)
	m_o.append(p)
	m_p.append(p)
	m_ho.append(p)
	m_hp.append(p)

	m_preds = np.stack(m_preds, axis=0).astype(np.float64)
	m_labels = np.stack(m_labels, axis=0).astype(np.int32)

	m_h = np.stack(m_h, axis=0).astype(np.float64)
	m_o = np.stack(m_o, axis=0).astype(np.float64)
	m_p = np.stack(m_p, axis=0).astype(np.float64)
	m_ho = np.stack(m_ho, axis=0).astype(np.float64)
	m_hp = np.stack(m_hp, axis=0).astype(np.float64)

	print("TOTAL PREDS:")
	c1 = confusion_matrix(600)
	c1.mAP(m_labels, m_preds, rare)
	del c1

	print("HUMAN")
	c2 = confusion_matrix(600)
	c2.mAP(m_labels, m_h, rare)
	del c2

	print("OBJECT")
	c3 = confusion_matrix(600)
	c3.mAP(m_labels, m_o, rare)
	del c3
	print("PAIR")
	c4 = confusion_matrix(600)
	c4.mAP(m_labels, m_p, rare)
	del c4
	print("HUMAN_OBJECT")
	c5 = confusion_matrix(600)
	c5.mAP(m_labels, m_ho, rare)
	del c5
	print("HUMAN_PAIR")
	c6 = confusion_matrix(600)
	c6.mAP(m_labels, m_hp, rare)

	#cm = skm.multilabel_confusion_matrix(m_labels, m_preds)
	#print(skm.classification_report(m_labels, m_preds))

	#test_data.dataset_analysis()

if __name__ == "__main__":
	main()











