import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torchvision import models
from PIL import Image, ImageDraw
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
sys.path.append('../FastRCNN')
from data_loader import HICODET_train, HICODET_test
from pair_only import HO_RCNN_Pair
from faster_RCNN_detector import FRCNN_Detector
import convert_hico_mat as tools

def main():
	parser = argparse.ArgumentParser(description="Testing the HORCNN Model on a single image!")
	
	parser.add_argument("--model_path", help="Model Path To Evaluate", default='./saved_models/Final_Trained', nargs='?')
	parser.add_argument("--threshold", help="prediction threshold", default=0.01, nargs='?', type=float)
	parser.add_argument("--det_threshold", help="detection threshold", default=0.9, nargs='?', type=float)
	parser.add_argument("--img_path", help="Path to Image Folder")
	parser.add_argument("--img_name", help="Name of image")
	parser.add_argument("--gpu", help="Run on GPU?", default=True, type=bool)
	parser.add_argument('--show', help="Show image results?", default=False, type=bool)

	args = parser.parse_args()

	detector = FRCNN_Detector()

	proposals = detector.get_data_preds([args.img_name], args.img_path, 10)
	print("Proposals")
	full_img_pth = os.path.join(args.img_path, args.img_name)
	proposals = proposals[0][1]
	#for i in range(len(proposals)):
	#	print('\n')
	#	print(proposals[i])
	del detector

	print('------------Setting Up Model------------------')
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

	good_props = []

	print('Running Infrence')
	for prop in proposals:
		human = prop[0]
		obj = prop[1]
		#print('Human: ' + str(human[1]))
		#print('Object: ' + str(obj[1]) +' ' + obj[2])

		if min(human[1], obj[1]) > args.det_threshold:
			#print('GOOD ' + obj[2])
			human_crop, object_crop = tools.crop_pair(human[0], obj[0], full_img_pth, 256)
			pairwise = tools.create_interaction_pattern(human[0], obj[0], 256)

			human_crop = torch.from_numpy(human_crop).unsqueeze(0).float().cuda()
			object_crop = torch.from_numpy(object_crop).unsqueeze(0).float().cuda()
			pairwise = pairwise.unsqueeze(0).float().cuda()

			with torch.no_grad():
				human_pred = human_model(human_crop)
				object_pred = object_model(object_crop)
				pairwise_pred = pairwise_model(pairwise)

			total_pred = torch.add(torch.add(human_pred, object_pred), pairwise_pred)
			#total_pred = pairwise_pred
			sig = nn.Sigmoid()
			total_pred = sig(total_pred)
			sig_pred = (total_pred >args.threshold).int()
			total_pred = total_pred.cpu().numpy()
			sig_pred = sig_pred.cpu().numpy()

			good_props.append([human, obj, sig_pred, total_pred])

	for prop in good_props:
		#print(prop[2][0])
		#print(prop[3][0])
		classes = np.where(prop[2]==1)

		classes = [x for x in classes[1]]
		scores = [prop[3][0][i] for i in classes]
		classes = [x+1 for x in classes]
		
		print("Prop: " + "object= " + prop[1][2] + ' score= ' + prop[1][2] + ' HOI=' + str(classes) + ' scores=' + str(scores))
		zip_list = zip(scores, classes)
		s_z = sorted(zip_list)
		l = len(s_z)-1
		print('Top3:')
		for s in range(3):
			print(s_z[l-s])

	if args.show == True and len(good_props) != 0:
		im = Image.open(os.path.join(args.img_path, args.img_name))
		im_d = ImageDraw.Draw(im)
		# Draw scoring hoi
		for prop in good_props:
		#prop = good_props[0]
			classes = np.where(prop[2]==1)
			classes = [x for x in classes[1]]
			scores = [prop[3][0][i] for i in classes]
			classes = [x+1 for x in classes]
			hbb = prop[0][0]
			obb = prop[1][0]
			im_d.rectangle([hbb[0],hbb[1],hbb[2],hbb[3]], outline='green', width=2)
			im_d.rectangle([obb[0],obb[1],obb[2],obb[3]], outline='blue', width=2)
		
		im.show()


if __name__ == "__main__":
	main()
