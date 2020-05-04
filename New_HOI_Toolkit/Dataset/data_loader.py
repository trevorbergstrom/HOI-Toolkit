from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
import sys
import convert_hico_mat as tools
sys.path.append('../FasterRCNN')
from faster_RCNN_detector import FRCNN_Detector
from random import choices
import pickle

'''
This file contains the HICODET_Dataloader class
'''
class HICODET_test(Dataset):
	def __init__(self, folder_path, bbox_matlist, img_size=256, proposal_count=8, props_file='none'):

		self.img_size = img_size
		self.proposal_count = proposal_count
		# Set path
		self.img_folder_path = folder_path
		# Create Simple Annotation lists
		self.annotations= tools.convert_bbox_matlist(bbox_matlist['bbox_test'], bbox_matlist['list_action'])
		self.interaction_prop_list = tools.convert_actions(bbox_matlist['list_action'])
		self.img_names = [img[0] for img in self.annotations[:-1]] # <-------- [:-1]
		#self.img_names = [img[0] for img in self.annotations[:50]] # <-------- [:-1]

		if props_file == 'none':
			# Now do proposal detection:
			detector = FRCNN_Detector()
			print('Test Set Generating Proposals with Detector')
			self.proposals = detector.get_data_preds(self.img_names, folder_path, proposal_count)
			print('Done')
			del(detector)
		else:
			with open(props_file, 'rb') as f:
				self.proposals = pickle.load(f)

	def __len__(self):
		return len(self.img_names)

	def __getitem__(self, idx):
		# im not 100% sure how this works but I assume getitem[idx] pulls a random img
		# We need to select 8 random proposals from tthe proposal list.
		# Then we need to create the human, object crops and pairwise img.
		# REsize the images to appropriate size
		# Then we need to createt the ground truth vector. This vector is 1x600 and the only classes the show up are interactions from the img_hoi_list that have objects in the proposal. 

		img = self.img_names[idx]
		annots = self.annotations[idx]
		props = self.proposals[idx]

		rand_props = choices(props, k=self.proposal_count)
		crop_list = []
		for pair in rand_props:
			crop_list.append([pair[0][0], pair[1][0]])

		input_list = tools.multi_crop_img(crop_list, os.path.join(self.img_folder_path, img), self.img_size)

		for i in range(len(input_list)):
			input_list[i].append( tools.create_interaction_pattern(annots[1], annots[2], crop_list[i][0], crop_list[i][1], self.img_size) )
	
		# Next we need to grab all the hoi_ids from the list of GT hois
		hois_in_img = []
		for hoi_l in annots[4]:
			hois_in_img.append(hoi_l[0])

		hois_in_img = tools.remove_dupes(hois_in_img)

		# next get all the objects from the proposals:
		objects_in_proposals = []
		for a_prop in props:
			if a_prop[0][2] != 'null':
				objects_in_proposals.append(a_prop[1][2])

		# Remove duplicate objects
		objects_in_proposals = tools.remove_dupes(objects_in_proposals)

		# then cross refrence all objects in the hois from the GTs. Add the hoi_id of any hoi that contains an interaction with the object:
		final_hoi_list = []
		for a_hoi in hois_in_img:
			if self.interaction_prop_list[a_hoi-1][1] in objects_in_proposals:
				final_hoi_list.append(a_hoi)
		
		#outputs = tools.build_gt_vec(final_hoi_list)
		outputs = final_hoi_list
		return input_list, outputs
#==========================================================================================================================================================================================
class HICODET_train(Dataset):
	def __init__(self, folder_path, bbox_matlist, img_size=256, proposal_count=8, props_file='none'):
		
		self.img_size = img_size
		self.proposal_count = proposal_count

		self.img_folder_path = folder_path

		self.annotations = tools.convert_bbox_matlist(bbox_matlist['bbox_train'], bbox_matlist['list_action'])
		self.interaction_prop_list = tools.convert_actions(bbox_matlist['list_action'])
		self.img_names = [img[0] for img in self.annotations[:-1]] # <--- Change size for larger set
		#self.img_names = [img[0] for img in self.annotations[:50]] # <--- Change size for larger set

		if props_file == 'none':
			detector = FRCNN_Detector()
			print('Test Set Generating Proposals with Detector')
			self.proposals = detector.get_data_preds(self.img_names, folder_path, proposal_count)
			print('Done')

			# We dont need FRCNN to hangout and clog GPU memory after generating proposals. 
			del(detector)
		else:
			with open(props_file, 'rb') as f:
				self.proposals = pickle.load(f)

	def __len__(self):
		return len(self.img_names)

	def __getitem__(self, idx):
		# im not 100% sure how this works but I assume getitem[idx] pulls a random img
		# We need to select 8 random proposals from tthe proposal list.
		# Then we need to create the human, object crops and pairwise img.
		# REsize the images to appropriate size
		# Then we need to createt the ground truth vector. This vector is 1x600 and the only classes the show up are interactions from the img_hoi_list that have objects in the proposal. 

		img = self.img_names[idx]
		print(img)
		annots = self.annotations[idx]
		props = self.proposals[idx]

		rand_props = choices(props, k=self.proposal_count)
		#print(rand_props)
		crop_list = []
		for pair in rand_props:
			if pair[0][0][0] != 0 and pair[0][0][1] != 0 and pair[0][0][2] != 0 and pair[0][0][3] != 0:
				crop_list.append([pair[0][0], pair[1][0]])

		input_list = tools.multi_crop_img(crop_list, os.path.join(self.img_folder_path, img), self.img_size)

		for i in range(len(input_list)):
			ip = tools.create_interaction_pattern(annots[1], annots[2], crop_list[i][0], crop_list[i][1], self.img_size)
			#print(ip.shape)
			input_list[i].append(ip)
	
		# Next we need to grab all the hoi_ids from the list of GT hois
		hois_in_img = []
		for hoi_l in annots[4]:
			hois_in_img.append(hoi_l[0])

		hois_in_img = tools.remove_dupes(hois_in_img)

		# next get all the objects from the proposals:
		objects_in_proposals = []
		for a_prop in props:
			if a_prop[0][2] != 'null':
				objects_in_proposals.append(a_prop[1][2])

		# Remove duplicate objects
		objects_in_proposals = tools.remove_dupes(objects_in_proposals)

		# then cross refrence all objects in the hois from the GTs. Add the hoi_id of any hoi that contains an interaction with the object:
		final_hoi_list = []
		for a_hoi in hois_in_img:
			#print(a_hoi)
			if self.interaction_prop_list[a_hoi-1][1] in objects_in_proposals:
				final_hoi_list.append(a_hoi)
		
		#outputs = tools.build_gt_vec(final_hoi_list)
		#print(final_hoi_list)
		outputs = final_hoi_list

		return input_list, outputs
