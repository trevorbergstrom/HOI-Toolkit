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
from random import choices, sample
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
		self.img_names = [img[0] for img in self.annotations] # <-------- [:-1]
		#self.img_names = [img[0] for img in self.annotations[:50]] # <-------- [:-1]

		if props_file == 'none':
			# Now do proposal detection:
			detector = FRCNN_Detector()
			print('Test Set Generating Proposals with Detector')
			self.proposals = detector.get_data_preds(self.img_names, folder_path, proposal_count)
			print('Saving proposals in pickle file')
			tools.pickle_proposals(self.proposals, 'images/pkl_files/fullTest.pkl')
			print('Done')
			del(detector)
		else:
			with open(props_file, 'rb') as f:
				self.proposals = pickle.load(f)

	def __len__(self):
		return len(self.img_names)

	def __getitem__(self, idx):
		# Image Name
		img = self.img_names[idx]
		print(img)
		# Ground truth annotataions
		annots = self.annotations[idx]
		# Proposals generated by object detector
		props = self.proposals[idx]
		# Pick random proposals fromt he current image
		rand_props = sample(props, k=min(self.proposal_count, len(props)))

		img_crops = []
		img_labels = []
		# For each proposal:
		for pair in rand_props:
			# Create crops of each object of interest from the original image
			human_img, obj_img = tools.crop_pair(pair[0][0], pair[1][0], os.path.join(self.img_folder_path, img), self.img_size)
			# Create the interaction pattern for the proposal crops:
			pair_img = tools.create_interaction_pattern(pair[0][0], pair[1][0], self.img_size)
			# Get the name of the object in question.
			obj_name = pair[1][2]

			gt_hois = annots.hoi_list
			confirmed_hoi_list = []

			# Loop though the list of ground truth hois, 
			for gt in gt_hois:
				# if a hoi contains the object in the current proposal
				if gt.obj == obj_name:
					# Loop through the ground truth connections
					for conn in gt.connections:
						# Get the index for the human and object in the GT bounding box list
						o_idx = conn[1]
						h_idx = conn[0]
						# Compute the current connection's human and object IOUs with the proposal
						iou_h = tools.compute_iou(pair[0][0], gt.human_boxes[h_idx-1])
						iou_o = tools.compute_iou(pair[1][0], gt.object_boxes[o_idx-1])
						# If we are above 0.5 we can add this hoi to our list of positive HOIs
						if min(iou_o, iou_h) > 0.5:
							confirmed_hoi_list.append(gt.hoi_id.astype(np.int32))
						#else:
							#confirmed_hoi_list.append(-1)

			img_crops.append([human_img, obj_img, pair_img])
			#img_labels.append(np.asarray(confirmed_hoi_list).astype(np.int32))
			img_labels.append(tools.build_gt_vec(confirmed_hoi_list).astype(np.int32))

		return img_crops, img_labels

#==========================================================================================================================================================================================
class HICODET_train(Dataset):
	def __init__(self, folder_path, bbox_matlist, img_size=256, proposal_count=8, props_file='none', props_list='none'):
		
		self.img_size = img_size
		self.proposal_count = proposal_count

		self.img_folder_path = folder_path

		self.annotations = tools.convert_bbox_matlist(bbox_matlist['bbox_train'], bbox_matlist['list_action'])
		self.interaction_prop_list = tools.convert_actions(bbox_matlist['list_action'])
		self.img_names = [img[0] for img in self.annotations] # <--- Change size for larger set
		#self.img_names = [img[0] for img in self.annotations[:50]] # <--- Change size for larger set

		if props_file == 'none':
			detector = FRCNN_Detector()
			print('Test Set Generating Proposals with Detector')
			self.proposals = detector.get_data_preds(self.img_names, folder_path, proposal_count)
			tools.pickle_proposals(self.proposals, 'images/pkl_files/fullTrain.pkl')
			print('Done')

			# We dont need FRCNN to hangout and clog GPU memory after generating proposals. 
			del(detector)
		else:
			print('Loading Precomputed Detection Proposals From Files')
			with open(props_file, 'rb') as f:
				self.proposals = pickle.load(f)

		if props_list == 'none':
			print('Creating list of proposals')
			self.create_prop_list()
			tools.pickle_proposals(self.proposal_set, 'images/pkl_files/fullTrain_proposals.pkl')
			print('Done')
		else:
			print('Loading Precomputed Human-Object Proposals from File')
			with open(props_list, 'rb') as f:
				self.proposal_set = pickle.load(f)

		del(self.proposals)
		del(self.annotations)

	def __len__(self):
		return len(self.proposal_set)

	def create_prop_list(self):
		# each prop: [img_name][propbboxh][propbboxo][GTvec][det_score_h][det_score_o]
		self.proposal_set = []

		# For each Image:
		for i in range(len(self.proposals)):
			# Need the image name later to crop out of 
			image_name = self.img_names[i]

			# For each proposal generated by the detector in this image:
			for proposal in self.proposals[i]:
				
				# Need the object name and the list of possible HOIs
				obj_name = proposal[1][2]
				gt_hois = self.annotations[i].hoi_list
				confirmed_hoi_list = []

				# For each possible HOI in the image
				for gt in gt_hois:
					# If the object in the HOI is the same as the object in the proposal from the detector
					if gt.obj == obj_name:
						
						# For all the connections in the groundtruth hoi:
						for conn in gt.connections:
							# The connections hold indicies of the humans and objects that participate in this hoi
							o_idx=conn[1]
							h_idx=conn[0]
							# Given the groundtruth bounding boxes that match the proposed object compute IOUs between the proposal and groudtruths
							iou_h = tools.compute_iou(proposal[0][0], gt.human_boxes[h_idx-1])
							iou_o = tools.compute_iou(proposal[1][0], gt.object_boxes[o_idx-1])

							# Check IOU mins to determine adding this as a true positive and assign to GT vector
							if min(iou_o, iou_h) > 0.5:
								confirmed_hoi_list.append(gt.hoi_id.astype(np.int32))

				gt_vector = tools.build_gt_vec(confirmed_hoi_list).astype(np.int32)
				#append all the relevant information to the proposal set
				self.proposal_set.append([image_name, proposal[0][0], proposal[1][0], gt_vector, proposal[0][1], proposal[1][1]])			

	def __getitem__(self, idx):
		# Need to take the proposal set and the index, crop the image, create the interaction pattern. returns 
		proposal = self.proposal_set[idx]
		# Crop human and objects from the main image
		human_crop, object_crop = tools.crop_pair(proposal[1], proposal[2], os.path.join(self.img_folder_path, proposal[0]), self.img_size)
		# Create interaction pattern:
		pair = tools.create_interaction_pattern(proposal[1], proposal[2], self.img_size)

		# When we need the object score this is where we need to send them!
		return human_crop, object_crop, pair, proposal[3]
'''		
	def __getitem__(self, idx):

		# Image Name
		img = self.img_names[idx]
		print(img)
		# Ground truth annotataions
		annots = self.annotations[idx]
		# Proposals generated by object detector
		props = self.proposals[idx]
		# Pick random proposals fromt he current image
		rand_props = sample(props, k=min(self.proposal_count, len(props)))

		img_crops = []
		img_labels = []
		# For each proposal:
		for pair in rand_props:
			# Create crops of each object of interest from the original image
			human_img, obj_img = tools.crop_pair(pair[0][0], pair[1][0], os.path.join(self.img_folder_path, img), self.img_size)
			# Create the interaction pattern for the proposal crops:
			pair_img = tools.create_interaction_pattern(pair[0][0], pair[1][0], self.img_size)
			# Get the name of the object in question.
			obj_name = pair[1][2]

			gt_hois = annots.hoi_list
			confirmed_hoi_list = []

			# Loop though the list of ground truth hois, 
			for gt in gt_hois:
				# if a hoi contains the object in the current proposal
				if gt.obj == obj_name:
					# Loop through the ground truth connections
					for conn in gt.connections:
						# Get the index for the human and object in the GT bounding box list
						o_idx = conn[1]
						h_idx = conn[0]
						# Compute the current connection's human and object IOUs with the proposal
						iou_h = tools.compute_iou(pair[0][0], gt.human_boxes[h_idx-1])
						iou_o = tools.compute_iou(pair[1][0], gt.object_boxes[o_idx-1])
						# If we are above 0.5 we can add this hoi to our list of positive HOIs
						if min(iou_o, iou_h) > 0.5:
							confirmed_hoi_list.append(gt.hoi_id.astype(np.int32))
						else:
							confirmed_hoi_list.append(-1)

			img_crops.append([human_img, obj_img, pair_img])
			img_labels.append(tools.build_gt_vec(confirmed_hoi_list).astype(np.int32))
			#img_labels.append(confirmed_hoi_list)

		return img_crops, img_labels
'''
