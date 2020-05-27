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
from collections import namedtuple
import random

img_proposal_set = namedtuple('img_proposal_set', ['img_pth', 'positives', 't1_negatives', 't2_negatives'])
test_prop = namedtuple('test_prop', ['img_pth', 'bbox_h', 'bbox_o', 'hoi_id'])
'''
This file contains the HICODET_Dataloader class
'''
class HICODET_test(Dataset):
	def __init__(self, folder_path, bbox_matlist, img_size=256, proposal_count=8, props_file='none'):
		
		bbox_matlist = loadmat(bbox_matlist)	
		self.img_size = img_size
		self.proposal_count = proposal_count

		self.img_folder_path = folder_path

		self.annotations = tools.convert_bbox_matlist(bbox_matlist['bbox_test'], bbox_matlist['list_action'])
		self.interaction_prop_list, self.no_interaction_idxs = tools.convert_actions(bbox_matlist['list_action'])
		self.img_names = [img[0] for img in self.annotations] # <--- Change size for larger set
		#self.img_names = [img[0] for img in self.annotations[:50]] # <--- Change size for larger set


		if props_file == 'none':
			detector = FRCNN_Detector()
			print('Test Set Generating Proposals with Detector')
			self.proposals = detector.get_data_preds(self.img_names, folder_path, proposal_count)
			tools.pickle_proposals(self.proposals, '../Dataset/images/pkl_files/full_test2015.pkl')
			print('Done')


			# We dont need FRCNN to hangout and clog GPU memory after generating proposals. 
			del(detector)
		else:
			print('Loading Precomputed Detection Proposals From Files')
			with open(props_file, 'rb') as f:
				self.proposals = pickle.load(f)

		self.good_props = self.compile_props()
		
		'''
		self.pos_props = []
		for i in range(len(self.img_names)):
			self.pos_props.extend(self.get_img_props(self.proposals[i], self.annotations[i], 0))
		'''

	def __len__(self):
		#return len(self.good_props)
		return len(self.img_names)

	def dataset_analysis(self):
		class_count = np.zeros(600)
		invisible_count = np.zeros(600)
		imgs_with_class = [ [] for i in range(600) ]

		for img in self.annotations:
			for hoi in img.hoi_list:
				if img.path not in imgs_with_class[hoi.hoi_id - 1]:
					imgs_with_class[hoi.hoi_id - 1].append(img.path)

				if hoi.invis == 1:
					invisible_count[hoi.hoi_id - 1] += 1
				else:
					class_count[hoi.hoi_id - 1] += 1

		print('+------------------ ANALYSIS RESULTS ---------------------+')
		print('Number of Images: ' + str(len(self)))
		for i in range(600):
			print('HOI_ID = {id} : interaction = {inter} : object = {obj} --- #visible = {vis} : #invisible = {invis} : #imgs = {num_imgs}'.format(id = i+1, inter=self.interaction_prop_list[i][2],
				obj=self.interaction_prop_list[i][1], vis=class_count[i], invis=invisible_count[i], num_imgs=len(imgs_with_class[i])))
			if i == 9:
				print(imgs_with_class[i])
		print('+---------------------------------------------------------+')

	def compile_props(self):
		good_props = []
		for i in range(len(self.annotations)):
			img_path = self.annotations[i].path
			#print(img_path)
			hoi_list = self.annotations[i].hoi_list

			humans = []
			objects = []
			hoi_ids = []

			for p in hoi_list:
				for c in p.connections:
					humans.append(p.human_boxes[c[0]-1])
					objects.append(p.object_boxes[c[1]-1])
					hoi_ids.append(p.hoi_id)

			# find distinct humans and objects
			unique_p = []
			n = len(humans)
			for j in range(len(humans)):
				h = humans[j]
				o = objects[j]
				idxs = []
				idxs.append(j)
				for k in range(len(humans)):
					if k != j:
						if min(tools.compute_iou2(h, humans[k]), tools.compute_iou2(o, objects[k])) > 0.5:
							idxs.append(k)
				unique_p.append(idxs)

			for p in unique_p:
				p.sort()

			up2 = []
			for p in unique_p:
				if p not in up2:
					up2.append(p)

			for p in up2:
				hois = []
				for idx in p:
					hois.append(hoi_ids[idx])
				gt_vector = tools.build_gt_vec(hois).astype(np.int32)

				h_box = np.array(humans[p[0]]).astype(np.int32)
				o_box = np.array(objects[p[0]]).astype(np.int32)
				good_props.append(test_prop(img_path, h_box, o_box, gt_vector))

		return good_props


	def get_img_props(self, det_props, annots, prop_number):
		img_name = det_props[0]
		t1_neg_set = []
		t2_neg_set = []
		pos_set = []
		if prop_number < 4 and prop_number > 1:
			prop_number=4

		for proposal in det_props[1]:
			obj_name = proposal[1][2].replace(' ','_')
			gt_hois = annots.hoi_list # Change this to be a dictionary
			confirmed_hoi_list = []

			t1 = False
			for gt in gt_hois:
				if gt.obj == obj_name:
					for conn in gt.connections:
						o_idx = conn[1]
						h_idx = conn[0]
						#Compute IOUs
						iou_h = tools.compute_iou(proposal[0][0], gt.human_boxes[h_idx-1])
						iou_o = tools.compute_iou(proposal[1][0], gt.object_boxes[o_idx-1])
						min_iou = min(iou_o, iou_h)

						if min_iou >= 0.5:
							confirmed_hoi_list.append(gt.hoi_id.astype(np.int32))
						elif min_iou > 0.1:
							t1 = True

			if len(confirmed_hoi_list) == 0:
				if t1 == True:
					confirmed_hoi_list.append(self.no_interaction_idxs[obj_name])
					gt_vector = tools.build_gt_vec(confirmed_hoi_list).astype(np.int32)
					t1_neg_set.append([img_name, proposal[0][0], proposal[1][0], gt_vector, proposal[0][1], proposal[1][1]])
				else:
					confirmed_hoi_list.append(self.no_interaction_idxs[obj_name])
					gt_vector = tools.build_gt_vec(confirmed_hoi_list).astype(np.int32)
					t2_neg_set.append([img_name, proposal[0][0], proposal[1][0], gt_vector, proposal[0][1], proposal[1][1]])
			else:
				gt_vector = tools.build_gt_vec(confirmed_hoi_list).astype(np.int32)
				pos_set.append([img_name, proposal[0][0], proposal[1][0], gt_vector, proposal[0][1], proposal[1][1]])

		
		
		# Now we choose a random selection from each 
		batch_prop_list = []
		if prop_number == 1:
			if len(pos_set) != 0:
				batch_prop_list.append(random.choice(pos_set))
			elif len(t2_neg_set) != 0:
				batch_prop_list.append(random.choice(t2_neg_set))
			elif len(t1_neg_set) != 0: 
				batch_prop_list.append(random.choice(t1_neg_set))
			else:
				index = random.choice(range(len(self)))
				batch_prop_list = self.get_img_props(self.proposals[index], self.annotations[index], self.proposal_count)
			return batch_prop_list

		n_pos = random.randrange(1,prop_number-1) # Random number between 1 and max props 

		if len(pos_set) < n_pos:
			for i in range(len(pos_set)):
				batch_prop_list.append(pos_set[i])
			n_pos = len(pos_set)
		else:
			picks = random.sample(pos_set,k=n_pos)
			for i in range(len(picks)):
				batch_prop_list.append(picks[i])


		n_t1 = random.randrange(1,prop_number-n_pos) # random number 
		if len(t1_neg_set) < n_t1:
			for i in range(len(t1_neg_set)):
				batch_prop_list.append(t1_neg_set[i])
			n_t1 = len(t1_neg_set)
		else:
			picks = random.sample(t1_neg_set, k=n_t1)
			for i in range(len(picks)):
				batch_prop_list.append(picks[i])

		n_t2 = prop_number-n_pos-n_t1
		if len(t2_neg_set) < n_t2:
			for i in range(len(t2_neg_set)):
				batch_prop_list.append(t2_neg_set[i])
		else:
			picks = random.sample(t2_neg_set, k=n_t2)
			for i in range(len(picks)):
				batch_prop_list.append(picks[i])

		while len(batch_prop_list) < prop_number:
			if len(t2_neg_set) != 0:
				batch_prop_list.append(random.choice(t2_neg_set))
			elif len(t1_neg_set) != 0:
				batch_prop_list.append(random.choice(t1_neg_set))
			elif len(pos_set) != 0: 
				batch_prop_list.append(random.choice(pos_set))
			else:
				index = random.choice(range(len(self)))
				batch_prop_list = self.get_img_props(self.proposals[index], self.annotations[index], self.proposal_count)
		
		
		return batch_prop_list
	
	def __getitem__(self,idx):
		props_list = self.get_img_props(self.proposals[idx], self.annotations[idx], self.proposal_count)

		human_list = []
		object_list = []
		pair_list = []
		label_list= []

		for proposal in props_list:
			human_crop, object_crop = tools.crop_pair(proposal[1], proposal[2], os.path.join(self.img_folder_path, proposal[0]), self.img_size)
			pair = tools.create_interaction_pattern(proposal[1], proposal[2], self.img_size)

			human_list.append(human_crop)
			object_list.append(object_crop)
			pair_list.append(pair)
			label_list.append(proposal[3])

		return human_list, object_list, pair_list, label_list
	'''

	def __getitem__(self, idx):
		p = self.good_props[idx]
		human_crop, object_crop = tools.crop_pair(p[1], p[2], os.path.join(self.img_folder_path, p[0]), self.img_size)
		pair = tools.create_interaction_pattern2(p[1], p[2], self.img_size)
		
		return human_crop, object_crop, pair, p[3]
	'''
	
#==========================================================================================================================================================================================
class HICODET_train(Dataset):
	def __init__(self, folder_path, bbox_matlist, img_size=256, proposal_count=8, props_file='none'):
		
		bbox_matlist = loadmat(bbox_matlist)	
		self.img_size = img_size
		self.proposal_count = proposal_count

		self.img_folder_path = folder_path

		self.annotations = tools.convert_bbox_matlist(bbox_matlist['bbox_train'], bbox_matlist['list_action'])
		self.interaction_prop_list, self.no_interaction_idxs = tools.convert_actions(bbox_matlist['list_action'])
		self.img_names = [img[0] for img in self.annotations] # <--- Change size for larger set
		#self.img_names = [img[0] for img in self.annotations[:50]] # <--- Change size for larger set


		if props_file == 'none':
			detector = FRCNN_Detector()
			print('Test Set Generating Proposals with Detector')
			self.proposals = detector.get_data_preds(self.img_names, folder_path, proposal_count)
			tools.pickle_proposals(self.proposals, '../Dataset/images/pkl_files/full_train2015.pkl')
			print('Done')


			# We dont need FRCNN to hangout and clog GPU memory after generating proposals. 
			del(detector)
		else:
			print('Loading Precomputed Detection Proposals From Files')
			with open(props_file, 'rb') as f:
				self.proposals = pickle.load(f)

	def __len__(self):
		return len(self.img_names)

	def dataset_analysis(self):
		class_count = np.zeros(600)
		invisible_count = np.zeros(600)
		imgs_with_class = [ [] for i in range(600) ]

		for img in self.annotations:
			for hoi in img.hoi_list:
				if img.path not in imgs_with_class[hoi.hoi_id - 1]:
					imgs_with_class[hoi.hoi_id - 1].append(img.path)

				if hoi.invis == 1:
					invisible_count[hoi.hoi_id - 1] += 1
				else:
					class_count[hoi.hoi_id - 1] += 1

		print('+------------------ ANALYSIS RESULTS ---------------------+')
		print('Number of Images: ' + str(len(self)))
		for i in range(600):
			print('HOI_ID = {id} : interaction = {inter} : object = {obj} --- #visible = {vis} : #invisible = {invis} : #imgs = {num_imgs}'.format(id = i+1, inter=self.interaction_prop_list[i][2],
				obj=self.interaction_prop_list[i][1], vis=class_count[i], invis=invisible_count[i], num_imgs=len(imgs_with_class[i])))
		print('+---------------------------------------------------------+')

	def get_img_props(self, det_props, annots, prop_number):
		img_name = det_props[0]
		t1_neg_set = []
		t2_neg_set = []
		pos_set = []
		if prop_number < 4:
			prop_number=4

		for proposal in det_props[1]:
			obj_name = proposal[1][2].replace(' ','_')
			gt_hois = annots.hoi_list # Change this to be a dictionary
			confirmed_hoi_list = []

			t1 = False
			for gt in gt_hois:
				if gt.obj == obj_name:
					for conn in gt.connections:
						o_idx = conn[1]
						h_idx = conn[0]
						#Compute IOUs
						iou_h = tools.compute_iou(proposal[0][0], gt.human_boxes[h_idx-1])
						iou_o = tools.compute_iou(proposal[1][0], gt.object_boxes[o_idx-1])
						min_iou = min(iou_o, iou_h)

						if min_iou >= 0.5:
							confirmed_hoi_list.append(gt.hoi_id.astype(np.int32))
						elif min_iou > 0.1:
							t1 = True

			if len(confirmed_hoi_list) == 0:
				if t1 == True:
					confirmed_hoi_list.append(self.no_interaction_idxs[obj_name])
					gt_vector = tools.build_gt_vec(confirmed_hoi_list).astype(np.int32)
					t1_neg_set.append([img_name, proposal[0][0], proposal[1][0], gt_vector, proposal[0][1], proposal[1][1]])
				else:
					confirmed_hoi_list.append(self.no_interaction_idxs[obj_name])
					gt_vector = tools.build_gt_vec(confirmed_hoi_list).astype(np.int32)
					t2_neg_set.append([img_name, proposal[0][0], proposal[1][0], gt_vector, proposal[0][1], proposal[1][1]])
			else:
				gt_vector = tools.build_gt_vec(confirmed_hoi_list).astype(np.int32)
				pos_set.append([img_name, proposal[0][0], proposal[1][0], gt_vector, proposal[0][1], proposal[1][1]])

		# Now we choose a random selection from each 
		batch_prop_list = []
		n_pos = random.randrange(1,prop_number-1) # Random number between 1 and max props 

		if len(pos_set) < n_pos:
			for i in range(len(pos_set)):
				batch_prop_list.append(pos_set[i])
			n_pos = len(pos_set)
		else:
			picks = random.sample(pos_set,k=n_pos)
			for i in range(len(picks)):
				batch_prop_list.append(picks[i])


		n_t1 = random.randrange(1,prop_number-n_pos) # random number 
		if len(t1_neg_set) < n_t1:
			for i in range(len(t1_neg_set)):
				batch_prop_list.append(t1_neg_set[i])
			n_t1 = len(t1_neg_set)
		else:
			picks = random.sample(t1_neg_set, k=n_t1)
			for i in range(len(picks)):
				batch_prop_list.append(picks[i])

		n_t2 = prop_number-n_pos-n_t1
		if len(t2_neg_set) < n_t2:
			for i in range(len(t2_neg_set)):
				batch_prop_list.append(t2_neg_set[i])
		else:
			picks = random.sample(t2_neg_set, k=n_t2)
			for i in range(len(picks)):
				batch_prop_list.append(picks[i])

		while len(batch_prop_list) < prop_number:
			if len(t2_neg_set) != 0:
				batch_prop_list.append(random.choice(t2_neg_set))
			elif len(t1_neg_set) != 0:
				batch_prop_list.append(random.choice(t1_neg_set))
			elif len(pos_set) != 0: 
				batch_prop_list.append(random.choice(pos_set))
			else:
				index = random.choice(range(len(self)))
				batch_prop_list = self.get_img_props(self.proposals[index], self.annotations[index], self.proposal_count)
				
		return batch_prop_list

	def __getitem__(self,idx):
		props_list = self.get_img_props(self.proposals[idx], self.annotations[idx], self.proposal_count)

		human_list = []
		object_list = []
		pair_list = []
		label_list= []

		for proposal in props_list:
			human_crop, object_crop = tools.crop_pair(proposal[1], proposal[2], os.path.join(self.img_folder_path, proposal[0]), self.img_size)
			pair = tools.create_interaction_pattern(proposal[1], proposal[2], self.img_size)

			human_list.append(human_crop)
			object_list.append(object_crop)
			pair_list.append(pair)
			label_list.append(proposal[3])

		return human_list, object_list, pair_list, label_list
