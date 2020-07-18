from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
import sys
import tools as tools
sys.path.append('../Faster_RCNN')
from faster_RCNN_detector import FRCNN_Detector
from random import choices, sample
import pickle
from collections import namedtuple
import random

img_proposal_set = namedtuple('img_proposal_set', ['img_pth', 'positives', 't1_negatives', 't2_negatives'])
test_prop = namedtuple('test_prop', ['img_pth', 'bbox_h', 'bbox_o', 'hoi_id'])

class HICODET_test(Dataset):
    def __init__(self, folder_path, bbox_matfile, img_size=256, props_file='none'):

        # Load the annotations from the supplied matfile
        print('Loading annotations from file...')
        bbox_matlist = loadmat(bbox_matfile)
        self.img_size = img_size
        
        self.img_folder_path = folder_path
        self.annotations = tools.convert_bbox_matlist(bbox_matlist['bbox_test'], bbox_matlist['list_action'])
        self.hoi_class_list, self.no_interaction_classes = tools.convert_actions(bbox_matlist['list_action'])
        self.img_names = [img[0] for img in self.annotations] # <--- Change size for larger set
        #self.img_names = [img[0] for img in self.annotations[:168]] # <--- Change size for larger set
        print('Done')

        if not os.path.exists(props_file):
            detector = FRCNN_Detector()
            print('Test Set Generating Proposals with Detector')
            self.detection_proposals = detector.get_data_preds(self.img_names, folder_path)
            tools.pickle_proposals(self.detection_proposals, props_file)
            print('Done')
            # We dont need FRCNN to hangout and clog GPU memory after generating proposals. 
            del(detector)
        else:
            print('Loading Precomputed Detection Proposals From Files')
            with open(props_file, 'rb') as f:
                self.detection_proposals = pickle.load(f)
            print('Done')

        # Have proposals which are a list of bounding boxes.

    def __len__(self):
        return len(self.img_names)


    def get_img_props(self, detection_proposals, annotations):
        img_name = detection_proposals[0]
        det_bb_pairs = detection_proposals[1]

        proposal_list = []

        # Iterate through each pair in the detected objects
        for pair in det_bb_pairs:
            object_name = pair.obj.object_name.replace(' ','_')
            gt_hois = annotations.hoi_list 
            confirmed_hoi_classes = []

            #Iterate through the the list of groundtruths
            for gt in gt_hois:
                if gt.obj == object_name:
                    # Iterate through the connection list in the ground truth annotation
                    for connection in gt.connections:
                        o_idx = connection[1]-1
                        h_idx = connection[0]-1

                        iou_h = tools.compute_iou(pair.human.bbox, gt.human_boxes[h_idx])
                        iou_o = tools.compute_iou(pair.obj.bbox, gt.object_boxes[o_idx])
                        min_iou = min(iou_o, iou_h)
                        if min_iou >= 0.5:
                            #print('Confirmed ' + str(gt.hoi_id.astype(np.int32)))
                            confirmed_hoi_classes.append(gt.hoi_id.astype(np.int32))
            
            # If no groundthruth for the pair is found, the pair must be a no interaction class:
            if confirmed_hoi_classes:

                # Append the human & object bboxes and the list of confirmed hoi_classes
                proposal_list.append([pair.human.bbox, pair.obj.bbox, confirmed_hoi_classes])

        return proposal_list

    def __getitem__(self,idx):
        print(self.img_names[idx])
        proposals = self.get_img_props(self.detection_proposals[idx], self.annotations[idx]) 

        final_list = []

        for proposal in proposals:
            human_crop, object_crop = tools.crop_pair(proposal[0], proposal[1], os.path.join(self.img_folder_path, self.img_names[idx]), self.img_size)
            interation_pattern = tools.create_interaction_pattern(proposal[0], proposal[1], self.img_size)

            gt_vec = tools.build_gt_vec(proposal[2])

            final_list.append([human_crop, object_crop, interation_pattern, gt_vec])

        return final_list


    def search_classes(self, parent_folder):
        img_list = []
        count = []
        for i in range(600):
            img_list.append([])
            count.append(0)

        for img in self.annotations:
            hois = img.hoi_list

            for hoi in hois:
                img_list[hoi.hoi_id-1].append(img.path)
                count[hoi.hoi_id-1] += 1

        base_path = 'class'
        itr = 0
        for klass in img_list:
            itr += 1
            pth = 'class' + str(itr) + '.txt'
            pth = os.path.join(parent_folder, pth)

            with open(pth, 'w') as f:
                f.write('NUMBER: %d\n' % count[itr-1])
                for k in klass:
                    f.write('%s\n' % k)

