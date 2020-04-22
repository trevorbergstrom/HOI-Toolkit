import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import os
import random

sys.path.append('../../Faster_RCNN')
from frcnn import FRCNN_Detector
from dataset_load import HICO_DET_Dataloader, get_interaction_pattern

def compute_iou(bbox_prop, bbox_truth):
    # bbox = x1, x2, y1, y2

    x1 = max(bbox_prop[0], bbox_truth[0])
    x2 = max(bbox_prop[1], bbox_truth[1])
    y1 = max(bbox_prop[2], bbox_truth[2])
    y2 = max(bbox_prop[3], bbox_truth[3])

    area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    area_prop = (bbox_prop[1] - bbox_prop[0] + 1) * (bbox_prop[3] - bbox_prop[2] + 1)
    area_truth = (bbox_truth[1] - bbox_truth[0] + 1) * (bbox_truth[3] - bbox_truth[2] + 1)

    return (area / float(area_prop + area_truth - area))

'''
The purpose for this intermediate layer between the data_loader and the model is to create minibatches and manage the shuffling of data. The minibatch should contain 64 samples of hoi proposals from randomly sampled images.
'''
class Batch_Loader(Dataset):

    def __init__(self, batch_size, frcnn_threshold, data_path):
        # Num proposals in each batch returned
        self.batch_sz = batch_size

        # Path for data:
        self.data_path = data_path

        # Create a dataloader to load annotations
        self.Data_loader = HICO_DET_Dataloader(os.path.join(data_path, 'smallTrain'), os.path.join(data_path, 'smallTest'), os.path.join(data_path, 'anno_bbox.mat'))
        #self.Data_loader = HICO_DET_Dataloader(os.path.join(data_path, 'train2015'), os.path.join(data_path, 'test2015'), os.path.join(data_path, 'anno_bbox.mat'))

        # Empty list for overflow proposals
        self.proposal_list = []

        # Object detector initialize
        self.detector = FRCNN_Detector()

        #Precompute proposals for entire dataset
        self.__compute_proposals__()

        # List of indecies of images
        self.train_idx = list(range(self.Data_loader.num_train))
        self.test_idx = list(range(self.Data_loader.num_test))

        #current index in the list of images
        self.cur_idx_train = 0
        self.cur_idx_test = 0

    #def __score_ious__(self
    def __compute_proposals__(self):
        print('Using Detector to Pre-Compute Proposals: Training')
        self.training_proposals = self.detector.get_data_preds(self.Data_loader.img_names_train[0:49], os.path.join(self.data_path, 'smallTrain'))
        #self.training_proposals = self.detector.get_data_preds(self.Data_loader.img_names_train, os.path.join(self.data_path, 'train2015'))
        print('Done')
        print('Using Detector to Pre-Compute Proposals: Test')
        self.test_proposals = self.detector.get_data_preds(self.Data_loader.img_names_test[0:49], os.path.join(self.data_path, 'smallTest'))
        #self.test_proposals = self.detector.get_data_preds(self.Data_loader.img_names_test, os.path.join(self.data_path, 'test2015'))
        print('Done')

    # Function to return an 64 batch of proposals, 8 random images with 8 random proposals each
    def get_batch(self, split):
        samples = []
        if split == 'train':
            if self.cur_idx_train == self.Data_loader.num_train-1:
                random.shuffle(self.train_idx)
                self.cur_idx_train = 0

            props = self.training_proposals
            cidx = self.cur_idx_train
            self.cur_idx_train = self.cur_idx_train + 8
        else:
            if self.cur_idx_test == self.Data_loader.num_test-1:
                random.shuffle(self.test_idx)
                self.cur_idx_test = 0

            props = self.test_proposals
            cidx = self.cur_idx_test
            self.cur_idx_test = self.cur_idx_test + 8


        img_idxs = props[cidx:cidx+8]

        for idx in img_idxs:
            random.shuffle(idx)

            for i in range(8):
                samples.append(idx[i])

        return samples, img_idxs

    def get_ground_truth(self, split, img_idx_lst):
        # From the list of hois in the image, if any are containing the object in the proposal, then we need to add that hoi index to the gt vector.
        hoi_in_imgs = []
        # need to look up the hois and objects that they contain from the groundtruth information
        for i in img_idx_lst:
            hoi_in_imgs.aappend( self.Data_loader.hoi_props[i])
        # then we should loop through the proposals and find any objects that are in the  gt list of hois
        # those hois that have objects in the proposals are added to the list.
loader = Batch_Loader(64, 0.01, '../images')

samp = loader.get_batch('test')
print(samp)


