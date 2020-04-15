import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import os
import random

sys.path.append('../../Faster_RCNN')
from frcnn import FRCNN_Detector
from dataset_load import HICO_DET_Dataloader, get_interaction_pattern

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

    def __compute_proposals__(self):
        print('Using Detector to Pre-Compute Proposals: Training')
        self.training_proposals = self.detector.get_data_preds(self.Data_loader.img_names_train[0:49], os.path.join(self.data_path, 'smallTrain'))
        #self.training_proposals = self.detector.get_data_preds(self.Data_loader.img_names_train, os.path.join(self.data_path, 'train2015'))
        print('Done')
        print('Using Detector to Pre-Compute Proposals: Test')
        self.test_proposals = self.detector.get_data_preds(self.Data_loader.img_names_test[0:49], os.path.join(self.data_path, 'smallTest'))
        #self.test_proposals = self.detector.get_data_preds(self.Data_loader.img_names_test, os.path.join(self.data_path, 'test2015'))
        print('Done')


    def get_batch(self, split):
        samples = []
        if split == 'train':
            if cur_idx_train == self.Data_loader.num_train-1:
                random.shuffle(self.train_idx)
                cur_idx_train = 0

            props = self.training_proposals
            cidx = self.cur_idx_train
            self.cur_idx_train = self.cur_idx_train + 8
        else:
            if cur_idx_test == self.Data_loader.num_test-1:
                random.shuffle(self.test_idx)
                cur_idx_test = 0

            props = self.test_proposals
            cidx = self.cur_idx_test
            self.cur_idx_test = self.cur_idx_test + 8

        while(len(samples) < 64):
            img_idxs = props[cidx:cidx+8]

            for idx in img_idxs:
#UGHHHHHHHH NEED TO KEEP A SHUFFLE ORDER IN THE PROPOSALS..... IDK how to do this


loader = Batch_Loader(64, 0.01, '../images')
