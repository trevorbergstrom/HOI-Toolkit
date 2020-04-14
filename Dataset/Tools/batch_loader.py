import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import os

sys.path.append('../../Faster_RCNN')
from frcnn import get_proposals
from dataset_load import HICO_DET_Dataloader, get_interaction_pattern

'''
The purpose for this intermediate layer between the data_loader and the model is to create minibatches and manage the shuffling of data. The minibatch should contain 64 samples of hoi proposals from randomly sampled images.
'''
class Batch_Loader(Dataset):

    def __init__(self, batch_size, frcnn_threshold, data_path):
        self.batch_sz = batch_size
        self.Data_loader = HICO_DET_Dataloader(os.path.join(data_path, 'train2015'), os.path.join(data_path, 'test2015'), os.path.join(data_path, 'anno_bbox.mat')
        self.proposal_list = []


