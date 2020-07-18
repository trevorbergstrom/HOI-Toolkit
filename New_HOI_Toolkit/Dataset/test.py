from data_loader_test import HICODET_test
from data_loader_train import HICODET_train
import convert_hico_mat as tools
import torch
import pickle
import numpy as np

def pickle_proposals(props, file_name):
	with open(file_name, 'wb') as f:
		pickle.dump(props, f)


#bbox_mat = tools.load_mat('images/anno_bbox.mat')


test_data = HICODET_test('images/smallTest', bbox_matlist='images/anno_bbox.mat', props_file='images/pkl_files/miniTest.pkl')
test_data_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=1, shuffle=False)
#train_data = HICODET_train('images/train2015', bbox_mat, props_file='images/pkl_files/fullTrain.pkl')
#train_data = HICODET_train('images/train2015', bbox_mat, props_file='images/pkl_files/fullTrain.pkl', props_list = 'images/pkl_files/fullTrain_proposals.pkl')
#train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=8, shuffle=False)

print(test_data.no_interaction_idxs)
i = 0

for h, o, p, out in test_data_loader:
	i += 1
	#print(h.shape)
	#print(o.shape)
	#print(p.shape)
	
	for j in out:
		x = j[0].numpy()

		print(np.nonzero(x))

