from test_dataloader import HICODET_test
from train_dataloader import HICODET_train
import tools as tools
import torch
import pickle
import numpy as np 

train_data = HICODET_train('images/train2015', 'images/anno_bbox.mat', props_file='images/pkl_files/FULLTrain.pkl')
train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=False)

test_data = HICODET_test('images/test2015', 'images/anno_bbox.mat', props_file='images/pkl_files/FULLTest.pkl')
test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

o = train_data[2517]

exit() 

classlist = []
for i in range(600):
	classlist.append(0)

itr = 0
for out in train_dataloader:
	itr += 1
	
	for i in out:
		x = i[3][0].numpy()
		n_zers = np.nonzero(x)[0]

		for j in n_zers:
			classlist[j] += 1
	
ir = 0
for i in classlist:
	print('class# ' + str(ir+1) + ':' + str(i))
	ir+=1

tools.pickle_proposals(classlist, 'classlist_train.pkl')




classlist = []
for i in range(600):
	classlist.append(0)

itr = 0
for out in test_dataloader:
	itr += 1
	
	for i in out:
		x = i[3][0].numpy()
		n_zers = np.nonzero(x)[0]

		for j in n_zers:
			classlist[j] += 1
	
ir = 0
for i in classlist:
	print('class# ' + str(ir+1) + ':' + str(i))
	ir+=1

tools.pickle_proposals(classlist, 'classlist_test.pkl')