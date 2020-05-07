from data_loader import HICODET_test, HICODET_train
import convert_hico_mat as tools
import torch
import pickle

def pickle_proposals(props, file_name):
	with open(file_name, 'wb') as f:
		pickle.dump(props, f)


bbox_mat = tools.load_mat('images/anno_bbox.mat')

#miou = tools.compute_iou([10,10,80,80], [30,60,30,60])
#print(miou)
#test_data = HICODET_test('images/test2015', bbox_mat, props_file='images/pkl_files/fullTest.pkl')
#test_data_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=1, shuffle=False)
#train_data = HICODET_train('images/train2015', bbox_mat, props_file='images/pkl_files/fullTrain.pkl')
train_data = HICODET_train('images/train2015', bbox_mat, props_file='images/pkl_files/fullTrain.pkl', props_list = 'images/pkl_files/fullTrain_proposals.pkl')
train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=8, shuffle=False)

#img_h, img_o, img_pair, outs = train_data[0]
#print(img_h)
#print(img_o)
#print(img_pair)
#print(outs)

for h, o, p, out in train_data_loader:
	print(h.shape)
	print(o.shape)
	print(p.shape)
	print(out.shape)
'''
train_data = HICODET_train('images/train2015', bbox_mat, props_file='images/pkl_files/fullTrain.pkl')

train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=1, shuffle=False)

for img_list, outputs in train_data_loader:
	print('Images len = ' + str(len(img_list)))
	print('labels shape = ' + str(outputs.shape))

#train_data[12355]
#train_data[12356]
'''
'''
train_data = HICODET_train('images/train2015', bbox_mat)
test_data = HICODET_test('images/test2015', bbox_mat)

#train_data = HICODET_train('images/smallTrain', bbox_mat, props_file='images/smallTrainProps.pkl')
#train_data = HICODET_train('images/smallTrain', bbox_mat)
#test_data = HICODET_test('images/smallTest', bbox_mat)

print('Save training proposals to pickle files:')
pickle_proposals(train_data.proposals, 'images/pkl_files/fullTrain.pkl')

print('Save test proposals to pickle files:')
pickle_proposals(test_data.proposals, 'images/pkl_files/fullTest.pkl')

train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=1, shuffle=False)

#train_data[0]
print(train_data.proposals)

for img_list, outputs in train_data_loader:
	print('Images len = ' + str(len(img_list)))
	print('labels shape = ' + str(outputs.shape))


test_data_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=1, shuffle=False)

for img_list, outputs in test_data_loader:
	print('Images len = ' + str(len(img_list)))
	print('labels shape = ' + str(outputs.shape))

'''