from data_loader import HICODET_test, HICODET_train
import convert_hico_mat as tools
import torch

bbox_mat = tools.load_mat('images/anno_bbox.mat')

train_data = HICODET_train('images/smallTrain', bbox_mat)
test_data = HICODET_test('images/smallTest', bbox_mat)

train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=8, shuffle=True)
#train_data[0]

