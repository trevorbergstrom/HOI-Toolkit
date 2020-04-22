from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset
import torch
#from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os

'''
This is the HICO DET dataloader class
It requires paths to the image folders and the path to the annotations file!

How the structures look:
Img_list:
    [ [img1_name, w, h, d, [im1_hoi_list] ],
      [img2_name, w, h, d, [im2_hoi_list] ],
      [...................................] ]

HOI_List:
    [  [  [im1_hoi1_id, inv, [ [bbox_h1], [bboxh2], ... ] , [ [bboxo1], [bboxo2], ... ], [ [conn1], [conn2], ... ] obj_name],
          [im1_hoi2_id, inv, [ [bbox_h1], [bboxh2], ... ] , [ [bboxo1], [bboxo2], ... ], [ [conn1], [conn2], ... ] obj_name],
          [........................................................................................................]  ]
       [  [im2_hoi1_id, inv, [ [bbox_h1], [bboxh2], ... ] , [ [bboxo1], [bboxo2], ... ], [ [conn1], [conn2], ... ] obj_name],
          [im2_hoi2_id, inv, [ [bbox_h1], [bboxh2], ... ] , [ [bboxo1], [bboxo2], ... ], [ [conn1], [conn2], ... ] obj_name],
          [........................................................................................................]  ]  ]

Bbox_list:
    [ [bb1_x1, bb1_x2, bb1_y1, bb1_y2], [bb2_x1, bb2_x2, bb2_y1, bb2_y2], [bb3_x1, bb3_x2, bb3_y1, bb3_y2] ]

Conn_list:
    [ [conn1_h, conn1_o], [conn2_h, conn2_o] ]

'''

class HICO_DET_Dataloader(Dataset):

    def convert_bb(self, data_split, data_list, action_list):

        img_list = []
        itr = 0

        # For each image in the split:
        for i in data_list[0]:

            # Get the image name
            img_name = i[0][0]

            # Get the image properties:
            width = i[1][0][0][0][0][0]
            height = i[1][0][0][1][0][0]
            depth = i[1][0][0][2][0][0]

            # Append to list of props for current image:
            img_line = []
            img_line.append(img_name)
            img_line.append(width)
            img_line.append(height)
            img_line.append(depth)

            # Add image properties to the image list:
            img_list.append(img_line)
            # Get the number of HOIs :
            num_hoi = i[2].size

            hoi_list = []

            # For each hoi:
            for l in range(num_hoi):

                # Get critical properties of the HOI:
                invisible = i[2][0][l][4][0][0]
                hoi_id = i[2][0][l][0][0][0]

                # Create a line container to hold properties of the HOI:
                hoi_inst = []
                hoi_inst.append(hoi_id)
                hoi_inst.append(invisible)
                act_obj = action_list[hoi_id-1][0][0][0]

                # Create containers for each:
                bb_h_list = []
                bb_o_list = []
                conn_list = []

                # If a HOI is invisible, no HOIs properties will be listed here:
                if not invisible:

                    # For all human bounding boxes:
                    for m in range(int(i[2][0][l][1].size)):
                        bb_h_line = []

                        # human bbox coords:
                        x1_h = i[2][0][l][1][0][m][0][0][0]
                        x2_h = i[2][0][l][1][0][m][1][0][0]
                        y1_h = i[2][0][l][1][0][m][2][0][0]
                        y2_h = i[2][0][l][1][0][m][3][0][0]

                        bb_h_line.append(x1_h)
                        bb_h_line.append(x2_h)
                        bb_h_line.append(y1_h)
                        bb_h_line.append(y2_h)

                        bb_h_list.append(bb_h_line)

                    # For all object bboxes
                    for n in range(i[2][0][l][2].size):
                        bb_o_line = []

                        # object bbox coords:
                        x1_o = i[2][0][l][2][0][n][0][0][0]
                        x2_o = i[2][0][l][2][0][n][1][0][0]
                        y1_o = i[2][0][l][2][0][n][2][0][0]
                        y2_o = i[2][0][l][2][0][n][3][0][0]

                        bb_o_line.append(x1_o)
                        bb_o_line.append(x2_o)
                        bb_o_line.append(y1_o)
                        bb_o_line.append(y2_o)

                        bb_o_list.append(bb_o_line)

                    # For all connections
                    for p in range(int(i[2][0][l][3].size / 2)): #screwy indexing here for some reason
                        conn_line = []
                        connection = i[2][0][l][3][p]
                        conn_line.append(connection[0])
                        conn_line.append(connection[1])

                        conn_list.append(conn_line)

                # Lists for hoi_instance
                hoi_inst.append(bb_h_list)
                hoi_inst.append(bb_o_list)
                hoi_inst.append(conn_list)
                hoi_inst.append(act_obj)

                # Adding entry to HOI file for image
                hoi_list.append(hoi_inst)

            # Writing Image Index file:
            img_line.append(hoi_list)

        img_list.append(img_line)
        return img_list

    def get_action_list(self, actions):
        list_hois = []
        for i in range(len(actions)):
            obj = actions[i][0][0][0]
            interaction = actions[i][0][1][0]

            list_hois.append([i, obj, interaction])

        return list_hois

    def __load_mat__(self, matfile):

        # anno_bb.mat file in dict:
        bbs = loadmat(matfile)
        return bbs

    def __init__(self, train_data_path, test_data_path, matfile):
        print('Loading Dataset...')
        bbs = self.__load_mat__(matfile)
        # Training Bounding Boxes:
        bb_train = bbs['bbox_train']
        # Testing Bounding Boexs:
        bb_test = bbs['bbox_test']
        # Actions:
        actions = bbs['list_action']

        self.hoi_names = self.get_action_list(actions)

        self.img_train = self.convert_bb('train', bb_train, actions)
        self.img_names_train = [img[0] for img in self.img_train[:-1]]
        self.num_train = len(self.img_train)

        self.img_test = self.convert_bb('test', bb_test, actions)
        self.img_names_test = [img[0] for img in self.img_test[:-1]]
        self.num_test = len(self.img_test)

        self.test_data_path = test_data_path
        self.train_data_path = test_data_path
        print('Done Loading Dataset...')

    def __len__(self):
        return len(self.img_train) + len(self.img_test)

    def get_image(self, name, split):
        img = 0
        if split == 'train':
            img = Image.open(os.path.join(self.train_data_path, name))
        else:
            img = Image.open(os.path.join(self.test_data_path, name))

        return img

    def __getitem__(self, idx, split):
        img = 0
        img_anno = 0

        if split == 'test':
            img_anno = self.img_test[idx]
            img = self.get_image(img_anno[0], split) # Here open the image and then load it into a torchtensor??
        else:
            img_anno = self.img_train[idx]
            img = self.get_image(img_anno[0], split) # here open again

        if self.transform is not None:
            img = self.transform(img)

        sample = {'X': img,
                  'Y': img_anno}

        return sample

    def __get_crop__(self, img_name, bbox, data_path):
        img = Image.open(os.path.join(data_path, name))
        img = img.crop((bbox[0], bbox[2], bbox[1], bbox[3]))
        return img

    def __get_object_crop__(self, idx, split):

        if split == 'test':
            img_anno = self.img_test[idx]
            img = self.get_image(img_anno[0], split) # Here open the image and then load it into a torchtensor??
        else:
            img_anno = self.img_train[idx]
            img = self.get_image(img_anno[0], split) # here open again

        if img_anno[4][0][1] == 0:
            bbox = img_anno[4][0][3][0]
            obj = img.crop(( bbox[0], bbox[2], bbox[1], bbox[3] ))
            return obj, bbox
        else:
            return 0

    def __get_human_crop__(self, idx, split):

        if split == 'test':
            img_anno = self.img_test[idx]
            img = self.get_image(img_anno[0], split) # Here open the image and then load it into a torchtensor??
        else:
            img_anno = self.img_train[idx]
            img = self.get_image(img_anno[0], split) # here open again

        if img_anno[4][0][1] == 0:
            bbox_h = img_anno[4][0][2][0]
            human = img.crop(( bbox_h[0], bbox_h[2], bbox_h[1], bbox_h[3] ))
            return human, bbox_h
        else:
            return 0

    def __get_img_dims__(self, idx, split):

        if split == 'test':
            img_anno = self.img_test[idx]
            return img_anno[1], img_anno[2]
        else:
            img_anno = self.img_train[idx]
            return img_anno[1], img_anno[2]

    def __get_output__(self, idx, split):
        outs = np.zeros(600)

        if split =='test':
            hoi_id = self.img_test[idx][4][0][0]
        else:
            hoi_id = self.img_train[idx][4][0][0]

        print(hoi_id)
        outs[hoi_id] = 1.0

        return outs


def get_interaction_pattern(w, h, bbox_h, bbox_o):
    ip_x1 = min(bbox_h[0], bbox_o[0])
    ip_x2 = max(bbox_h[1], bbox_o[1])
    ip_y1 = min(bbox_h[2], bbox_o[2])
    ip_y2 = max(bbox_h[3], bbox_o[3])

    print('x1:' + str(ip_x1) + ' x2:' + str(ip_x2) + 'y1:' + str(ip_y1) + ' y2:' + str(ip_y2))

    human_channel = Image.new('1', (w,h), color=0)
    human_channel = human_channel.crop((ip_x1, ip_y1, ip_x2, ip_y2))
    img_h = ImageDraw.Draw(human_channel)
    img_h.rectangle([bbox_h[0], bbox_h[2], bbox_h[1], bbox_h[3]], fill=1)
    #human_channel.show()
    human_channel = human_channel.resize((256,256))
    human_channel = np.asarray(human_channel)
    human_channel = torch.from_numpy(human_channel)

    object_channel = Image.new('1', (w,h), color=0)
    object_channel = object_channel.crop((ip_x1, ip_y1, ip_x2, ip_y2))
    img_o = ImageDraw.Draw(object_channel)
    img_o.rectangle([bbox_o[0], bbox_o[2], bbox_o[1], bbox_o[3]], fill=1)
    #object_channel.show()
    object_channel = object_channel.resize((256,256))
    object_channel = np.asarray(object_channel)
    object_channel = torch.from_numpy(object_channel)

    return torch.stack([human_channel, object_channel], dim=0)
'''
#Testing stuff delete later
data = HICO_DET_Dataloader('~/Documents/hico/images/train2015', 'test2015','../images/anno_bbox.mat')
print(data.hoi_names)

#for i in data.img_names_train:
#    print(i)
#print(data.img_names_test)
#data.__get_human_crop__(1, 'test').show()
#data.__get_object_crop__(1, 'test').show()

#NEED TO FIX: image paths...
#ip = get_interaction_pattern(640,480, [100,200,100,500], [300,500,300,400])
#print(ip.shape)
'''
