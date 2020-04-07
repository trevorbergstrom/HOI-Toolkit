from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

# anno_bb.mat file in dict:
bbs = loadmat('../anno_bbox.mat')
print(bbs.keys())


# Training Bounding Boxes:
bb_train = bbs['bbox_train']
# Testing Bounding Boexs:
bb_test = bbs['bbox_test']
# Actions:
actions = bbs['list_action']

'''
How the structures look:
Img_list:
    [ [img1_name, w, h, d, [im1_hoi_list] ],
      [img2_name, w, h, d, [im2_hoi_list] ],
      [...................................] ]

HOI_List:
    [  [  [im1_hoi1_id, inv, [ [bbox_h1], [bboxh2], ... ] , [ [bboxo1], [bboxo2], ... ], [ [conn1], [conn2], ... ] ],
          [im1_hoi2_id, inv, [ [bbox_h1], [bboxh2], ... ] , [ [bboxo1], [bboxo2], ... ], [ [conn1], [conn2], ... ] ],
          [........................................................................................................]  ]
       [  [im2_hoi1_id, inv, [ [bbox_h1], [bboxh2], ... ] , [ [bboxo1], [bboxo2], ... ], [ [conn1], [conn2], ... ] ],
          [im2_hoi2_id, inv, [ [bbox_h1], [bboxh2], ... ] , [ [bboxo1], [bboxo2], ... ], [ [conn1], [conn2], ... ] ],
          [........................................................................................................]  ]  ]

Bbox_list:
    [ [bb1_x1, bb1_x2, bb1_y1, bb1_y2], [bb2_x1, bb2_x2, bb2_y1, bb2_y2], [bb3_x1, bb3_x2, bb3_y1, bb3_y2] ]

Conn_list:
    [ [conn1_h, conn1_o], [conn2_h, conn2_o] ]

'''

def convert_bb(data_split, data_list):

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

            # Adding entry to HOI file for image
            hoi_list.append(hoi_inst)

        # Writing Image Index file:
        img_line.append(hoi_list)

    img_list.append(img_line)
    return img_list

class HICO_DET_Dataloader(Dataset):

    def __init__(self, train_csv_path, test_csv_path, train_data_path, test_data_path):
        self.img_train = convert_bb('train', bb_train)
        self.img_test = convert_bb('test', bb_test)
        self.test_data_path = test_data_path
        self.train_data_path = test_data_path

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
            img = get_image(idx, split) # Here open the image and then load it into a torchtensor??
            img_anno = self.img_test[idx]
        else:
            img = get_image(idx, split) # here open again
            img_anno = self.img_train[idx]

        if self.transform is not None:
            img = self.transform(img)

        sample = {'X': img,
                  'Y': img_anno}

        return sample

    def __get_human_crop__(self, idx, split):

        img = 0
        img_anno = 0

        if split == 'test':
            img = get_image(idx, split) # Here open the image and then load it into a torchtensor??
            img_anno = self.img_test[idx]
        else:
            img = get_image(idx, split) # here open again
            img_anno = self.img_train[idx]

        if img_anno == 0:
            bbox_h = img_anno[idx][2][0]
            human = img.crop(( bbox_h[0], bbox_h[2], bbox_h[1], bbox_h[3] ))
            return human
        else:
            return 0

data = HICO_DET_Dataloader()

