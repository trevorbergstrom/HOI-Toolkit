from scipy.io import loadmat
import numpy as np
from os import mkdir
from shutil import rmtree

# anno_bb.mat file in dict:

bbs = loadmat('../anno_bbox.mat')
print(bbs.keys())


# Training Bounding Boxes:
bb_train = bbs['bbox_train']
# Testing Bounding Boexs:
bb_test = bbs['bbox_test']
# Actions:
actions = bbs['list_action']

# Create Directory for annotations
parent_dir = 'hico_det_anno/'
try:
    mkdir(parent_dir)
except FileExistsError:
    print('Parent Directory Exists. Removing now...')
    rmtree(parent_dir)
    mkdir(parent_dir)

def convert_bb(data_split, data_list):

    # Create directory for train/ test splits:
    split_folder = parent_dir + data_split + '_hoi/'
    mkdir(split_folder)

    img_list = []
    itr = 0

    # For each image in the split:
    for i in data_list[0]:

        # Get the image name
        img_name = i[0][0]

        # Create Directory for image annotations
        img_anno_folder = split_folder + img_name[:-4] + '/'
        mkdir(img_anno_folder)

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

        # Individual csv file for HOI list for current image:
        hoi_file_name = img_anno_folder + img_name[:-4] + '_HOI.csv'

        # Add HOI file name to image entry:
        img_line.append(hoi_file_name)

        # Add image properties to the image list:
        img_list.append(img_line)



        # Get the number of HOIs :
        num_hoi = i[2].size

        hoi_list = []

        # For each hoi:
        for l in range(num_hoi):

            # Create individual dir for each HOI:
            hoi_dir = img_anno_folder + '_hoi_' + str(l) + '_/'
            mkdir(hoi_dir)
            # Create individual csv files for bb_human, bb_obj, and connection:
            bb_h_file = hoi_dir + 'bb_h.csv'
            bb_o_file = hoi_dir + 'bb_o.csv'
            conn_file = hoi_dir + 'conn.csv'

            # Get critical properties of the HOI:
            invisible = i[2][0][l][4][0][0]
            hoi_id = i[2][0][l][0][0][0]

            # Create a line container to hold properties of the HOI:
            hoi_line = []
            hoi_line.append(hoi_id)
            hoi_line.append(invisible)
            hoi_line.append(bb_h_file)
            hoi_line.append(bb_o_file)
            hoi_line.append(conn_file)

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

            # Saving each bb list to a file
            bb_h_list = np.asarray(bb_h_list)
            bb_o_list = np.asarray(bb_o_list)
            conn_list = np.asarray(conn_list)
            np.savetxt(bb_h_file, bb_h_list, delimiter=',', fmt='%s')
            np.savetxt(bb_o_file, bb_o_list, delimiter=',', fmt='%s')
            np.savetxt(conn_file, conn_list, delimiter=',', fmt='%s')

            # Adding entry to HOI file for image
            hoi_list.append(hoi_line)

        # Writing HOI file for image:
        hoi_list = np.asarray(hoi_list)
        np.savetxt(hoi_file_name, hoi_list, delimiter=',', fmt='%s')
        print('Created file for: ' + img_name)

    # Writing Image Index file:
    img_list = np.asarray(img_list)
    mf_name = 'hico_det_anno/' + data_split + '_annos.csv'
    np.savetxt(mf_name, img_list, delimiter=',', fmt='%s')


convert_bb('train', bb_train)
convert_bb('test', bb_test)
