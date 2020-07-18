# This file contains dataset creation and data manipulation tools

from scipy.io import loadmat
import numpy as np
from PIL import Image, ImageDraw
import os
import torch
from collections import namedtuple
import pickle

gt_image = namedtuple('gt_image', ['path', 'w', 'h', 'd', 'hoi_list'])
gt_hoi = namedtuple('gt_hoi', ['hoi_id', 'invis', 'obj', 'human_boxes', 'object_boxes', 'connections'])
hoi_prop = namedtuple('hoi_prop', ['id', 'obj', 'interaction'])

''' 
=====================================================================================================================================
convert_actions function:
	converts the action_list to a better list representation 
Input:
	needs action lat_list
Returns:
	List of HOIs with the needed properties.  [ [hoi_id, object_name, interaction_name], ... ]
=====================================================================================================================================
'''
def convert_actions(action_mat_list):
    list_hois = []
    no_interactions = {}
    for i in range(len(action_mat_list)):
        obj = action_mat_list[i][0][0][0]
        interaction = action_mat_list[i][0][1][0]
        list_hois.append(hoi_prop(i, obj, interaction))

        if interaction == 'no_interaction':
            no_interactions[obj] = i+1 #Zero_indexed

    return list_hois, no_interactions

''' 
=====================================================================================================================================
crop_pair function:

    crops a pair of bounding boxes from the image
Input:
    bbox_human, bbox_object: x1,x2,y1,y2 (hicodet format), img_path, size of resized crop
Returns:
    two numpy arrays, transposed to torch tensor format
=====================================================================================================================================
'''
def crop_pair(bbox_human, bbox_object, img_path, size):
    img = Image.open(img_path)
    if img.mode == 'L':
        img=img.convert('RGB')

    human = img.crop((bbox_human[0], bbox_human[1], bbox_human[2], bbox_human[3])).resize((size,size))
    obj = img.crop((bbox_object[0], bbox_object[1], bbox_object[2], bbox_object[3])).resize((size,size))
    #human.show()
    #obj.show()
    return np.asarray(human).transpose(-1,0,1).astype(np.int32), np.asarray(obj).transpose(-1,0,1).astype(np.int32)

''' 
=====================================================================================================================================
create_interaction_pattern function:

	creates an interaction pattern from the human and object bounding boxes
Input:
	w = image width, h=image height, bbox_h: x1,x2,y1,y2 (hicodet format), bbox_o: x1,x2,y1,y2 (hicodet format), 
Returns:
	a 2 channel interaction pattern image in tensor format
=====================================================================================================================================
'''
def create_interaction_pattern(bbox_h, bbox_o, size):
    ip_x1 = min(bbox_h[0], bbox_o[0])
    ip_x2 = max(bbox_h[2], bbox_o[2])
    ip_y1 = min(bbox_h[1], bbox_o[1])
    ip_y2 = max(bbox_h[3], bbox_o[3])
    w = ip_x2-ip_x1
    h = ip_y2-ip_y1

    human_channel = Image.new('1', (w,h), color=0)
    human_channel_draw = ImageDraw.Draw(human_channel)
    human_channel_draw.rectangle([bbox_h[0], bbox_h[2], bbox_h[1], bbox_h[3]], fill=1)
    human_channel = human_channel.resize((size,size))
    human_channel = np.asarray(human_channel)
    human_channel = torch.from_numpy(human_channel).type(torch.FloatTensor)

    object_channel = Image.new('1', (w,h), color=0)
    object_channel_draw = ImageDraw.Draw(object_channel)
    object_channel_draw.rectangle([bbox_o[0], bbox_o[2], bbox_o[1], bbox_o[3]], fill=1)
    object_channel = object_channel.resize((size,size))
    object_channel = np.asarray(object_channel)
    object_channel = torch.from_numpy(object_channel).type(torch.FloatTensor)

    return torch.stack([human_channel, object_channel], dim=0)

''' 
=====================================================================================================================================
convert_bbox_matlist function:

	Converts from the bounding box matfile returned list to a more legible list.

Input:
	data_list the list of the data that needs to be cleaned up.
Return Format:

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


=====================================================================================================================================
'''

def convert_bbox_matlist(mat_list, action_list):

    # Empty list to store the object
    img_list = []
    itr = 0

    # For each image in the split:
    for i in mat_list[0]:

        # Get the image name
        img_name = i[0][0]
        
        # Get the image properties:
        width = i[1][0][0][0][0][0]
        height = i[1][0][0][1][0][0]
        depth = i[1][0][0][2][0][0]
        
        # Get the number of HOIs :
        num_hoi = i[2].size

        hoi_list = []

        # For each hoi:
        for l in range(num_hoi):

            # Get critical properties of the HOI:
            invisible = i[2][0][l][4][0][0]
            hoi_id = i[2][0][l][0][0][0]

            # Create a line container to hold properties of the HOI:
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

            cur_hoi = gt_hoi(hoi_id, invisible, act_obj, bb_h_list, bb_o_list, conn_list)
            # Adding entry to HOI file for image
            hoi_list.append(cur_hoi)

        # Writing Image Index file:
        img_list.append(gt_image(img_name, width, height, depth, hoi_list))

    return img_list

''' 
=====================================================================================================================================
compute_iou function:

	computes the Intersection Over Union between two bounding boxes.
Input:
	two HICO-DET style bounding box annotations
Returns:
	float, iou
=====================================================================================================================================
'''
def compute_iou(bbox_prop, bbox_truth):
    
    bbox_truth = np.array(bbox_truth).astype(np.int32)
    bbox_prop = np.array(bbox_prop).astype(np.int32)

    x1 = max(bbox_prop[0], bbox_truth[0])
    x2 = min(bbox_prop[2], bbox_truth[1])
    y1 = max(bbox_prop[1], bbox_truth[2])
    y2 = min(bbox_prop[3], bbox_truth[3])

    x_bound = x2-x1
    y_bound = y2-y1
    if x_bound <=0 or y_bound <= 0:
        return 0.0

    area_intersection = (x2-x1) * (y2-y1)
    area_prop = ((bbox_prop[2] - bbox_prop[0]) * (bbox_prop[3] - bbox_prop[1])) 
    area_truth = ((bbox_truth[1] - bbox_truth[0]) * (bbox_truth[3] - bbox_truth[2]))
    area_union = (area_prop + area_truth) - area_intersection

    return float(area_intersection) / (float(area_union) + 0.001)

''' 
=====================================================================================================================================
build_gt_vec function:

	builds the multilabel ground truth vector 
Input:
	list of ints that represent the classes
Returns:
	numpy array of labels
=====================================================================================================================================
'''
def build_gt_vec(img_hoi_list):
	classes = np.zeros(600)

	for i in img_hoi_list:
		classes[i-1] = 1.0
	
	return classes


def pickle_proposals(props, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(props, f)


def random_sampling(num_choices, l1, l2, l3):
	ret_sample = []

	r_num = sorted(random.sample(range(num_choices),2))
	l1_num = r_num[0]
	l2_num = r_num[1] - r_num[0]
	l3_num = num_choices - r_num[1]

	miss_val = 0

	if len(l1) < l1_num:
		miss_val += l1_num - len(l1)
		l1_num = len(l1)

	if len(l2) < l2_num:
		miss_val += l2_num - len(l2)
		l2_num = len(l2)

	if len(l3) < l3_num:
		miss_val += l3_num - len(l3)
		l3_num = len(l3)

	'''
	l1_choices
	if l1_num + l2_num + l3_num < num_choices:
	'''
	return 0

	
