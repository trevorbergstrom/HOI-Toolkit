'''
This file contains code to convert the hico_det matfiles found with the dataset into lists of annotations
Abstracting this into another file makes the Data_Loader class look cleaner. 
'''
from scipy.io import loadmat
import numpy as np
from PIL import Image, ImageDraw
import os
import torch
from collections import namedtuple
import pickle

gt_image = namedtuple('gt_image', ['path', 'w', 'h', 'd', 'hoi_list'])
gt_hoi = namedtuple('gt_hoi', ['hoi_id', 'invis', 'obj', 'human_boxes', 'object_boxes', 'connections'])

''' 
=====================================================================================================================================
LoadMat function:

	loads the matfile to the matlist representation 
Input:
	needs the file path for the matfile
Returns:
	matlist format of all the data 
=====================================================================================================================================
'''
def load_mat(matfile_path):
	return loadmat(matfile_path)


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
    for i in range(len(action_mat_list)):
        obj = action_mat_list[i][0][0][0]
        interaction = action_mat_list[i][0][1][0]
        list_hois.append([i, obj, interaction])

    return list_hois

''' 
=====================================================================================================================================
multi_crop_img function:

	crops an image given the bounding box list. Avoids multiple calls to open each image 
Input:
	bbox_list: x1,x2,y1,y2 (hicodet format), img_path
Returns:
	list of PIL images that are the cropped image paths
=====================================================================================================================================
'''
def multi_crop_img(bbox_list, img_path, size):
	img = Image.open(img_path)
	if img.mode == 'L':
		img = img.convert('RGB')

	list_crops = []
	for box in bbox_list:
		if box[0][0] != 0 and box[0][1] != 0 and box[0][2] != 0 and box[0][3] != 0:
			
			human = img.crop((box[0][0], box[0][2], box[0][1], box[0][3])).resize((size,size))
			obj = img.crop((box[1][0], box[1][2], box[1][1], box[1][3])).resize((size,size))
		
			try:
				human = np.asarray(human).transpose(-1,0,1)
			except ValueError:
				print('Error detect: ')
				print(img_path)
				print(img.mode)
				print(img.getbands())
				exit()
			obj = np.asarray(obj).transpose(-1,0,1)

			human = torch.from_numpy(human).type(torch.FloatTensor)
			obj = torch.from_numpy(obj).type(torch.FloatTensor)
			list_crops.append([human,obj])

	img.close()


	return list_crops

def crop_pair(bbox_human, bbox_object, img_path, size):
    img = Image.open(img_path)
    if img.mode == 'L':
        img=img.convert('RGB')

    human = img.crop((bbox_human[0], bbox_human[1], bbox_human[2], bbox_human[3])).resize((size,size))
    obj = img.crop((bbox_object[0], bbox_object[1], bbox_object[2], bbox_object[3])).resize((size,size))

    return np.asarray(human).transpose(-1,0,1), np.asarray(obj).transpose(-1,0,1)
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
'''
def create_interaction_pattern(w, h, bbox_h, bbox_o, size):
	
	if bbox_h[0] == 0 and bbox_h[1] == 0 and bbox_h[2] == 0 and bbox_h[3] == 0:
		return 0 #$torch.zeros(1) #<------ Previously set to 1

	else:

		ip_x1 = min(bbox_h[0], bbox_o[0])
		ip_x2 = max(bbox_h[1], bbox_o[1])
		ip_y1 = min(bbox_h[2], bbox_o[2])
		ip_y2 = max(bbox_h[3], bbox_o[3])

		#print('x1:' + str(ip_x1) + ' x2:' + str(ip_x2) + 'y1:' + str(ip_y1) + ' y2:' + str(ip_y2))

		human_channel = Image.new('1', (w,h), color=0)
		human_channel = human_channel.crop((ip_x1, ip_y1, ip_x2, ip_y2))
		img_h = ImageDraw.Draw(human_channel)
		img_h.rectangle([bbox_h[0], bbox_h[2], bbox_h[1], bbox_h[3]], fill=1)
		#human_channel.show()
		human_channel = human_channel.resize((size,size))
		human_channel = np.asarray(human_channel)
		human_channel = torch.from_numpy(human_channel).type(torch.FloatTensor)

		object_channel = Image.new('1', (w,h), color=0)
		object_channel = object_channel.crop((ip_x1, ip_y1, ip_x2, ip_y2))
		img_o = ImageDraw.Draw(object_channel)
		img_o.rectangle([bbox_o[0], bbox_o[2], bbox_o[1], bbox_o[3]], fill=1)
		#object_channel.show()
		object_channel = object_channel.resize((size,size))
		object_channel = np.asarray(object_channel)
		object_channel = torch.from_numpy(object_channel).type(torch.FloatTensor)
	
		return torch.stack([human_channel, object_channel], dim=0)
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

def remove_dupes(a_list):
	final_list = []
	for i in a_list:
		if i not in final_list:
			final_list.append(i)

	return final_list

def build_gt_vec(img_hoi_list):
	classes = np.zeros(600)

	for i in img_hoi_list:
		classes[i-1] = 1.0

	#classes = torch.from_numpy(classes).type(torch.long)
	
	return classes
'''
def compute_iou(bbox_prop, bbox_truth):
    # bbox = x1, x2, y1, y2
    print(bbox_prop)
    print(bbox_truth)
    x1 = max(bbox_prop[0], bbox_truth[0])
    x2 = min(bbox_prop[2], bbox_truth[1])
    y1 = max(bbox_prop[1], bbox_truth[2])
    y2 = min(bbox_prop[3], bbox_truth[3])
    print(x1)
    print(x2)
    print(y1)
    print(y2)

    area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    area_prop = (bbox_prop[1] - bbox_prop[0] + 1) * (bbox_prop[3] - bbox_prop[2] + 1)
    area_truth = (bbox_truth[1] - bbox_truth[0] + 1) * (bbox_truth[3] - bbox_truth[2] + 1)

    return (float(area) / float(area_prop + area_truth - area))
'''

def compute_iou(bbox_prop, bbox_truth):
    print(bbox_prop)
    print(bbox_truth)

    bbox_prop = bbox_prop.tolist()
    x1 = max(bbox_prop[0], bbox_truth[0])
    x2 = min(bbox_prop[2], bbox_truth[1])
    y1 = max(bbox_prop[1], bbox_truth[2])
    y2 = min(bbox_prop[3], bbox_truth[3])

    area_intersection = (x2-x1) * (y2-y1)
    area_prop = ((bbox_prop[2] - bbox_prop[0]) * (bbox_prop[3] - bbox_prop[1])) 
    area_truth = ((bbox_truth[1] - bbox_truth[0]) * (bbox_truth[3] - bbox_truth[2]))
    area_union = (area_prop + area_truth) - area_intersection

    return float(area_intersection) / (float(area_union) + 0.001)

def pickle_proposals(props, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(props, f)