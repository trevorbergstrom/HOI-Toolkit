'''
==============================FRCNN_Detector Class =====================================
This is the image detector. It will loop though all the images in the list and
create human and object proposals for each image. This should be computed ahead of time
in training, to reduce the GPU memory footprint.

Uses the pretrained Faster-RCNN model from TorchVision, with the resnet50 backbone.

Proposals are listed as follows:
    [
      [
        [[bbox_h], conf_score], [[bbox_o], conf_score]], img_path]
        ...
        ...
      ]
    ]
========================================================================================
'''
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np

class FRCNN_Detector():

    def __init__(self):
        # Calling TorchVision model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # Downloads model if needed
        self.model.eval()
        # Move to GPU:
        self.model.cuda()
        # Plain English Category names (not needed but good to have)
        self.COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire_hydrant', 'N/A', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball',
        'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
        'bottle', 'N/A', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'N/A', 'dining_table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
        ]

    # Function to predict a single image
    def get_predictions(self, img_path, threshold):
        img = Image.open(img_path) # Load the image
        transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
        img = transform(img).cuda() # Apply the transform to the image
        pred = self.model([img]) # Pass the image to the model

        return (pred)

    # Fucntion to get a list of proposals for a set of images
    def get_data_preds(self, imgs, root_dir, proposal_count):
        set_prop_list = []
        total_imgs = len(imgs)
        img_idx = 0

        for i in imgs:
            print('FRCNN Proposal Computation for Image# ' + str(img_idx+1) + '/' + str(total_imgs))
            img_idx = img_idx+1
            # Compute proposals
            x = self.get_predictions(os.path.join(root_dir, i), 0.001)
            img_proplist = []
            objs = []
            humans = []
            idx = 0

            # Converting Detection Results from Tensors to numpy
            bboxes = list(x[0]['boxes'].cpu().detach().numpy())
            scores = list(x[0]['scores'].cpu().detach().numpy())

            # Looping through detections and separate humans and other objects:
            for j in list(x[0]['labels'].cpu().detach().numpy()):
                if j == 1:
                    humans.append([bboxes[idx], scores[idx], j])
                else:
                    objs.append([bboxes[idx], scores[idx], self.COCO_INSTANCE_CATEGORY_NAMES[j]])
                idx = idx+1

            # only get top 10 humans and objects: (note: looks like the ordering of the predictions are in highest confidence to lowest)
            if len(humans) > 10:
                humans = humans[0:10]
            if len(objs) > 10:
                objs = objs[0:10]

            # Create proposals with each detected human, convert to int coordinates:
            for human in humans:
                for obj in objs:
                    prop = []
                    human[0] = human[0].astype(int)
                    prop.append(human)
                    obj[0] = obj[0].astype(int)
                    prop.append(obj)
                    img_proplist.append(prop)

            # If there are less than 8 proposals per image need to pad for batch size:
            '''
            while len(img_proplist) < proposal_count:
                img_proplist.append([[np.zeros(4, dtype=int), 0., 'null'], [np.zeros(4, dtype=int), 0., 'null']])
            '''
            # Add the img path to the list
            #img_proplist.append(i)
            set_prop_list.append(img_proplist)

        return set_prop_list

    def __del__(self):
    	del(self.model)
    	torch.cuda.empty_cache()

def sort_scores(prop_list):
    for i in range(len(prop_list)):
        for j in range(len(prop_list) - i - 1):
            if prop_list[j][1] < prop_list[j+1][1]:
                swp = prop_list[j]
                prop_list[j] = prop_list[j+1]
                prop_list[j+1] = swp


'''
===================== Testing stuff don't uncomment! ==============================
det = FRCNN_Detector()
i = det.get_data_preds(['img.jpg', 'img2.jpg'], '.')
print(len(i))
#for j in i:
#    for k in j:
#        print(k)
'''
#    print()
