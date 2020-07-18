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
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from collections import namedtuple

detected_object = namedtuple('detected_object', ['bbox', 'score', 'object_name'])
proposal_pair = namedtuple('proposal_pair', ['human', 'obj'])

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

    def detect_image(self, img_path, threshold, to='none'):
        print('DETECTION FOR: ' + img_path)
        preds = self.get_predictions(img_path, threshold)
        im = Image.open(img_path)
        #im.show()
        im_d = ImageDraw.Draw(im)
        bboxes = list(preds[0]['boxes'].cpu().detach().numpy())
        scores = list(preds[0]['scores'].cpu().detach().numpy())
        labels = list(preds[0]['labels'].cpu().detach().numpy())
        '''
        print('BOXES:')
        for i in bboxes:
            print(i)
        #print(bboxes)
        print('SCORES:')
        for i in scores:
            print(i)
        #print(scores)
        print([self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in labels])
        '''
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 10)
        for i in range(len(bboxes)):
            if to == 'none':
                if scores[i] > threshold:
                    label = self.COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
                    score = scores[i]
                    m_txt = 'Category: {c}, Score: {s:.2f}'
                    m_txt = m_txt.format(c=label, s=score)
                    print(m_txt)
                    print(bboxes[i])
                    im_d.rectangle([bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]], outline='blue', width=2)
                    #im_d.rectangle([bboxes[i][0], bboxes[i][3], bboxes[i][0]+200, bboxes[i][3]-20], fill='blue')
                    #im_d.text((bboxes[i][0]+5, bboxes[i][3]-15), m_txt, font=fnt, fill='white')
            else:
                if scores[i] > threshold and self.COCO_INSTANCE_CATEGORY_NAMES[labels[i]] == to:
                    label = self.COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
                    score = scores[i]
                    m_txt = 'Category: {c}, Score: {s:.2f}'
                    m_txt = m_txt.format(c=label, s=score)
                    print(m_txt)
                    print(bboxes[i])
                    im_d.rectangle([bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]], outline='blue', width=2)
                    #im_d.rectangle([bboxes[i][0], bboxes[i][3], bboxes[i][0]+200, bboxes[i][3]-20], fill='blue')
                    #im_d.text((bboxes[i][0]+5, bboxes[i][3]-15), m_txt, font=fnt, fill='white')
        im.show()
        print('----------------------------------------------------------')

    def label_image(self, path, x1, x2, y1, y2):
        im = Image.open(path)
        im_d = ImageDraw.Draw(im)
        im_d.rectangle([x1, y1, x2, y2], outline='red', width=2)
        im.show()
        
    # Fucntion to get a list of proposals for a set of images
    def get_data_preds(self, imgs, root_dir):
        set_prop_list = []
        total_imgs = len(imgs)
        img_idx = 0

        for i in imgs:
            print('FRCNN Proposal Computation for Image# ' + str(img_idx+1) + '/' + str(total_imgs))
            img_idx = img_idx+1
            # Compute proposals
            x = self.get_predictions(os.path.join(root_dir, i), 0.5)
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
                    humans.append(detected_object(bboxes[idx].astype(int), scores[idx], self.COCO_INSTANCE_CATEGORY_NAMES[j]))
                # Append all humans and objects to the list of objects
                objs.append(detected_object(bboxes[idx].astype(int), scores[idx], self.COCO_INSTANCE_CATEGORY_NAMES[j]))
                idx = idx+1

            # only get top 10 humans and objects: (note: looks like the ordering of the predictions are in highest confidence to lowest)
            if len(humans) > 10:
                humans = humans[0:10]
            if len(objs) > 10:
                objs = objs[0:10]

            # Create proposals with each detected human, convert to int coordinates:
            for human in humans:
                for obj in objs:
                    
                    #human.bbox = human.bbox.astype(int)
                    
                    #obj.bbox = obj.bbox.astype(int)
                    if not np.array_equal(human.bbox, obj.bbox):
                        img_proplist.append(proposal_pair(human, obj))

            set_prop_list.append([i,img_proplist])

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
