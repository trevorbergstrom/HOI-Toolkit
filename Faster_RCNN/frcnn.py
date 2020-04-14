import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

class FRCNN_Detector():
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.cuda()
        self.COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def get_predictions(self, img_path, threshold):
        img = Image.open(img_path) # Load the image
        transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
        img = transform(img).cuda() # Apply the transform to the image
        pred = self.model([img]) # Pass the image to the model

        return (pred)
        '''
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return pred_boxes, pred_class
        '''
    def get_data_preds(self, imgs):
        predlist = []
        for i in imgs:
             predlist.append(self.get_predictions(i, 0.01))

        set_prop_list = []
        for i in predlist:
            img_proplist = []
            objs = []
            humans = []
            idx = 0
            bboxes = list(i[0]['boxes'].cpu().detach().numpy())
            scores = list(i[0]['scores'].cpu().detach().numpy())
            for j in list(i[0]['labels'].cpu().detach().numpy()):
                if j == 1:
                    humans.append([bboxes[idx], scores[idx]])
                else:
                    objs.append([bboxes[idx], scores[idx]])
                idx = idx+1

            for human in humans:
                prop = []
                prop.append(human)
                for obj in objs:
                    prop.append(obj)
                    img_proplist.append(prop)

            set_prop_list.append(img_proplist)

            # each human is a new proposal.
        return set_prop_list

det = FRCNN_Detector()
i = det.get_data_preds(['img.jpg'])
print(i)
