from faster_RCNN_detector import FRCNN_Detector

detect = FRCNN_Detector()

detect.detect_image('../Dataset/images/train2015/HICO_train2015_00000017.jpg', .8)

