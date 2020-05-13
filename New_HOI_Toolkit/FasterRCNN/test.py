from faster_RCNN_detector import FRCNN_Detector

detect = FRCNN_Detector()

preds = detect.get_data_preds(['wtf.jpg'], '.', 10)


for i in preds[0]:
    print(i)
    print('\n')

