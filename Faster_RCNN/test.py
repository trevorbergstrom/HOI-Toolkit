from frcnn import FRCNN_Detector

det = FRCNN_Detector()

i = det.get_data_preds(['25.jpg'], '.')

for j in i:
    print('OUTTER LOOP')
    for k in j:
        print('INNER LOOP')
        print(k)


