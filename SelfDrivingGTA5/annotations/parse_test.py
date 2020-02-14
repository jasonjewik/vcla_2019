import csv
import pickle
import numpy as np
import os
import natsort

def process(dstPath, beginIdx=0, endIdx='end'):
    root_dir = '/home/keze/GTAV/rgbImgs_first/'
    src_images = os.listdir(root_dir)
    src_images = natsort.natsorted(src_images)
    if endIdx == 'end':
       src_images = src_images[beginIdx:]
    else:
       src_images = src_images[beginIdx:endIdx]
    
    annotations = []
    for image in src_images:
        isPlayer = True
        image_id = image[:len(image)-8].zfill(9)
        csv_data = '../csv/gddata' + image_id + '.csv'
        bboxes = []
        labels = []
        # with open(csv_data) as csvfile:
        #     csvReader = csv.reader(csvfile, delimiter=',')
        #     next(csvReader)       
    
        #     for row in csvReader:
        #         x_list = []
        #         y_list = []
        #         #label = row[0]
        #         if row[0] == 'Veh' and isPlayer:
        #             label = 2
        #             isPlayer = False
        #         elif row[0] == 'Veh':
        #             label = 1
        #         elif row[0] == 'Ped':
        #             label = 3
        #         else:
        #             label = 0
        #         for each in range(1, 17):
        #             if each % 2 == 0:
        #                 row[each] = row[each].replace('(', '').replace(')', '')
        #                 row[each] = row[each].split()
        #                 x_list.append(int(row[each][1]))
        #                 y_list.append(int(row[each][0]))
        #         x0 = min(x_list)
        #         x1 = max(x_list)
        #         y0 = min(y_list)
        #         y1 = max(y_list)
        #         bbox = [x0, y0, x1, y1]
        #         bboxes.append(bbox)
        #         labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        bboxes_ignore = np.zeros((0, 4))
        annotation = {
            'filename': root_dir  + image, 
            'width':1280,
            'height':720,
            'ann':{
                'bboxes' : bboxes.astype(np.float32),
                'labels' : labels.astype(np.int64),
                'bboxes_ignore' : bboxes_ignore.astype(np.float32),
            }
        }
        annotations.append(annotation)

    with open(dstPath, 'wb') as fp:
        pickle.dump(annotations, fp)
    
if __name__ == '__main__':
#    process('top_train.pkl', 0, 5000)
   process('first_val.pkl', 0)
