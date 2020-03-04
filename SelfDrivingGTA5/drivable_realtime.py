import time
import cv2
import numpy as np
import os

import torch
from torch.autograd import Variable
torch.backends.cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get mmdetection
CONFIG_FILE = 'D:\\mmdetection\\configs\\cascade_rcnn_x101_64x4d_fpn_1x_feat.py'
CHECKPOINT_FILE = 'D:\\mmdetection\\checkpoints\\cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth'
from mmdet.apis import init_detector, inference_detector
det_model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device)

from tools.grabscreen import grab_screen
from drivable_classifier import Net
from tools.utils_extra import showFrameTime

# The screen region containing GTA5 window
SCREEN_REGION = (768, 192, 1152, 912)

# Saved model filepath
MODEL_PATH = os.path.join(os.getcwd(), 'model_best.pth')

# Possible outputs
OUTPUTS = ['UNDRIVABLE', 'DRIVABLE']

def predictFromImage(model, im):
    output = model(im)
    output = output.to('cpu').detach().numpy()[0]

    return output, OUTPUTS[np.argmax(output)]

def processImage(im):
    result = inference_detector(det_model, im)
    result[1] = np.concatenate((result[1], result[3]), axis=0)

    if result[1].size == 0:        
        bbox_int = np.array([550, 430, 720, 710]) # just a guess
    else:
        bbox_int = result[1][0].astype(np.int32)
    
    margin_top = 200
    margin_side = 100
    left_top = (max(bbox_int[0] - margin_side, 0), max(bbox_int[1] - margin_top, 0))
    right_bottom = (min(bbox_int[2] + margin_side, im.shape[1]), min(bbox_int[1], im.shape[0]))

    topRegion = im[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    topRegion = cv2.resize(image, (320,160))
    topRegion = topRegion.T
    topRegion = topRegion / 255

    tensor = Variable(torch.from_numpy(topRegion).unsqueeze(0)).to(device)

    return tensor.float()
    

if __name__ == '__main__':
    model = torch.load(MODEL_PATH)
    model = model.to(device)
    model.eval()

    while True:
        try:
            start = time.time()
            image = grab_screen(SCREEN_REGION)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            res = processImage(image)
            output, pred = predictFromImage(model, res)
            os.system('cls')
            print(output, pred)

            cv2.waitKey(30)
            showFrameTime(start)
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()


