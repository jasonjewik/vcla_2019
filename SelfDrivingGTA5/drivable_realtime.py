import time
import cv2
import numpy as np
import os

import torch
from torch.autograd import Variable
torch.backends.cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get mmdetection
CONFIG_FILE = "D:/mmdetection/configs/cascade_rcnn_x101_64x4d_fpn_1x_feat.py"
CHECKPOINT_FILE = "D:/mmdetection/checkpoints/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth"
from mmdet.apis import init_detector, inference_detector
det_model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device)

from grabscreen import grab_screen
from drivable_classifier import Net
from utils import showFrameTime

# The screen region containing GTA5 window
SCREEN_REGION = (768, 192, 1152, 912)

# Saved model filepath
DRIVABLE_CLASSIFIER_MODEL = "C:\\Users\\kezew\\Documents\\SelfDrivingGTA5\\model_best.pth"

# Possible outputs
OUTPUTS = ['UNDRIVABLE', 'DRIVABLE']

"""
Parameters:
    model_file -> The path of the best model
Returns:
    The loaded model
"""
def initModel(model_file=DRIVABLE_CLASSIFIER_MODEL):
    model = torch.load(model_file)
    model = model.to(device)
    model.eval()

    return model

def process_one_item(im):
    result = inference_detector(det_model, im)
    result[1] = np.concatenate((result[1], result[3]), axis=0)

    if len(result[1]) > 0:
        playerBbox = result[1][0]
        bbox_int = playerBbox.astype(np.int32)
        margin_top = 100
        margin_side = 150
        
        left_top = [bbox_int[0] - margin_side, bbox_int[1] - margin_top]
        right_bottom = [bbox_int[2] + margin_side, bbox_int[1]]
        
        # make sure nothing is out of bounds
        if left_top[0] < 0:
            right_bottom[0] -= left_top[0]
            left_top[0] = 0
        
        top_region = np.copy(im[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]) / 255
        top_region = cv2.resize(top_region, (224, 224))
    else:
        # if the player is not detected, we still need to feed in an image
        width = im.shape[1]
        left_x = round(width * 0.5 - 112)
        right_x = round(width * 0.5 + 112)
        top_region = np.copy(im[:224, left_x:right_x, :]) / 255
    
    tensor = np.zeros((3, 224, 224), dtype='float32')
    for i in range(3):
        tensor[i] = top_region[:,:,i]
    
    tensor = Variable(torch.from_numpy(tensor).unsqueeze(0)).to(device)
		
    return tensor

"""
Parameters:
    model -> The model to use for predictions
    inputs -> The input for the network
Returns:
    The value of the activations and drivability.
"""
def predictFromImages(model, top):
    output = model(top)
    output = output.to('cpu').detach().numpy()[0]
    
    return output, OUTPUTS[np.argmax(output)]

def main():
    model = initModel()

    while True:
		# Check for keyboard interrupt
        try:
            start = time.time()
            image = grab_screen(SCREEN_REGION)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            top = process_one_item(image)

            output, prediction = predictFromImages(model, top)
            os.system('cls')
            print(output, prediction)

            cv2.waitKey(30)
            showFrameTime(start)
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
