# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 01:48:58 2020

@author: Jason
"""

from utils import timeout
from grabscreen import grab_screen
import keyboard

import cv2
import os
import os.path as osp
import pickle

# The screen region where the GTAV default opens
SCREEN_REGION = (320, 192, 1599, 911)

if __name__ == "__main__":
    # Create output directory
    parent_path = "C:\\Users\\kezew\\Documents\\VisionDataMixed"
    if not osp.exists(parent_path):
        os.mkdir(parent_path)
        
    num_subdirs = len(os.listdir(parent_path))
    output_path = osp.join(parent_path, "visiondata{}".format(num_subdirs))
    os.mkdir(output_path)
    
    timeout(5) # Waits for 5 seconds
    
    index = 0
    print("Collecting frames")
    # Collect screenshots until keyboard interrupt or Escape
    while True:
        try:
            fname = str(index).zfill(6)            
            screen = grab_screen(SCREEN_REGION)
            output = [screen, not keyboard.is_pressed('v')]
            pickle.dump(output, open(osp.join(output_path, fname + ".pkl"), "wb"))
            
            index += 1
        except KeyboardInterrupt:
            print("Done")
            break
        
    cv2.destroyAllWindows()
        