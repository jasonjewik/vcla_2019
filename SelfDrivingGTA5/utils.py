# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 01:39:49 2020

@author: Jason
"""

import numpy as np
import time
import pickle
import random
import cv2
import sys

NOISE_TYPES = {0: "gauss", 1: "s&p", 2: "poisson", 3: "speckle"}

# Code from https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = tuple([np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape])
        out[coords] = 1
        
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = tuple([np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape])
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    
def timeout(num_seconds):
    for i in range(num_seconds):
        print(num_seconds - i)
        time.sleep(1)
        
def process_one_item(file_path):        
    arr = pickle.load(open(file_path, 'rb'))
    if type(arr) == list:
        img, label = arr[0], arr[1]
    else:
        img = arr
        label = True
        
    percent_cutoff = 0.35
    width, height = img.shape[1], img.shape[0]
    left_x = round(percent_cutoff * width)
    right_x = width - left_x
    top_y = round(percent_cutoff * height)

    top_region = img[:top_y, left_x:right_x, :] / 255
    left_region = img[:, :left_x, :] / 255
    right_region = img[:, right_x:, : ] / 255

    cv2.imshow('whole', img)    
    cv2.imshow('top', top_region)
    cv2.imshow('left', left_region)
    cv2.imshow('right', right_region)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()
    
    # randomly add noise sometimes
    nt = random.randint(0, len(NOISE_TYPES) * 3)
    if NOISE_TYPES.get(nt) is not None:
        top_region = noisy(NOISE_TYPES[nt], top_region)
        left_region = noisy(NOISE_TYPES[nt], left_region)
        right_region = noisy(NOISE_TYPES[nt], right_region)
        
    return top_region.T, left_region.T, right_region.T, int(label)

def showFrameTime(previousTime):
    print('Total time used ' + str(time.time() - previousTime) + ' seconds.')
    print()