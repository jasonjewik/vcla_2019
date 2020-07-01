import pickle
import os
import cv2
import sys

dirnum = input("Which folder? ")
imnum = input("Which image? ")

DIRECTORY = "E:\\GitHub\\vcla_2019\\image_data\\visiondata" + dirnum

if os.path.isdir(DIRECTORY):
    file_list = os.listdir(DIRECTORY)
else:
    print("Not a valid directory!")
    sys.exit()

if imnum == "a":
    for i in range(0, len(file_list)):
        f = pickle.load(open(os.path.join(DIRECTORY, file_list[i]), 'rb'))
        data = f
        cv2.imshow(f'window {i}', data)
        cv2.waitKey(500)
        cv2.destroyWindow(f'window {i}')
    cv2.waitKey(0)
    sys.exit()

imnum = int(imnum)
if imnum < len(file_list):
    f = pickle.load(open(os.path.join(DIRECTORY, file_list[imnum]), 'rb'))
else:
    print("Not a valid image!")
    sys.exit()

data = f[0]
cv2.imshow('window', data)
cv2.waitKey(0)
