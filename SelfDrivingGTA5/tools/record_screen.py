from utils_extra import timeout
from grabscreen import grab_screen
# import keyboard

import os
import os.path as osp

import pickle

SCREEN_REGION = (321, 192, 1600, 912)
FORWARD = 0
LEFT = 1
RIGHT = 2
BACKWARD = 3
NOT_MOVING = 4

FRAMES_TO_COLLECT = 20000

parent_path = "E:\\GitHub\\vcla_2019\\image_data"
if not osp.exists(parent_path):
    os.mkdir(parent_path)

num_subdirs = len(os.listdir(parent_path))
output_path = osp.join(parent_path, "visiondata{}".format(num_subdirs))
os.mkdir(output_path)

timeout(5)

index = 0
print("Collecting frames!")
while index <= FRAMES_TO_COLLECT:
    try:
        fname = str(index).zfill(6)
        screen = grab_screen(SCREEN_REGION)
        # drivable = int(not keyboard.is_pressed('v'))

        move = [0, 0, 0, 0, 0]  # FORWARD, LEFT, ... , NOT_MOVING
        # move[FORWARD] = int(keyboard.is_pressed('w'))
        # move[LEFT] = int(keyboard.is_pressed('a'))
        # move[RIGHT] = int(keyboard.is_pressed('s'))
        # move[BACKWARD] = int(keyboard.is_pressed('d'))
        # if max(move[:NOT_MOVING]) == 0:
        #     move[NOT_MOVING] = 1

        # output = [screen, drivable, move]
        output = screen
        pickle.dump(output, open(
            osp.join(output_path, fname + ".pkl"), 'wb'))
        index += 1
    except KeyboardInterrupt:
        break

print("Done")
