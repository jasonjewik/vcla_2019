import os
import csv
import matplotlib.pyplot as plt
import math

data_path = "C:\\GTAV_program\\visiondata"
files = os.listdir(data_path)

x_coords = []
y_coords = []
xy_coords = []

for file in files:
    if file[-3:] == 'csv':
        fpath = os.path.join(data_path, file)
        with open(fpath, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            row = next(reader)
            x = float(row[0][1:])
            y = float(row[1])
            x_coords.append(x)
            y_coords.append(y)
            xy_coords.append([x, y])


def plot_all_points():
    plt.plot(x_coords, y_coords)
    plt.show()


def get_turns():
    # array of arrays like [W, A, S, D] - 1 means pressed, 0 means not pressed
    directions = []
    NO_DIR = [0, 0, 0, 0]
    RIGHT = [0, 0, 0, 1]
    FORWARD_RIGHT = [1, 0, 0, 1]
    FORWARD = [1, 0, 0, 0]
    FORWARD_LEFT = [1, 1, 0, 0]
    LEFT = [0, 1, 0, 0]

    for coord, next_coord in zip(xy_coords, xy_coords[1:]):
        adjusted_coord = [next_coord[0] - coord[0], next_coord[1] - coord[1]]
        angle = math.atan2(adjusted_coord[1], adjusted_coord[0]) / math.pi

        result = NO_DIR
        if (angle > 0 and angle < 1/4):
            result = RIGHT
        elif (angle >= 1/4 and angle < 5/12):
            result = FORWARD_RIGHT
        elif (angle >= 5/12 and angle < 7/12):
            result = FORWARD
        elif (angle >= 7/12 and angle < 3/4):
            result = FORWARD_LEFT
        elif (angle >= 3/4 and angle < 1):
            result = LEFT

        directions.append(result)

    return directions


if __name__ == "__main__":
    directions = get_turns()

    with open("output.csv", 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(directions)
