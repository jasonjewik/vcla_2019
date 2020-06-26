import os
import csv
import matplotlib.pyplot as plt

data_path = "C:\\GTAV_program\\visiondata"
files = os.listdir(data_path)
x_coords = []
y_coords = []

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

plt.plot(x_coords, y_coords)
plt.show()
