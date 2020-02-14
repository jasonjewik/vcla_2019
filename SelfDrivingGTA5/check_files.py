import os
import pickle
import sys

files = os.listdir(sys.argv[1])
files = [os.path.join(sys.argv[1], x) for x in files]
drivable, undrivable = 0, 0
for file in files:
    data = pickle.load(open(file, 'rb'))
    if data[1]:
            drivable += 1
    else:
            undrivable += 1
print(f'Drivable: {drivable}\nUndrivable: {undrivable}')