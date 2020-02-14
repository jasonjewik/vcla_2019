# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 00:53:23 2020

@author: Jason
"""

import os
import numpy as np
import copy
import random
import sys
import cv2
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

UNDRIVABLE = 0
DRIVABLE = 1

OS = sys.argv[1]
torch.backends.cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
split_ratio = 0.9

# Get all the data
if OS == "win":
    root_folder = "C:/Users/kezew/Documents/VisionDataMixed/visiondata{}" 
    CONFIG_FILE = "D:/mmdetection/configs/cascade_rcnn_x101_64x4d_fpn_1x_feat.py"
    CHECKPOINT_FILE = "D:/mmdetection/checkpoints/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth"
elif OS == "lnx":
    root_folder = "/home/keze/GTAV/VisionDataMixed/visiondata{}"
    CONFIG_FILE = "/home/keze/Codes/mmdetection/configs/cascade_rcnn_x101_64x4d_fpn_1x_feat.py"
    CHECKPOINT_FILE = "/home/keze/Codes/mmdetection/checkpoints/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth"
    
# Get mmdetection
from mmdet.apis import init_detector, inference_detector
det_model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device)

folder_paths = [root_folder.format(i) for i in range(0, 10)]
file_list = list() 
for folder_path in folder_paths:
    file_paths = os.listdir(folder_path)
    for file_path in file_paths:
        file_list.append(os.path.join(folder_path, file_path))
random.shuffle(file_list) 

def process_one_item(file_path):
    arr = pickle.load(open(file_path, 'rb'))
    
    if type(arr) == list:
        im, label = arr[0], arr[1]
    else:
        im = arr
        label = True

    im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)

    result = inference_detector(det_model, im)
    # print(result)
    result[1] = np.concatenate((result[1], result[3]), axis=0)

    if len(result[1]) > 0:
        playerBbox = result[1][0]
        bbox_int = playerBbox.astype(np.int32)
        cv2.rectangle(im, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (255, 255, 255), 3)
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

    cv2.imshow('im', im)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()
    
    tensor = np.zeros((3, 224, 224), dtype='float32')
    for i in range(3):
        tensor[i] = top_region[:,:,i]
    
    tensor = Variable(torch.from_numpy(tensor)).to(device)
		
    return tensor, int(label)
    
class TestDataset(Dataset):
    def __init__(self):
        self.file_list = file_list[round(len(file_list) * split_ratio):]
        print('Testing Sample Number: ', len(self.file_list))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        return process_one_item(self.file_list[idx])

class TrainDataset(Dataset):
    def __init__(self, split, train_val_ratio=0.9):
        self.file_list = file_list[:round(len(file_list) * split_ratio)]
        
        if split == 'train':
            self.file_list = self.file_list[:round(train_val_ratio * len(self.file_list))]
            print('Training Sample Number: ', len(self.file_list))
        elif split == 'val':
            self.file_list = self.file_list[round(train_val_ratio * len(self.file_list)):]
            print('Validating Sample Number: ', len(self.file_list))
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        return process_one_item(self.file_list[idx]) 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fctop = nn.Linear(44944, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 2)
        
    def forward(self, top):
        top = self.pool(F.relu(self.conv1(top)))        
        top = self.pool(F.relu(self.conv2(top)))
        top = top.reshape(top.size(0), -1)
        
        top = F.relu(self.fctop(top))
        output = F.relu(self.fc2(top))
        output = F.relu(self.fc3(output))

        return output
    
def trainModel(model, trainLoader, valLoader, criterion, optimizer, numEpochs=5):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            for epoch in range(numEpochs):
                print(f'Epoch: {epoch + 1}/{numEpochs}')
                for batch_idx, (top, label) in enumerate(trainLoader):  
                    outputs = model(top)

                    labels = Variable(label).to(device)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()      
        elif phase == 'val':
            accuracy = testModel(model, valLoader)
            return accuracy
        
def testModel(model, testLoader):
    model.eval()
    accuracy_pre = np.zeros((2,), dtype='float32')
    smpSum_pre = np.ones((2,), dtype='float32')
    accuracy_rec = np.zeros((2,), dtype='float32')
    smpSum_rec = np.zeros((2,), dtype='float32')
    
    for batch_idx, (top, labels) in enumerate(testLoader):
        outputs = model(top)
        outputs = outputs.cpu().detach().numpy()
        labels = labels.numpy()

        for i in range(labels.shape[0]):
            pred = outputs[i].argmax()

            if labels[i] == pred:
               accuracy_pre[labels[i]] += 1
            smpSum_pre[pred] += 1

            if labels[i] == pred:
               accuracy_rec[labels[i]] += 1
            smpSum_rec[labels[i]] += 1

    for i in range(2):
        accuracy_pre[i] /= smpSum_pre[i]
        accuracy_rec[i] /= smpSum_rec[i]
    
    print('Precision:', accuracy_pre, 'Avg: ' + str(np.mean(accuracy_pre)))
    print('Recall:', accuracy_rec, 'Avg: ' + str(np.mean(accuracy_rec)))

    return np.mean(accuracy_pre) * np.mean(accuracy_rec) / (np.mean(accuracy_pre) + np.mean(accuracy_rec))

if __name__ == "__main__":        
    train_loader = DataLoader(dataset=TrainDataset('train'), batch_size=32, shuffle=True, num_workers=0)
    validate_loader = DataLoader(dataset=TrainDataset('val'), batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=TestDataset(), batch_size=32, shuffle=False, num_workers=0)

    net = Net()
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    
    # Trains the model
    bestAcc = 0
    bestModel = None
    currValAcc = trainModel(net, train_loader, validate_loader, criterion, optimizer, numEpochs=10)
    currTestAcc = testModel(net, test_loader)
    if currValAcc > bestAcc:
        # Updates and saves the current best model
        bestModel = copy.deepcopy(net)
        # Updates and saves the curent best accuracy
        bestAcc = currValAcc

    # Tests the best model on the testing set
    print('Best Results:')
    testModel(bestModel, test_loader)
    torch.save(bestModel, 'model_best.pth')

        
            
        