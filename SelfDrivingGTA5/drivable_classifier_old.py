# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 00:53:23 2020

@author: Jason
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import os
import numpy as np
import copy
import random

import utils

UNDRIVABLE = 0
DRIVABLE = 1

# Get all the data
root_folder = "C:\\Users\\kezew\\Documents\\VisionDataMixed\\visiondata{}"
folder_paths = [root_folder.format(i) for i in range(0, 10)]

# additional_root = "C:\\Users\\kezew\\Documents\\VisionData\\visiondata{}\\rgbImgs_first"
# additional_paths = [additional_root.format(i) for i in range(1, 10)]

file_list = list() 
for folder_path in folder_paths:
    file_paths = os.listdir(folder_path)
    for file_path in file_paths:
        file_list.append(os.path.join(folder_path, file_path))
# for folder_path in additional_paths:
# 	file_paths = os.listdir(folder_path)
# 	for file_path in file_paths:
# 		file_list.append(os.path.join(folder_path, file_path))
random.shuffle(file_list) 

split_ratio = 0.9
    
class DrivableDataset(Dataset):
    def __init__(self, split, splitRatio=0.9):
        self.root_folder = "C:\\Users\\kezew\\Documents\\VisionData\\visiondata{}\\rgbImgs_first"
        self.folder_paths = [self.root_folder.format(i) for i in range(1, 10)]
        
        self.file_list = list() 
        for folder_path in self.folder_paths:
            file_paths = os.listdir(folder_path)
            for file_path in file_paths:
                self.file_list.append(os.path.join(folder_path, file_path))
        
        if split == 'train':
            self.file_list = self.file_list[:int(splitRatio * len(self.file_list))]
            print('Training Sample Number: ', len(self.file_list))
        elif split == 'val':
            self.file_list = self.file_list[int(splitRatio * len(self.file_list)):]
            print('Validating Sample Number: ', len(self.file_list))
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        return utils.process_one_item(self.file_list[idx], DRIVABLE)

class UndrivableDataset(Dataset):
    def __init__(self, split, splitRatio=0.9):
        self.root_folder = "C:\\Users\\kezew\\Documents\\VisionDataBad\\visiondata{}"
        self.folder_paths = [self.root_folder.format(i) for i in range(0, 10)]
        
        self.file_list = list() 
        for folder_path in self.folder_paths:
            file_paths = os.listdir(folder_path)
            for file_path in file_paths:
                self.file_list.append(os.path.join(folder_path, file_path))
        
        if split == 'train':
            self.file_list = self.file_list[:int(splitRatio * len(self.file_list))]
            print('Training Sample Number: ', len(self.file_list))
        elif split == 'val':
            self.file_list = self.file_list[int(splitRatio * len(self.file_list)):]
            print('Validating Sample Number: ', len(self.file_list))
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        return utils.process_one_item(self.file_list[idx], UNDRIVABLE)
    
class TestDataset(Dataset):
    def __init__(self):
        # self.drivable_folder = "C:\\Users\\kezew\\Documents\\VisionData\\visiondata{}\\rgbImgs_first"
        # self.drivable_paths = [self.drivable_folder.format(i) for i in range(12, 16)]
        
        # self.file_list = list()
        # for folder_path in self.drivable_paths:
        #     file_paths = os.listdir(folder_path)
        #     for file_path in file_paths:
        #         self.file_list.append(os.path.join(folder_path, file_path))

        # self.split = len(self.file_list)
        
        # self.undrivable_folder = "C:\\Users\\kezew\\Documents\\VisionDataBad\\visiondata{}"
        # self.undrivable_paths = [self.undrivable_folder.format(i) for i in range(6, 8)]
            
        # for folder_path in self.undrivable_paths:
        #     file_paths = os.listdir(folder_path)
        #     for file_path in file_paths:
        #         self.file_list.append(os.path.join(folder_path, file_path))

        self.file_list = file_list[round(len(file_list) * split_ratio):]
        print('Testing Sample Number: ', len(self.file_list))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # if idx < self.split:
        #     return utils.process_one_item(self.file_list[idx], DRIVABLE)
        # else:
        #     return utils.process_one_item(self.file_list[idx], UNDRIVABLE)
        return utils.process_one_item(self.file_list[idx])

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
        return utils.process_one_item(self.file_list[idx])       
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fctop = nn.Linear(89280, 120)
        self.fcsides = nn.Linear(308688, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        
    def forward(self, left):
        left = self.pool(F.relu(self.conv1(left)))
        left = self.pool(F.relu(self.conv2(left)))
        left = left.reshape(left.size(0), -1)
        
        left = F.relu(self.fcsides(left))
        left = F.relu(self.fc2(left))
        left = F.relu(self.fc3(left))
        return left
    
def trainModel(model, trainLoader, valLoader, criterion, optimizer, numEpochs=5):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            for epoch in range(numEpochs):
                print(f'Epoch: {epoch + 1}/{numEpochs}')
                for batch_idx, (top, left, right, labels) in enumerate(trainLoader):  
                    left = Variable(left).float().cuda()
                    outputs = model(left)
                    labels = Variable(labels).cuda()
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
    
    for batch_idx, (top, left, right, labels) in enumerate(testLoader):
        left = Variable(left).float().cuda()
        outputs = model(left)
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
    train_loader = DataLoader(dataset=TrainDataset('train'), batch_size=64, shuffle=True, num_workers=2)
    validate_loader = DataLoader(dataset=TrainDataset('val'), batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=TestDataset(), batch_size=32, shuffle=False, num_workers=2)

    net = Net()
    net = net.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    
    # Trains the model
    bestAcc = 0
    bestModel = None
    currValAcc = trainModel(net, train_loader, validate_loader, criterion, optimizer, 10)
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

        
            
        