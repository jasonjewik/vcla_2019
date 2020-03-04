#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:07:38 2019

@author: keze
"""

import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import os
import os.path as osp
import cv2
import pickle
import numpy as np
import copy
import torch.nn.functional as F

DirectionDict = {'LEFT': 0, 'RIGHT': 1, 'STRAIGHT': 2}
classNum = 3

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    """
    Paramters:
        input -> [N, C], float32
        target -> [N, ], int64
    Returns:
        loss -> the computed loss of the network
    """
    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss
    
class DirectionDataset(Dataset):
    def __init__(self, split, splitRatio=0.9):
        self.dataFolder = 'C:\\GTAV_old\\4Train'
        filelist = os.listdir(self.dataFolder)
        if split == 'train':
            self.dataFileList = filelist[:int(splitRatio * len(filelist))]
            print('Training Sample Number: ', len(self.dataFileList))
        elif split == 'val':
            self.dataFileList = filelist[int(splitRatio * len(filelist)):]
            print('Validating Sample Number: ', len(self.dataFileList))

    def __len__(self):
        return len(self.dataFileList)

    def __getitem__(self, idx):
        return processOneItem(os.path.join(self.dataFolder, self.dataFileList[idx]))

class TestDirectionDataset(Dataset):
    def __init__(self):
        self.dataFolder = 'C:\\GTAV_old\\4Test'
        self.dataFileList = os.listdir(self.dataFolder)
        print('Testing Sample Number: ', len(self.dataFileList))

    def __len__(self):
        return len(self.dataFileList)

    def __getitem__(self, idx):
        return processOneItem(os.path.join(self.dataFolder, self.dataFileList[idx]))
    
class UnlabeledDataset(Dataset):
    def __init__(self, folder):
        self.dataFolder = folder
        self.dataFileList = os.listdir(self.dataFolder)
        print('Unlabeled Sample Number: ', len(self.dataFileList))
    
    def __len__(self):
        return len(self.dataFileList)
    
    def __getitem__(self, idx):
        return processOneItem(os.path.join(self.dataFolder, self.dataFileList[idx]))
    
    def getitempath(self, idx):
        return os.path.join(self.dataFolder, self.dataFileList[idx])

"""
Parameters:
    itemPath -> The file path of the item
Returns:
    swappedImg -> The original image but flipped
    label -> The numerical representation of the direction the vehicle is turning
"""
def processOneItem(itemPath):
    datas = pickle.load(open(itemPath, 'rb'))
    topImg = datas[0]
    topImg = cv2.resize(topImg, (224, 224))
    
    swappedImg = np.zeros((3, 224, 224), dtype='float32')
    for i in range(3):
        swappedImg[i] = topImg[:, :, i]
        
    # condenses our data down to just three categories from six
    direction = datas[1]
    if direction == 'BIG_RIGHT':
        direction = 'RIGHT'
    elif direction == 'BIG_LEFT':
        direction = 'LEFT'
    elif direction == 'NOT_MOVING':
        direction = 'STRAIGHT'
        
    label = DirectionDict[direction]
    return swappedImg, label

"""
Parameters:
    model -> The network to train
    trainloader -> The training data
    unlabeledLoaders -> The unlabeled data
    valLoader -> The validation data
    criterion -> The function for finding loss
    optimizer -> The optimizer to use
    unlabeled_data_path -> The path to the text file containing all the selected unlabeled data.
    numEpochs -> The number of epochs to train for
Returns:
    accuracy -> A measure of how accurate the network's guesses are.
"""
def trainModel(model, trainLoader, unlabeledLoaders, valLoader, criterion, optimizer, unlabeled_data_path, numEpochs=5):
    for phase in ['train', 'add_unlabeled', 'val']:
        if phase == 'train':
            model.train()
            for epoch in range(numEpochs):
#                print('Sub-Epoch {}/{}'.format(epoch + 1, numEpochs))
                for batch_idx, (inputs, labels) in enumerate(trainLoader):
                    inputs = Variable(inputs).float().cuda()
                    outputs = model(inputs)
                    labels = Variable(labels).cuda()
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        elif phase == 'add_unlabeled':
            for i in range(len(unlabeledLoaders)):
                print('Processing unlabeled dataset {}'.format(i))
                addUnlabeledData(model, unlabeledLoaders[i], unlabeled_data_path)         
        elif phase == 'val':
            accuracy = testModel(model, valLoader)
            return accuracy

"""
Parameters:
    model -> The network to train
    testLoader -> The testing data
Returns:
    The accuracy of the network
"""
def testModel(model, testLoader):
    model.eval()
    accuracy_pre = np.zeros((classNum,), dtype='float32')
    smpSum_pre = np.ones((classNum,), dtype='float32')
    accuracy_rec = np.zeros((classNum,), dtype='float32')
    smpSum_rec = np.zeros((classNum,), dtype='float32')
    
    for batch_idx, (inputs, labels) in enumerate(testLoader):
        inputs = Variable(inputs).cuda()
        outputs = model(inputs)
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

    for i in range(classNum):
        accuracy_pre[i] /= smpSum_pre[i]
        accuracy_rec[i] /= smpSum_rec[i]
#        print(smpSum_pre[i], smpSum_rec[i])
    
    print('Precision:', accuracy_pre, 'Avg: ' + str(np.mean(accuracy_pre)))
    print('Recall:', accuracy_rec, 'Avg: ' + str(np.mean(accuracy_rec)))

    return np.mean(accuracy_pre) * np.mean(accuracy_rec) / (np.mean(accuracy_pre) + np.mean(accuracy_rec))

"""
Parameters:
    model -> The network to run on.
    unlabeledLoader -> The unlabeled data.
    unlabeled_data_path -> The text file to write the selected data to.
    threshold -> The difference needed between the two highest outputs for the data to be selected.
Returns:
    selectedDataPath -> A list of paths to the selected unlabeled data into the txt file
    selectedDataPkl -> A pickle dumped file to save the data and its label
"""
def addUnlabeledData(model, unlabeledLoader, unlabeled_data_path, threshold=0.8):
    model.eval()

    for batch_idx, (inputs, labels) in enumerate(unlabeledLoader):
        inputs = Variable(inputs).cuda()
        outputs = model(inputs)
        outputs = outputs.cpu().detach().numpy()
        
        for i in range(outputs.shape[0]):
            # Gets the index of the highest element
            index = outputs[i].argmax()
            
            # only do the following if the predicted direction is not straight
            if (index != 4):
                highestElem = outputs[i][index]
                outputs[i][index] = -20 # this is really dumb, but I hope it works
    
                # Gets the second highest element            
                index = outputs[i].argmax()
                secondHighest = outputs[i][index]
                            
                # Check if the highest element exceeds the second highest by a threshold
                if (highestElem - secondHighest) > threshold:
                    unlabeledDataset = unlabeledLoader.dataset
                    fileindex = ((batch_idx + 1) * outputs.shape[0]) + (i + 1)

                    try:
                        itemPath = unlabeledDataset.getitempath(fileindex)
                        # Writes out the pickle file's path to a text file
                        with open(unlabeled_data_path, 'a') as f:
                            f.write(itemPath + '\n')         
                    except Exception as e:
                        print(e, fileindex)       
    
                    # Copies the selected unlabeled pickle files into a new directory
                    folderpath = 'E:\\ScriptRun - Copy\\selected_unlabeled\\'
                    if not osp.exists(folderpath):
                        os.mkdir(folderpath)
                    os.system('mklink ' + '"' + folderpath + itemPath.split('\\')[-1] + '" ' + itemPath)

if __name__ == '__main__':
    # Unlabeled data
    data_root_list = [osp.join('C:\\GTAV_old\\visiondata{}\\direction_det_pkls'.format(i)) for i in range(2, 4)]
    
    # Find unlabeled data text file
    unlabeled_data_path = 'E:\\ScriptRun - Copy\\unlabeled_data.txt'
    if not osp.exists(unlabeled_data_path):
        open(unlabeled_data_path, 'w+')
    
    # Gets the resnet model
    resNet = models.resnet18(pretrained=True)
    resNet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resNet.fc = nn.Sequential(
        *list(resNet.fc.children())[:-1] + [nn.Linear(in_features=512, out_features=classNum, bias=False), nn.LogSoftmax(dim=1)])
    resNet = nn.DataParallel(resNet)
    resNet = resNet.cuda() 
    
    # Gets the optimizer and sets the weights
    optimizer = optim.Adam(resNet.parameters(), lr=0.001)
#    weights = [1.0, 2.0, 1.0, 2.0, 0.5, 2.0]
    weights = [1.0, 2.0, 1.0]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = FocalLoss(weight=class_weights).cuda()
    
    # Training, validation, and testing data
    trainLoader = DataLoader(dataset=DirectionDataset('train'), batch_size=16, shuffle=True, num_workers=8)
    valLoader = DataLoader(dataset=DirectionDataset('val'), batch_size=16, shuffle=False, num_workers=2)
    testLoader = DataLoader(dataset=TestDirectionDataset(), batch_size=16, shuffle=False, num_workers=2)
#    unlabeledLoaders = [
#            DataLoader(dataset=UnlabeledDataset(data_root_list[i]), batch_size=16, shuffle=True, num_workers=1)
#            for i in range(1, 3)
#        ]   

    # Trains the model
    bestAcc = 0
    bestModel = None
    wholeNum = 100
    for i in range(wholeNum):
        print('-' * 10)
        print('Epoch {}/{}'.format(i + 1, wholeNum))
        currValAcc = trainModel(resNet, trainLoader, [], valLoader, criterion, optimizer, unlabeled_data_path)
#        currTestAcc = testModel(resNet, testLoader)
        if currValAcc > bestAcc:
            # Updates and saves the current best model
            bestModel = copy.deepcopy(resNet)
            # Updates and saves the curent best accuracy
            bestAcc = currValAcc

    # Tests the best model on the testing set
    print('Best Results:')
    testModel(bestModel, testLoader)
