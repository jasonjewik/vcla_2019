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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import os
import os.path as osp
import cv2
import pickle
import numpy as np
import copy
import tools.uitls_extra as utils

DirectionDict = {
    'FORWARD': 0,
    'LEFT': 1,
    'RIGHT': 2,
    'BACKWARD': 3,
    'NOT_MOVING': 4
}
classNum = len(DirectionDict.values())
torch.backends.cudnn.enabled = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAIN_TEST_SPLIT = 0.9

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

class TrainDataset(Dataset):
    def __init__(self, file_list, mmdet_cfg, dataset_type='train', train_val_ratio=0.9, train_test_split=TRAIN_TEST_SPLIT):
        self.file_list = file_list[:round(len(file_list) * train_test_split)]
        self.cfg = mmdet_cfg
        
        if dataset_type == 'train':
            self.file_list = self.file_list[:round(train_val_ratio * len(self.file_list))]
            print('Training Sample Number: ', len(self.file_list))
        elif dataset_type == 'val':
            self.file_list = self.file_list[round(train_val_ratio * len(self.file_list)):]
            print('Validating Sample Number: ', len(self.file_list))
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        cache_res = utils.checkCache(filepath)

        if cache_res is not None:
            return cache_res[0], cache_res[2]

        arr = pickle.load(open(filepath, 'rb'))
        img, label, move = arr[0], arr[1], arr[2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data = utils.prepare_data(img, 
            self.cfg['img_transform'], self.cfg['img_scale'], self.cfg['keep_ratio'])
        data['img'][0] = data['img'][0].to(DEVICE)
        res = utils.processOneImg(self.cfg['model'], data, DEVICE, img)
        res = utils.preprocessInput(res)

        utils.cacheResult([res, label, move], filepath)

        return res, move

class TestDataset(Dataset):
    def __init__(self, file_list, mmdet_cfg, train_test_split=TRAIN_TEST_SPLIT):
        self.file_list = file_list[round(len(file_list) * train_test_split):]
        self.cfg = mmdet_cfg

        print('Testing Sample Number: ', len(self.file_list))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        cache_res = utils.checkCache(filepath)

        if cache_res is not None:
            return cache_res[0], cache_res[2]

        arr = pickle.load(open(self.file_list[idx], 'rb'))
        img, move = arr[0], arr[2]
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = utils.prepare_data(img, 
            self.cfg['img_transform'], self.cfg['img_scale'], self.cfg['keep_ratio']) 
        data['img'][0] =  data['img'][0].to(DEVICE)
        res = utils.processOneImg(self.cfg['model'], data, DEVICE, img)
        res = utils.preprocessInput(res)

        return res, move

def trainModel(model, trainLoader, valLoader, criterion, optimizer):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            for batch_idx, (im, move) in enumerate(trainLoader):
                outputs = model(Variable(im).to(DEVICE))

                labels = Variable(move).to(DEVICE)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        elif phase == 'val':
            accuracy = testModel(model, valLoader)
            return accuracy

def testModel(model, testLoader):
    model.eval()
    accuracy_pre = np.zeros((classNum,), dtype='float32')
    smpSum_pre = np.ones((classNum,), dtype='float32')
    accuracy_rec = np.zeros((classNum,), dtype='float32')
    smpSum_rec = np.zeros((classNum,), dtype='float32')
    
    for batch_idx, (im, move) in enumerate(testLoader):
        outputs = model(Variable(im).to(DEVICE))
        outputs = outputs.cpu().detach().numpy()
        move = move.numpy()

        for i in range(move.shape[0]):
            pred = outputs[i].argmax()

            if move[i] == pred:
               accuracy_pre[move[i]] += 1
            smpSum_pre[pred] += 1

            if move[i] == pred:
               accuracy_rec[move[i]] += 1
            smpSum_rec[move[i]] += 1

    print(accuracy_pre, smpSum_pre, accuracy_rec, smpSum_rec)

    for i in range(classNum):
        accuracy_pre[i] /= smpSum_pre[i]
        accuracy_rec[i] /= smpSum_rec[i]
    
    print('Precision:', accuracy_pre, 'Avg: ' + str(np.mean(accuracy_pre)))
    print('Recall:', accuracy_rec, 'Avg: ' + str(np.mean(accuracy_rec)))

    return np.mean(accuracy_pre) * np.mean(accuracy_rec) / (np.mean(accuracy_pre) + np.mean(accuracy_rec))

if __name__ == '__main__':    
    # Gets the resnet model
    resNet = models.resnet18(pretrained=True)
    resNet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resNet.fc = nn.Sequential(
        *list(resNet.fc.children())[:-1] + [nn.Linear(in_features=512, out_features=classNum, bias=False), nn.LogSoftmax(dim=1)])
    resNet = nn.DataParallel(resNet)
    resNet = resNet.to(DEVICE)
    
    # Gets the optimizer and sets the weights
    optimizer = optim.Adam(resNet.parameters(), lr=0.00001)
    weights = [0.1, 1.0, 1.0, 0.1, 0.1]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = FocalLoss(weight=class_weights).cuda()
    
    # Training, validation, and testing data
    trainLoader = DataLoader(dataset=TrainDataset('train'), batch_size=16, shuffle=False, num_workers=8)
    valLoader = DataLoader(dataset=TrainDataset('val'), batch_size=16, shuffle=False, num_workers=2)
    testLoader = DataLoader(dataset=TestDataset(), batch_size=16, shuffle=False, num_workers=2)

    # Trains the model
    numEpochs = 100
    bestAcc = 0
    bestModel = copy.deepcopy(resNet)
    for epoch in range(numEpochs):
        print(f'Epoch: {epoch + 1}/{numEpochs}')
        currValAcc = trainModel(resNet, trainLoader, valLoader, criterion, optimizer)
        if currValAcc > bestAcc:
            # Updates and saves the current best model
            bestModel = copy.deepcopy(resNet)
            # Updates and saves the curent best accuracy
            bestAcc = currValAcc

    # Tests the best model on the testing set
    print('Best Results:')
    testModel(bestModel, testLoader)
    torch.save(bestModel, 'dir_model_best.pth')
