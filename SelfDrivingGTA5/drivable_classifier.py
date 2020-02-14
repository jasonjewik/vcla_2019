import os
import numpy as np
import copy
import cv2
import pickle
import argparse

import mmcv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

import tools.utils_extra as utils

torch.backends.cudnn.enabled = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAIN_TEST_SPLIT = 0.9

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
        arr = pickle.load(open(self.file_list[idx], 'rb'))

        if type(arr) == list:
            img, label = arr[0], arr[1]
        else:
            img = arr
            label = True
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = utils.prepare_data(img, 
            self.cfg['img_transform'], self.cfg['img_scale'], self.cfg['keep_ratio'])
        data['img'][0] =  data['img'][0].to(DEVICE)
        res = utils.processOneImg(self.cfg['model'], data, DEVICE)

        return res, int(label)

class TestDataset(Dataset):
    def __init__(self, file_list, mmdet_cfg, train_test_split=TRAIN_TEST_SPLIT):
        self.file_list = file_list[round(len(file_list) * train_test_split):]
        self.cfg = mmdet_cfg

        print('Testing Sample Number: ', len(self.file_list))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        arr = pickle.load(open(self.file_list[idx], 'rb'))

        if type(arr) == list:
            img, label = arr[0], arr[1]
        else:
            img = arr
            label = True
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = utils.prepare_data(img, 
            self.cfg['img_transform'], self.cfg['img_scale'], self.cfg['keep_ratio']) 
        data['img'][0] =  data['img'][0].to(DEVICE)
        res = utils.processOneImg(self.cfg['model'], data, DEVICE)

        return res, int(label)

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

                    labels = Variable(label).to(DEVICE)
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
    args = utils.parse_args()  

    # Initialize mmdetection and get paths to the data folders
    mmdet_cfg = utils.initModels(DEVICE)
    file_list = utils.getFiles()

    # Get data loaders    
    training_loader = DataLoader(dataset=TrainDataset(file_list, mmdet_cfg), 
        batch_size=32, shuffle=True, num_workers=0)
    validation_loader = DataLoader(dataset=TrainDataset(file_list, mmdet_cfg, 'val'), 
        batch_size=32, shuffle=False, num_workers=0)
    testing_loader = DataLoader(dataset=TestDataset(file_list, mmdet_cfg), 
        batch_size=32, shuffle=False, num_workers=0)

    # Initialize neural net
    net = Net()
    net = net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    
    # Trains the model
    bestAcc = 0
    bestModel = None
    currValAcc = trainModel(net, training_loader, validation_loader, criterion, optimizer)
    currTestAcc = testModel(net, testing_loader)
    if currValAcc > bestAcc:
        # Updates and saves the current best model
        bestModel = copy.deepcopy(net)
        # Updates and saves the curent best accuracy
        bestAcc = currValAcc

    # Tests the best model on the testing set
    print('Best Results:')
    testModel(bestModel, testing_loader)
    torch.save(bestModel, 'model_best.pth')