import torch
#from torch.autograd import Variable
#import torch.nn as nn
#import torch.optim as optim
#from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout
#from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

## Loading the data of the time domain (after preprocessing)
class ParkinsonsDataset(Dataset):
    def __init__(self):

        pathLabels = '/work3/s164229/train_labels2.csv'
        self.pathFiles = '/work3/s164229/stdnorm'
        #self.pathFiles = 'data/stdnorm'

        self.y = pd.read_csv(pathLabels)
        self.label = self.y['filenames']
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        tempath = self.y.iloc[index]['filenames']
        signalpath = os.path.join(self.pathFiles,tempath)
        self.x = torch.from_numpy(np.load(signalpath).astype(np.double))
        return self.x, self.y.iloc[index]['binary']

## Loading the data of the frequency domain
class ParkinsonsDatasetFreq(Dataset):
    def __init__(self):
        pathLabels = '/work3/s164229/train_labels2.csv'
        self.pathFiles = '/work3/s164229/freq_domain'
        self.y = pd.read_csv(pathLabels)
        self.label = self.y['filenames']
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        tempath = self.y.iloc[index]['filenames']
        signalpath = os.path.join(self.pathFiles,'freq_'+tempath)
        self.x = torch.from_numpy(np.load(signalpath).astype(np.double))
        return self.x, self.y.iloc[index]['binary']
