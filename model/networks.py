import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Dropout
#from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
import pickle
from sklearn.model_selection import KFold
import optuna
from optuna.trial import TrialState


# Model that takes the raw filtered data of 12 channels
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.L = 16
        self.K = 1
        self.M = 64
        ## feature extraction
        self.featureExtract = nn.Sequential(nn.Conv1d(in_channels = 12 , out_channels = 32, kernel_size=8, padding =1,stride = 1),
                                            nn.BatchNorm1d(32),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.MaxPool1d(2),
                                            #nn.AvgPool1d(2),
                                            Dropout(p=0.1, inplace=False),

                                            nn.Conv1d(in_channels = 32,out_channels = 32, kernel_size=8,padding =1, stride = 1),
                                            nn.BatchNorm1d(32),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.MaxPool1d(2),
                                            #nn.AvgPool1d(2),
                                            Dropout(p=0.1, inplace=False),

                                            nn.Conv1d(in_channels = 32, out_channels = 16, kernel_size=16, padding =1, stride = 1),
                                            nn.BatchNorm1d(16),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.MaxPool1d(2),
                                            #nn.AvgPool1d(2),
                                            Dropout(p=0.1, inplace=False),

                                            nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size=16, padding =1, stride = 1),
                                            nn.BatchNorm1d(16),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.MaxPool1d(2),
                                            #nn.AvgPool1d(2),
                                            Dropout(p=0.1, inplace=False),

                                            nn.Flatten(),
                                            nn.Linear(320, self.M))

        self.attention = nn.Sequential(nn.Linear(self.M, self.L),
                                       nn.Tanh(),
                                       nn.Linear(self.L, self.K))

        self.classify = nn.Sequential(nn.Linear(self.M,32),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      Dropout(p = 0.2, inplace = False),
                                      nn.Linear(32,16),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      Dropout(p = 0.5, inplace = False),
                                      nn.Linear(16,2),
                                      nn.Softmax(dim = 1))

    def forward(self, x):
        ## feature extraction

        H = self.featureExtract(x)

        A = self.attention(H)
        A = torch.transpose(A,1,0)
        s = nn.Softmax(dim=1)
        A = s(A)

        z = torch.mm(A,H)

        output = self.classify(z)
        Y_prob , Y_hat = torch.max(output,dim=1)

        return Y_prob, Y_hat, A, output



# Model that takes frequency input of 76 length
class Freq_Net(nn.Module):
    def __init__(self):
        super(Freq_Net, self).__init__()

        self.L = 16
        self.K = 1
        self.M = 64
        ## feature extraction
        self.featureExtract = nn.Sequential(nn.Linear(76,256),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            Dropout(p = 0.5, inplace = False),
                                            nn.Linear(256,128),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            Dropout(p = 0.5, inplace = False),
                                            nn.Linear(128,self.M))

        self.attention = nn.Sequential(nn.Linear(self.M, self.L),
                                       nn.Tanh(),
                                       nn.Linear(self.L, self.K))

        self.classify = nn.Sequential(nn.Linear(self.M,32),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      Dropout(p = 0.1, inplace = False),
                                      nn.Linear(32,16),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      Dropout(p = 0.4, inplace = False),
                                      nn.Linear(16,2),
                                      nn.Softmax(dim = 1))

    def forward(self, x):
        ## feature extraction

        H = self.featureExtract(x)

        A = self.attention(H)
        A = torch.transpose(A,1,0)
        s = nn.Softmax(dim=1)
        A = s(A)

        z = torch.mm(A,H)

        output = self.classify(z)
        Y_prob , Y_hat = torch.max(output,dim=1)

        return Y_prob, Y_hat, A, output
