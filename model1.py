# Contains the neural network
# Task:- Implement the Neural Network module according to problem statement specifications


from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, num_channels=3):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,8,3)
        self.conv3 = nn.Conv2d(8,3,1)

    def forward(self, x):
        #Inputs - 4x4 patches
        (N,n,m,c) = x.shape
        y = x.view(N,c,n,m)
        y = F.interpolate(y,size=(2*n,2*m))
        y = torch.tensor(y,dtype=torch.float)    #Otherwise were running into error "expected scalar type Byte but found Float"
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.relu(y)
        return y

model = Net()

