from torch.nn import Module
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from module.hyperparameters import HyperParameters as hp

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
#print(f'use device: {DEVICE}')

class Gate(Module):
    def __init__(self, batch):
        super(Gate, self).__init__()
        self.batch = batch
        self.layer1 = nn.Conv2d(1,64,7,padding=3)
        self.layer2 = nn.Conv2d(64,64,7,padding=3)
        self.layer3 = nn.Conv2d(64,128,7,padding=3)
        self.layer4 = nn.Conv2d(128,128,7,padding=3)
        self.layer5 = nn.Conv2d(128,256,7,padding=3)
        self.layer6 = nn.Conv2d(256,256,7,padding=3)
        self.layer7 = nn.Conv2d(256,512,7,padding=3)
        self.layer8 = nn.Conv2d(512,512,7,padding=3)
        self.layer9 = nn.Conv2d(512,512,7,padding=3)
        self.layer10 = nn.Conv2d(512,512,7,padding=3)


        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.pool4 = nn.MaxPool2d(2,2)
        self.pool5 = nn.MaxPool2d(2,2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)

        self.mlp1 = nn.Linear(131072,1000)
        self.mlp2 = nn.Linear(1000,4)

    def forward(self, x):
        x = x.reshape(16,1,512,512)
        x = self.bn1(F.gelu(self.layer1(x)))
        recurrence = x 
        x = F.gelu(self.layer2(x))
        x = x + recurrence
        x = self.pool1(x)

        x = self.bn2(F.gelu(self.layer3(x)))
        recurrence = x 
        x = F.gelu(self.layer4(x))
        x = x + recurrence
        x = self.pool2(x)

        x = self.bn3(F.gelu(self.layer5(x)))
        recurrence = x 
        x = F.gelu(self.layer6(x))
        x = x + recurrence
        x = self.pool3(x)

        x = self.bn4(F.gelu(self.layer7(x)))
        recurrence = x 
        x = F.gelu(self.layer8(x))
        x = x + recurrence
        x = self.pool4(x)

        x = self.bn5(F.gelu(self.layer9(x)))
        recurrence = x 
        x = F.gelu(self.layer10(x))
        x = x + recurrence
        x = self.pool5(x)
        
        x = x.reshape(self.batch, -1)

        x = self.mlp1(x)
        x = self.mlp2(x)


        return x