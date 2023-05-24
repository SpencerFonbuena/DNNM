import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FCN(nn.Module):
    def __init__(self, data_in, data_out, layers=[256, 512, 256], kss=[8, 5, 3]):
        super().__init__()

        self.conv1 = nn.Conv1d(data_in, layers[0], kss[0], 1, 3)
        self.conv2 = nn.Conv1d(layers[0], layers[1], kss[1], 1, 2)
        self.conv3 = nn.Conv1d(layers[1], layers[2], kss[2], 1, 1)
        
        self.bn1 = nn.BatchNorm1d(layers[0])
        self.bn2= nn.BatchNorm1d(layers[1])
        self.bn3 = nn.BatchNorm1d(layers[2])

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(layers[2], data_out)


        
    def forward(self, x):
        x = x.transpose(-1,-2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = F.softmax(x, 1)
        return x
    