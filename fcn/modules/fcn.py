import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCN(nn.Module):
    def __init__(self, data_in, data_out, layers=[128, 256, 512], kss=[8, 5, 3]):
        super().__init__()

        self.conv1 = nn.Conv1d(data_in, layers[2], kss[1], 1, 3)
        self.conv2 = nn.Conv1d(layers[2], layers[2], kss[1], 1, 2)
        self.conv3 = nn.Conv1d(layers[2], layers[2], kss[1], 1, 2)
        self.conv4 = nn.Conv1d(layers[2], layers[2], kss[1], 1, 2)
        self.conv5 = nn.Conv1d(layers[2], layers[2], kss[1], 1, 2)
        self.conv6 = nn.Conv1d(layers[2], layers[2], kss[1], 1, 2)
        self.conv7 = nn.Conv1d(layers[2], layers[2], kss[1], 1, 2)
        self.conv8 = nn.Conv1d(layers[2], layers[2], kss[1], 1, 2)
        self.conv9 = nn.Conv1d(layers[2], layers[2], kss[1], 1, 2)
        self.convout = nn.Conv1d(layers[2], layers[2], kss[1], 1, 2)
        
        self.bn1 = nn.BatchNorm1d(layers[2])
        self.bn2= nn.BatchNorm1d(layers[2])
        self.bn3 = nn.BatchNorm1d(layers[2])
        self.bn4 = nn.BatchNorm1d(layers[2])
        self.bn5= nn.BatchNorm1d(layers[2])
        self.bn6 = nn.BatchNorm1d(layers[2])
        self.bn7 = nn.BatchNorm1d(layers[2])
        self.bn8= nn.BatchNorm1d(layers[2])
        self.bn9 = nn.BatchNorm1d(layers[2])
        self.bnout = nn.BatchNorm1d(layers[2])

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(layers[2], data_out)


        
    def forward(self, x):
        x = x.transpose(-1,-2)
        x = F.relu(self.bn1(self.conv1(x)))
        identity = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x + identity
        identity = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x + identity
        identity = x
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = x + identity
        identity = x
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = x + identity
        identity = x
        x = F.relu(self.bnout(self.convout(x)))
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = F.softmax(x, 1)
        return x
    