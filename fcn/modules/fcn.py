import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.ops.stochastic_depth as std
from modules.resblock import ResBlock
from torch.nn import ModuleList


class FCN(nn.Module):
    def __init__(self, data_in, data_out, layers=list, kss=list, device = str, p = float, stack = int):
        super().__init__()

        self.convin = nn.Conv1d(data_in, layers[1], kss[2], 1, 3)

        self.FCN = ModuleList([
            ResBlock(
                 layers=layers,
                 kss = kss,
                 p = p,
                 device = device,
                 
            ) for _ in range(stack)
        ])

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(layers[1], data_out)


        
    def forward(self, x):
        x = x.transpose(-1,-2)
        x = F.relu(self.bn1(self.conv1(x)))
        
        for resnet in self.FCN:
            x = resnet(x)

        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = F.softmax(x, 1)
        return x
    