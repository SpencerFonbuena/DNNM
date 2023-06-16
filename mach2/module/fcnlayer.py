import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import torchvision.ops.stochastic_depth as std

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''


class ResBlock(pl.LightningModule):
    def __init__(self, layers = list, kss = list, p = float):
        super(ResBlock, self).__init__()
        self.p = p

        self.conv1 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv2 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)

        self.bn1 = nn.BatchNorm1d(layers[1])
        self.bn2= nn.BatchNorm1d(layers[1])

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

    def forward(self, x):
        identity = x

        #x = std(x, self.p, 'batch')
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + identity

        return x