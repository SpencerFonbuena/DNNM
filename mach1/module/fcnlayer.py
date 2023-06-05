from torch.nn import Module
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''


class FCNLayer(Module):
    def __init__(self,
                 timestep_in = int,
                 channel_in = int,
                 layers=list, 
                 kss=list,
                 class_num = int
                 ):
        super(FCNLayer, self).__init__()

            # [convolutions for channel]
        self.conv1 = nn.Conv1d(timestep_in, layers[1], kss[2], 1, 3)
        self.conv2 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv3 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)

        self.bn1 = nn.BatchNorm1d(layers[1])
        self.bn2= nn.BatchNorm1d(layers[1])
        self.bn3 = nn.BatchNorm1d(layers[1])

        self.gapchannel = nn.AdaptiveAvgPool1d(1)
        self.fcchannel = nn.Linear(layers[1], class_num)
            # [end convolutions for channel]

            # [convolutions for timestep]
        self.conv4 = nn.Conv1d(channel_in, layers[1], kss[2], 1, 3)
        self.conv5 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv6 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)

        self.bn4 = nn.BatchNorm1d(layers[1])
        self.bn5= nn.BatchNorm1d(layers[1])
        self.bn6 = nn.BatchNorm1d(layers[1])

        self.gaptimestep = nn.AdaptiveAvgPool1d(1)
        self.fctimestep = nn.Linear(layers[1], class_num)
            # [end convolutions for timestep]

    '''-----------------------------------------------------------------------------------------------------'''
    '''====================================================================================================='''
    
    def forward(self, x_channel, x_timestep):
            # [combine channel features]
        x_channel = F.relu(self.bn1(self.conv1(x_channel)))
        identity = x_channel

        x_channel = F.relu(self.bn2(self.conv2(x_channel)))
        x_channel = x_channel + identity
        identity = x_channel

        x_channel = F.relu(self.bn3(self.conv3(x_channel)))
        x_channel = x_channel + identity
        
        x_channel = self.gapchannel(x_channel)
        x_channel = x_channel.reshape(x_channel.shape[0], -1)
        x_channel = self.fcchannel(x_channel)
            # [End combination]

            # (combine timestep features)
        x_timestep = F.relu(self.bn4(self.conv4(x_timestep)))
        identity = x_timestep

        x_timestep = F.relu(self.bn5(self.conv5(x_timestep)))
        
        x_timestep = x_timestep + identity
        identity = x_timestep

        x_timestep = F.relu(self.bn6(self.conv6(x_timestep)))
        x_timestep = x_timestep + identity
        
        x_timestep = self.gapchannel(x_timestep)
        x_timestep = x_timestep.reshape(x_timestep.shape[0], -1)
        x_timestep = self.fcchannel(x_timestep)

        return x_timestep, x_channel