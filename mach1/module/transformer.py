from torch.nn import Module
from torch.nn import ModuleList


import torch
import math
import random
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from module.embedding import Embedding
from module.encoder import Encoder


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
#print(f'use device: {DEVICE}')

# [Maintain random seed]
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# [End maintenance]



'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''


class Transformer(Module):
    def __init__(self,
                 
                 #Embedding Variables
                 window_size = int,
                 timestep_in = str,
                 channel_in = str,

                 #MHA Variables
                 heads = int,
                 d_model = int,
                 qkpair = int,
                 value_count = int,
                 device = str,
                 stack = int,
                 
                 #FFN Variables
                 inner_size = int,
                 
                 #FCN Variables
                 layers=[128, 256, 512], 
                 kss=[7, 5, 3],
                 class_num = int
                 ):
        super(Transformer, self).__init__()
        
        # [Initialize Embedding]

        #Channel embedding Init
        self.channel_embedding = Embedding(
                 channel_in = channel_in,
                 timestep_in = timestep_in,
                 d_model = d_model,
                 window_size = window_size,
                 tower='channel')
        
        #Timestep embedding Init
        self.timestep_embedding = Embedding(
                 channel_in = channel_in,
                 timestep_in = timestep_in,
                 d_model = d_model,
                 window_size = window_size,
                 tower='timestep')
        
        # [End Embedding]


        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

        # [Initialize Towers]
        #Channel Init
        self.channel_tower = ModuleList([
            Encoder(
                 heads = heads,
                 d_model = d_model,
                 qkpair = qkpair,
                 value_count = value_count,
                 device = device,
                 
                 inner_size = inner_size
            ) for _ in range(stack)
        ])

        #Timestep Init
        self.timestep_tower = ModuleList([
            Encoder(
                 heads = heads,
                 d_model = d_model,
                 qkpair = qkpair,
                 value_count = value_count,
                 device = device,
                 
                 inner_size = inner_size
            ) for _ in range(stack)
        ])
        # [End Init]

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

        # [Gate & Out Init]

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
        # [End Gate & Out]

        self.finalout = nn.Linear(8,4)

    def forward(self, x, stage):
        #Embed channel and timestep
        x_channel = self.channel_embedding(x).to(torch.float32) # (16,9,512)
        x_timestep = self.timestep_embedding(x).to(torch.float32) # (16,120,512)

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

        # [Loop through towers]

        # Channel tower
        for encoder in self.channel_tower:
            x_channel = encoder(x=x_channel, stage=stage)
        
        #Timestep tower
        for encoder in self.timestep_tower:
            x_timestep = encoder(x = x_timestep, stage=stage)

        # [End loop]

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

        # [Combine tower features]

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

        out = self.finalout(torch.cat([x_channel,x_timestep], dim=-1))
        # [End tower combination]
        return out


# [Mock test the MHA]
'''
mockdata = torch.tensor(np.random.randn(16,120,9)).to(torch.float32)
test_init = Transformer(window_size=120,timestep_in=120,channel_in=9,heads=8,d_model=512,qkpair=8,value_count=8,device=DEVICE,inner_size=2048,class_num=4)
test_init(mockdata, 'test')
'''
# [End Mock Test]