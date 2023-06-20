from torch.nn import Module
from torch.nn import ModuleList


import torch
import math
import random
import torch.nn.functional as F
import torchvision.ops.stochastic_depth as std
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from module.hyperparameters import HyperParameters as hp
from module.gate import Gate
from module.embedding import Embedding
from torch.nn import TransformerEncoderLayer
from module.fcnlayer import ResBlock
from tsai.models.FCN import FCN


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
                 device = str,
                 stack = int,
                 dropout = float,
                 
                 
                 #FCN Variables
                 class_num = int,
                 p = float,
                 ):
        
        super(Transformer, self).__init__()

        self.p = p
        self.stack = stack  



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
        


        # [Initialize Towers]
        #Channel Init
        self.channel_tower = ModuleList([
            TransformerEncoderLayer(
                 d_model=d_model,
                 nhead=heads,
                 dim_feedforward=4 * d_model,
                 dropout=dropout,
                 activation=F.gelu,
                 batch_first=True,
                 norm_first=True,
                 device=device
            ) for _ in range(stack)
        ])

        #Timestep Init
        self.timestep_tower = ModuleList([
            TransformerEncoderLayer(
                 d_model=d_model,
                 nhead=heads,
                 dim_feedforward=4 * d_model,
                 dropout=dropout,
                 activation=F.gelu,
                 batch_first=True,
                 norm_first=True,
                 device=device
            ) for _ in range(stack)
        ])
        # [End Towers]
        

        self.channelclassifier = nn.Linear(d_model, class_num)
        self.timestepclassifier = nn.Linear(d_model, class_num)
        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

    def forward(self, x):
        #Embed channel and timestep
        x_channel = self.channel_embedding(x).to(torch.float32) # (16,9,512)
        x_timestep = self.timestep_embedding(x).to(torch.float32) # (16,120,512)

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

        # [Loop through towers]

        # Channel tower
        for i, encoder in enumerate(self.channel_tower):
            identity = x_channel
            x_channel = std(x_channel, (i/self.stack) * self.p, 'batch')
            y_channel = encoder(x_channel)
            x_channel = y_channel + identity
        
        #Timestep tower
        for i, encoder in enumerate(self.timestep_tower):
            identity = x_timestep
            x_timestep = std(x_timestep, (i/self.stack) * self.p, 'batch')
            y_timestep = encoder(x_timestep)
            x_timestep = y_timestep + identity

        # [End loop]

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

        x_channel = self.channelclassifier(x_channel)
        x_channel = x_channel.mean(dim=1)
        x_channel = x_channel.reshape(64,1,4)

        x_timestep = self.timestepclassifier(x_timestep)
        x_timestep = x_timestep.mean(dim=1)
        x_timestep = x_timestep.reshape(64,1,4)

        preout = torch.cat([x_channel, x_timestep], dim=1)
        out = preout.mean(dim=1)

        return out


# [Mock test the MHA]
'''
mockdata = torch.tensor(np.random.randn(16,120,9)).to(torch.float32)
test_init = Transformer(window_size=120,timestep_in=120,channel_in=9,heads=8,d_model=512,qkpair=8,value_count=8,device=DEVICE,inner_size=2048,class_num=4)
test_init(mockdata, 'test')
'''
# [End Mock Test]