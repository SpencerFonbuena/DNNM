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
from module.embedding import Embedding
from torch.nn import TransformerEncoderLayer
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

        


        # [Gate & Out Init]

        self.fcnchannel = FCN(timestep_in,class_num)
        self.fcntimestep = FCN(channel_in, class_num)

        self.out = nn.Linear(8,class_num)

        '''self.gate = torch.nn.Linear(in_features=timestep_in * d_model + channel_in * d_model, out_features=2)
        self.linear_out = torch.nn.Linear(in_features=timestep_in * d_model + channel_in * d_model,
                                          out_features=class_num)'''
        
        
        # [End Gate & Out]

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

        # [Combine tower features]
        def hiddenStates():

            # [FCN]
            '''#embed channels and timesteps into convolution
            x_channel = F.relu(self.bnchannel(self.convchannel(x_channel)))
            x_timestep = F.relu(self.bntimestep(self.convtimestep(x_timestep)))

            #feed them through the resblocks
            for module in self.fcnchannel:
                y = module(x_channel)
                x_channel = y
            
            for module in self.fcntimestep:
                y = module(x_timestep)
                x_timestep = y

            #prepare for combination
            x_channel = self.gapchannel(x_channel)
            x_channel = x_channel.reshape(x_channel.shape[0], -1)
            x_channel = self.fcchannel(x_channel)

            x_timestep = self.gaptimestep(x_timestep)
            x_timestep = x_timestep.reshape(x_timestep.shape[0], -1)
            x_timestep = self.fctimestep(x_timestep)

            preout = self.pre_out(torch.cat([x_timestep, x_channel], dim=-1))
            out = self.out(preout)'''
            # [End FCN]

            # [Gates]
            '''x_timestep = x_timestep.reshape(x_timestep.shape[0], -1)
            x_channel = x_channel.reshape(x_channel.shape[0], -1)

            gate = torch.nn.functional.softmax(self.gate(torch.cat([x_timestep, x_channel], dim=-1)), dim=-1)

            gate_out = torch.cat([x_timestep * gate[:, 0:1], x_channel * gate[:, 1:2]], dim=-1)

            out = self.linear_out(gate_out)'''

            # [End Gates]
        
        #out = self.fcnchannel(x_channel)

        channelout = self.fcnchannel(x_channel)
        timestepout = self.fcntimestep(x_timestep)
        out = self.out(torch.cat([channelout,timestepout]))


        return out


# [Mock test the MHA]
'''
mockdata = torch.tensor(np.random.randn(16,120,9)).to(torch.float32)
test_init = Transformer(window_size=120,timestep_in=120,channel_in=9,heads=8,d_model=512,qkpair=8,value_count=8,device=DEVICE,inner_size=2048,class_num=4)
test_init(mockdata, 'test')
'''
# [End Mock Test]