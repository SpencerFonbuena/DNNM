from torch.nn import Module
from torch.nn import ModuleList


import torch
import math
import torch.nn.functional as F
import numpy as np
import random

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
                 
                 #Gate Variables
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
        self.gate = torch.nn.Linear(in_features=timestep_in * d_model + channel_in * d_model, out_features=2)
        self.linear_out = torch.nn.Linear(in_features=timestep_in * d_model + channel_in * d_model, out_features=class_num)
        # [End Gate & Out]

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
        #print(x_timestep.shape, x_channel.shape) | ((16,120,512), (16,9,512))
        x_timestep = x_timestep.reshape(x_timestep.shape[0], -1) #(16,61440)
        x_channel = x_channel.reshape(x_channel.shape[0], -1)# (16,4608)

        gate = torch.nn.functional.softmax(self.gate(torch.cat([x_timestep, x_channel], dim=-1)), dim=-1)
        gate_out = torch.cat([x_timestep * gate[:, 0:1], x_channel * gate[:, 1:2]], dim=-1)
        out = self.linear_out(gate_out)
        # [End tower combination]
        return out


# [Mock test the MHA]
'''
mockdata = torch.tensor(np.random.randn(16,120,9)).to(torch.float32)
test_init = Transformer(window_size=120,timestep_in=120,channel_in=9,heads=8,d_model=512,qkpair=8,value_count=8,device=DEVICE,inner_size=2048,class_num=4)
test_init(mockdata, 'test')
'''
# [End Mock Test]