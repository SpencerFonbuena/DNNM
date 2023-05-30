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
from module.fcnlayer import FCNLayer


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
                 layers=list, 
                 kss=list,
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

        self.fcn = FCNLayer(channel_in=channel_in,timestep_in=timestep_in,layers=layers, kss=kss, class_num=class_num)
        self.finalout = nn.Linear(8,4)

        # [End Gate & Out]

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

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
        channel, timestep = self.fcn(x_channel=x_channel, x_timestep=x_timestep)
        out = channel + timestep
        # [End tower combination]
        return out


# [Mock test the MHA]
'''
mockdata = torch.tensor(np.random.randn(16,120,9)).to(torch.float32)
test_init = Transformer(window_size=120,timestep_in=120,channel_in=9,heads=8,d_model=512,qkpair=8,value_count=8,device=DEVICE,inner_size=2048,class_num=4)
test_init(mockdata, 'test')
'''
# [End Mock Test]