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
        self.channel_tower = TransformerEncoderLayer(
                 d_model=d_model,
                 nhead=heads,
                 dim_feedforward=4 * d_model,
                 dropout=dropout,
                 activation=F.gelu,
                 batch_first=True,
                 norm_first=True,
                 device=device
            ) 
        
        self.channel_encoder = nn.TransformerEncoder(
            encoder_layer=self.channel_tower,
            num_layers=stack,
            norm=nn.LayerNorm(d_model)
            
        )
        
        #Timestep Init
        timestep_tower = TransformerEncoderLayer(
                 d_model=d_model,
                 nhead=heads,
                 dim_feedforward=4 * d_model,
                 dropout=dropout,
                 activation=F.gelu,
                 batch_first=True,
                 norm_first=True,
                 device=device
            )
        
        self.timestep_encoder = nn.TransformerEncoder(
            encoder_layer=timestep_tower,
            num_layers=stack,
            norm=nn.LayerNorm(d_model)
            
        )
        # [End Towers]

        self.gate = torch.nn.Linear(in_features=timestep_in * d_model + channel_in * d_model, out_features=2)
        self.linear_out = torch.nn.Linear(in_features=timestep_in * d_model + channel_in * d_model,
                                          out_features=class_num)

    def forward(self, x):
        #Embed channel and timestep
        x_channel = self.channel_embedding(x).to(torch.float32) # (16,512,512)
        x_timestep = self.timestep_embedding(x).to(torch.float32) # (16,8,512)

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''
        
        channel_out = self.channel_encoder(x_channel).reshape(hp.BATCH_SIZE, 1, -1)
        timestep_out = self.timestep_encoder(x_timestep).reshape(hp.BATCH_SIZE, 1, -1)

        gate = torch.nn.functional.softmax(self.gate(torch.cat([x_timestep, x_channel], dim=-1)), dim=-1)

        gate_out = torch.cat([x_timestep * gate[:, 0:1], x_channel * gate[:, 1:2]], dim=-1)

        out = self.linear_out(gate_out)


        return out


# [Mock test the MHA]
'''
mockdata = torch.tensor(np.random.randn(16,120,9)).to(torch.float32)
test_init = Transformer(window_size=120,timestep_in=120,channel_in=9,heads=8,d_model=512,qkpair=8,value_count=8,device=DEVICE,inner_size=2048,class_num=4)
test_init(mockdata, 'test')
'''
# [End Mock Test]