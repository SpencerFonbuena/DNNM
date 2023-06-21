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
        channel_encoder = TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
                device=device
        )
        self.channel_tower = nn.TransformerEncoder(
                encoder_layer=channel_encoder,
                num_layers=stack,
                norm=nn.LayerNorm(d_model)
        )

        #Timestep Init
        timestep_encoder = TransformerEncoderLayer(
                 d_model=d_model,
                 nhead=heads,
                 dim_feedforward=4 * d_model,
                 dropout=dropout,
                 activation=F.gelu,
                 batch_first=True,
                 norm_first=True,
                 device=device
            )
        self.timestep_tower = nn.TransformerEncoder(
                encoder_layer=timestep_encoder,
                num_layers=stack,
                norm=nn.LayerNorm(d_model)
        )
        
        '''self.gates = Gates(gate='enc-dec',c_in = channel_in, t_in=timestep_in, c_out = class_num, d_model = d_model, class_num=class_num,
                           heads=heads, dropout=dropout, device=device, stack=stack)'''

        decoder_layer = nn.TransformerDecoderLayer(
                 d_model=d_model,
                 nhead=heads,
                 dim_feedforward=4 * d_model,
                 dropout=dropout,
                 activation=F.gelu,
                 batch_first=True,
                 norm_first=True,
                 device=device
            )
        self.decoder_tower = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=stack,
            norm=nn.LayerNorm(d_model)
        )

        # [End Towers]
        
        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

    def forward(self, x):
        #Embed channel and timestep
        x_channel = self.channel_embedding(x).to(torch.float32) # (16,9,512)
        x_timestep = self.timestep_embedding(x).to(torch.float32) # (16,120,512)

        x_channel = self.channel_tower(x_channel)
        x_timestep = self.timestep_tower(x_timestep)
        

        
        
        return out


class Gates(Module):
    def __init__(self,
                 gate = str,
                 c_in = int,
                 t_in = int, 
                 c_out = int, 
                 d_model = int ,
                 class_num = int):
        super().__init__()
        self.gate = gate

        if gate == 'C_FCN':
            self.cfcn = FCN(c_in=c_in, c_out=c_out)
        elif gate == 'T_FCN':
            self.tfcn = FCN(c_in=t_in, c_out=c_out)
        elif gate == 'naive':
            self.linear = nn.Linear(d_model, d_model)
            self.classifier = nn.Linear(d_model, class_num)
            

    def forward(self, x):
            
        if self.gate == 'C-FCN':
            return self.cfcn(x)
        elif self.gate == 'T-FCN':
            return self.tfcn(x)
        elif self.gate == 'naive':
            pooled = nn.Tanh(self.linear(x[:, 0]))
            logits = self.classifier(pooled)
            return logits
