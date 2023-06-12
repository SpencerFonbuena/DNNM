from torch.nn import Module
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import random

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
#print(f'use device: {DEVICE}')

# [Initialize stat-tracking]
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# [End init]


'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''


class MultiHeadAttention(Module):
    def __init__(self,
                 heads = int,
                 d_model = int,
                 qkpair = int,
                 value_count = int,
                 device = str):
        super(MultiHeadAttention, self).__init__()
        # [Making init variables class-wide available]
        self.qkpair = qkpair
        self.heads = heads
        self.inf = -2**32+1
        self.device= device
        # [End availability]

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

        # [Create Layers]
        self.query_weight = nn.Linear(d_model, qkpair * heads) 
        self.key_weight = nn.Linear(d_model, qkpair * heads)
        self.value_weight = nn.Linear(d_model, value_count * heads)
        self.output_layer = nn.Linear(value_count * heads, d_model)
        # [End creation]
        

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''


    def forward(self, x, stage):
        # [Begin Query, Key, and Value pair creation]
        queries = torch.cat(self.query_weight(x).chunk(self.heads, dim=-1), dim=0) # (128,120,8)
        keys = torch.cat(self.key_weight(x).chunk(self.heads, dim=-1), dim=0) # (128,120,8)
        values = torch.cat(self.value_weight(x).chunk(self.heads, dim=-1), dim=0) # (128, 120 ,8)
        # [End pair creation]


        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''


        # [Calculation of Multi-Head Attention]
        #Create pairwise connections between Queries and Keys
        score = torch.matmul((queries *0(self.qkpair**-.5) ), keys.transpose(-1,-2))  #(128,120,120)

        #mask future for realistic training
        if stage == 'train':
            mask = torch.ones_like(score[0])
            mask = mask.tril(diagonal=0)
            score = torch.where(mask > 0, score, (torch.ones_like(mask) * self.inf).to(self.device))

        #Raw attention scores
        score = F.softmax(score, dim=-1) #(128,120,120)
        #Multiply attention by the values, and concatenate the heads to prepare for FFN
        weight_V = torch.cat(torch.matmul(score, values).chunk(self.heads, dim=0), dim=-1) #(16,120,64)
        #Combine all features from  MHA through an FFN
        out = self.output_layer(weight_V) #(16,120,512)
        # [End calculations]

        return out
    


'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

# [Mock test the MHA]
'''
mockdata = torch.tensor(np.random.randn(16,120,512)).to(torch.float32)
test_init = MultiHeadAttention(heads=8, d_model=512, qkpair=8, value_count=8, device=DEVICE)
test_init(mockdata, 'test')
'''
# [End Mock Test]