import torch
import torch.nn as nn
from torch.nn import Module
import numpy as np
import random
import math
# Make us of GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
print(f'use device: {DEVICE}')

# [Initialize stat-tracking]
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# [End init]

'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

class Embedding(Module):
    def __init__(self,
                 channel_in = int,
                 d_model = int,
                 window_size = int,):
        super(Embedding, self).__init__()

        # [Making init variables class-wide available]
        self.d_model = d_model
        self.window_size = window_size

        # [End availability]

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

        # [Init layers]
        self.ffchannelembedding = nn.Linear(channel_in, 512)
        # positional encoding of some sort
        # [End Init]
    
        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''
    
    def forward(self, x):
        x = self.ffchannelembedding(x)#(16,120,512)
        x = positional_encoding(x)
        return x
    
'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

def positional_encoding(x):

    pe = torch.ones_like(x[0])
    position = torch.arange(0, x.shape[1]).unsqueeze(-1)
    temp = torch.Tensor(range(0, x.shape[-1], 2))
    temp = temp * -(math.log(10000) / x.shape[-1])
    temp = torch.exp(temp).unsqueeze(0)
    temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
    pe[:, 0::2] = torch.sin(temp)
    pe[:, 1::2] = torch.cos(temp)
    return x + pe
'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

# [Mock test the embedding]
'''
mockdata = torch.tensor(np.random.randn(16,120,9)).to(torch.float32)
test_emb = Embedding(channel_in=mockdata.shape[2], timestep_in=mockdata.shape[1], d_model=512,window_size=mockdata.shape[1], tower='timestep')
test_emb(mockdata)
'''
# [End mock embedding]