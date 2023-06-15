from torch.nn import Module
import torch.nn as nn
import torch
import numpy as np
import random
import torchvision


from module.multiHeadAttention import MultiHeadAttention
from module.feedForward import FeedForward

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
#print(f'use device: {DEVICE}')

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''


class Encoder(Module):
    def __init__(self,
                 heads = int,
                 d_model = int,
                 qkpair = int,
                 value_count = int,
                 device = str,
                 
                 inner_size = int,
                 dropout = float):
        super(Encoder, self).__init__()

        self.multi_head_func = MultiHeadAttention(
                 d_model = d_model,
                 num_heads = heads,
                 device = str
        )
        self.ffn_func = FeedForward( 
                d_model = d_model,
                inner_size= inner_size,
        )
        
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_model)


        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

    def forward(self, x, stage):
        recurrence = x #(16,120,512)
        
        x = self.layernorm(x) #(16,120,512)
        x = self.multi_head_func(x, stage) #(16,120,512)
        x = self.dropout(x)
        x = self.ffn_func(x) #(16,120,512)
        x = x + recurrence

        return x

'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

# [Mock test the MHA]
'''
mockdata = torch.tensor(np.random.randn(16,120,512)).to(torch.float32)
test_init = Encoder(heads=8, d_model=512, qkpair=8, value_count=8, device=DEVICE, inner_size=2048)
test_init(mockdata, 'train')
'''
# [End Mock Test]