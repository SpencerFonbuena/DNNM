from torch.nn import Module
import torch.nn as nn
import torch
import numpy as np
import random
import torchvision
import matplotlib.pyplot as plt

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
                 
                 inner_size = int):
        super(Encoder, self).__init__()

        self.multi_head_func = MultiHeadAttention(
                heads = heads,
                d_model = d_model,
                qkpair = qkpair,
                value_count = value_count,
                device = device
        )
        self.ffn_func = FeedForward( 
                d_model = d_model,
                inner_size= inner_size,
        )

        self.layernorm = nn.LayerNorm(d_model)


        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

    def forward(self, x, stage):
        #fig, axs = plt.subplots(2,3, figsize=(10,4))
        #axs[0,0].hist(x.view(-1).tolist(), 80)
        #axs[0,0].set_title('MHA input')

        recurrence = x #(16,120,512)
        #axs[0,1].hist(recurrence.view(-1).tolist(), 80)
        #axs[0,1].set_title('MHA recurrence')

        x = self.multi_head_func(x, stage) #(16,120,512)
        #axs[0,2].hist(x.view(-1).tolist(), 80)
        #axs[0,2].set_title('MHA output')
        
        x = self.layernorm(recurrence + x) #(16,120,512)
        #axs[1,0].hist(x.view(-1).tolist(), 80)
        #axs[1,0].set_title('encoder layernorm')

        x = self.ffn_func(x) #(16,120,512)
        #axs[1,1].hist(x.view(-1).tolist(), 80)
        #axs[1,1].set_title('encoder output')


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