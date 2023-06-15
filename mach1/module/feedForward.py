from torch.nn import Module
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''


class FeedForward(Module):
    def __init__(self,
                 d_model: int,
                 inner_size: int):
        super(FeedForward, self).__init__()

        self.in_layer = nn.Linear(d_model, inner_size)
        self.out_layer = nn.Linear(inner_size, d_model)

        self.layernorm = nn.LayerNorm(normalized_shape=d_model)

    '''-----------------------------------------------------------------------------------------------------'''
    '''====================================================================================================='''
    
    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.in_layer(x) #(16,120,2048)
        x = F.gelu(x) #(16,120,2048)
        x = self.out_layer(x) #(16,120,512)
        x = x + residual
        return x 
    

'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

# [Mock test the FFN]
'''
mockdata = torch.tensor(np.random.randn(16,120,512)).to(torch.float32)
test_init = MultiHeadAttention(heads=8, d_model=512, qkpair=8, value_count=8, device=DEVICE)
test_init(mockdata, 'test')
'''
# [End Mock Test]