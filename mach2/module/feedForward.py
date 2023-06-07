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
        #fig, axs = plt.subplots(2,3, figsize=(10,4))
        #axs[0,0].hist(x.view(-1).tolist(), 80)
        #axs[0,0].set_title('input')

        residual = x
        #axs[0,1].hist(residual.view(-1).tolist(), 80)
        #axs[0,1].set_title('residual')
        
        x = self.in_layer(x) #(16,120,2048)
        #axs[0,2].hist(x.view(-1).tolist(), 80)
        #axs[0,2].set_title('ffn layer in')

        x = F.relu(x) #(16,120,2048)
        #axs[1,0].hist(x.view(-1).tolist(), 80)
        #axs[1,0].set_title('ffn relu')

        x = self.out_layer(x) #(16,120,512)
        #axs[1,1].hist(x.view(-1).tolist(), 80)
        #axs[1,1].set_title('ffn layer out')

        x = self.layernorm(x + residual)
        #axs[1,2].hist(x.view(-1).tolist(), 80)
        #axs[1,2].set_title('out with res')
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