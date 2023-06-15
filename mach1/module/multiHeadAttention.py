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
                 d_model: int,
                 num_heads: int,
                 qkv_bias: bool = True,
                 device = str):
        super(MultiHeadAttention, self).__init__()
        # [Making init variables class-wide available]
        
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = head_dim**-0.5
        self.inf = -2**32+1

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model)
        
        self.device= device


    def forward(self, x, stage):
        
        B, T, F = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B,T, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads,T, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        
        if stage == 'train':
            mask = torch.ones_like(attn[0])
            mask = mask.tril(diagonal=0)
            attn = torch.where(mask > 0, attn, (torch.ones_like(mask) * self.inf)).to(self.device)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).view(B, self.num_heads, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1)
        x = self.proj(x)

        return x
    


'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

# [Mock test the MHA]
'''
mockdata = torch.tensor(np.random.randn(16,120,512)).to(torch.float32)
test_init = MultiHeadAttention(heads=8, d_model=512, qkpair=8, value_count=8, device=DEVICE)
test_init(mockdata, 'test')
'''
# [End Mock Test]