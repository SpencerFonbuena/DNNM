from torch.nn import Module
import torch.nn as nn
import torch
import numpy as np
import random


from module.multiHeadAttention import MultiHeadAttention
from module.feedForward import FeedForward

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

class Encoder(Module):
    def __init__(self,
                 heads = int,
                 d_model = int,
                 qkpair = int,
                 value_count = int,
                 device = str,
                 
                 inner_size = int):
        super(Encoder, self).__init__()
        
        # [Making init variables class-wide available]
        self.qkpair = qkpair
        self.heads = heads
        self.device=device
        self.d_model = d_model
        self.value_count = value_count

        self.inner_size = inner_size
        # [End availability]

        multi_head_func = MultiHeadAttention(
                heads = int,
                d_model = int,
                qkpair = int,
                value_count = int,
                device = str
        )
        ffn_func = FeedForward( 
                d_model = int,
                inner_size= int,
        )

    def forward(self, x, stage):
        
        recurrence = x
        x = MultiHeadAttention(x)
        x = nn.LayerNorm(recurrence + x)
        recurrence = nn.LayerNorm(recurrence + x)

        x = FeedForward(x)
        x = nn.LayerNorm(recurrence + x)
        return x
