from torch.nn import Module
import torch
import numpy as np
import random

from module.feedForward import FeedForward
from module.multiHeadAttention import MultiHeadAttention

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

class Encoder(Module):
    def __init__():
        super(Encoder, self).__init__()

    def forward(self, x, stage):  
        return
