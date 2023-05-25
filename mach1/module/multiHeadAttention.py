from torch.nn import Module
import torch
import math
import torch.nn.functional as F
import numpy as np
import random
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

class MultiHeadAttention(Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def forward():
        return 