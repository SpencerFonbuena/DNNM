from torch.nn import Module
import torch
import torch.nn.functional as F
import numpy as np
import random

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

class FeedForward(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int = 512):
        super(FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)

    def forward(self, x):

        x = self.linear_1(x)
        x = F.tanh(x)
        x = self.linear_2(x)
        return x