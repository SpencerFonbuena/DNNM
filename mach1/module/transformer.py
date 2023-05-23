from torch.nn import Module
import torch
from torch.nn import ModuleList
from module.encoder import Encoder
import math
import torch.nn.functional as F
import numpy as np
import random

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
#print(f'use device: {DEVICE}')

class Transformer(Module):
    def __init__(self,
                 d_model: int,
                 d_timestep: int,
                 d_channel: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.0,
                 pe: bool = False,
                 mask: bool = False):
        super(Transformer, self).__init__()

        #These for loops will loop through the encodder N amount of times, which N is the number of heads. So this is creating the "multi" portion of the multi-headed attention layer
        self.encoder_channel = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_timestep = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.embedding_channel =  torch.nn.Linear(d_timestep, d_model)
        self.embedding_timestep = torch.nn.Linear(d_channel, d_model)
        

        self.gate = torch.nn.Linear(d_model * d_timestep + d_model * d_channel, 2)
        self.output_linear = torch.nn.Linear(d_model * d_timestep + d_model * d_channel, d_output)

        self.pe = pe
        self.d_timestep = d_timestep
        self.d_model = d_model

    def forward(self, x , stage):

        

        return 
