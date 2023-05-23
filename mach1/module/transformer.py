from torch.nn import Module
import torch
from torch.nn import ModuleList
from module.encoder import Encoder
import math
import torch.nn.functional as F
import numpy as np
import random
from module.hyperparameters import HyperParameters as hp

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
        self.encoder_timestep = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_channel = ModuleList([Encoder(d_model=d_model,
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

        # [Positional Encoding]
        def positional_encoding(max_position, d_model, min_freq=1e-4):
            position = np.arange(max_position)
            freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
            pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
            pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
            pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
            return pos_enc
        positional_code = positional_encoding(max_position=hp.WINDOW_SIZE, d_model=self.d_model)
        # [End Positional Encoding]
        

        
        # [Embedding inputs]
        #Embedding time-step 
        time_input = F.tanh(self.embedding_timestep(x))
            #add the positional encoding for time_step
        time_input = time_input + torch.tensor(positional_code).to(DEVICE)
            #cast to a float32
        time_input = time_input.to(torch.float32)

        #Embedding channel
        channel_input = F.tanh(self.embedding_channel(x.transpose(-1,-2))).to(torch.float32)
        # [End embedding]


        # [Step-Wise encoder]
        for encoder in self.encoder_timestep:
            time_input, _ = encoder(time_input, stage = stage)
        # [End Step-Wise encoder]


        # [Channel-Wise Encoder]
        for encoder in self.encoder_channel:
            channel_input, _ = encoder(channel_input, stage = stage)
        # [End Channel_Wise Encoder]


        # [Package Features]
        # you can't concatenate two tensors with different second dimensions. Therefore, you have to flatten the last two dimensions to get the concatenation to work
        time_input = time_input.reshape(time_input.shape[0], -1)
        channel_input = channel_input.reshape(channel_input.shape[0], -1)
        
        #Package together all the information
        self.fgate = torch.nn.functional.softmax(self.gate(torch.cat([time_input, channel_input], dim=-1)), dim=-1)
        gate_out = torch.cat([time_input * self.fgate[:, 0:1], channel_input * self.fgate[:, 1:2]], dim=-1)
        out = self.output_linear(gate_out)
        # [End Package Features]


        return out
