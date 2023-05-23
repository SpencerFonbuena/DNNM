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
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool=False,
                 dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)

        self.W_o = torch.nn.Linear(v * h, d_model)

        self.device = device
        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout)
        self.score = None

    def forward(self, x, stage):
            #print('input time step:', x.shape)
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
            #print('Q:', Q.shape)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
            #print('K:', K.shape)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)
            #print('V:', V.shape)

        #This creates the attention, which means that self.score is the un-softmaxed attnention weights
        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
            #print('Queries times Keys:', score.shape)
        self.score = score

        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2**32+1]).expand_as(score[0]).to(self.device))

        score = F.softmax(score, dim=-1)
            #print('softmaxed values: ',score.shape)
        attention = torch.matmul(score, V)
            #print("Attention Values: ",attention.shape)
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
            #print('Attention Heads: ',attention_heads.shape)
        self_attention = self.W_o(attention_heads)
            #print('Output Shape:', self_attention.shape) # (16,120,512) | The output is the same as the input, it's just been encoded
        return self_attention, self.score