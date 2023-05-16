from torch.nn import Module
import torch
import math
import torch.nn.functional as F


class MultiHeadAttention(Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool=False,
                 dropout: float = 0.2):
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
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        with torch.no_grad():
            print('M', Q.min(), Q.max(), Q.mean())
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        with torch.no_grad():
            print('M', K.min(), K.max(), K.mean())
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)
        with torch.no_grad():
            print('M', V.min(), V.max(), V.mean())

        #This creates the attention, which means that self.score is the un-softmaxed attnention weights
        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        with torch.no_grad():
            print('M', score.max(), score.min(), score.mean())
        self.score = score

        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2**32+1]).expand_as(score[0]).to(self.device))

        score = F.softmax(score, dim=-1)
        with torch.no_grad():
            print('M', score.max(), score.min(), score.mean())
        #print('M', score[0], score.shape) #(128,100,100)
        attention = torch.matmul(score, V)
        #print('M', attention[0], attention.shape) #(128,100,8)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self.W_o(attention_heads)
        with torch.no_grad():
            print('M', self_attention.max(), self_attention.min(), self_attention.mean())
        #print('M', score[0], self.score[0], score.shape, self.score.shape)
        return self_attention, self.score