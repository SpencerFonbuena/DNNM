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
                 d_input: int,
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
        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.embedding_channel =  torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)
        

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x , stage):
        """
        forward propagation
        :param x: input
        :param stage: Used to describe whether it is the training process of the training set or the testing process of the testing set at this time. The mask mechanism is not added during the testing process
        :return: output, two-dimensional vector after gate, score matrix in step-wise encoder, score matrix in channel-wise encoder, three-dimensional matrix after step-wise embedding, three-dimensional matrix after channel-wise embedding, gate
        """
        # step-wise
        # The score matrix is ​​input, and mask and pe are added by default
        encoding_1 = self.embedding_channel(x.type(torch.FloatTensor).to(DEVICE)) 
        input_to_gather = encoding_1 

        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe

        #this applies the encoder, which includes the MHA, Add & Norm, Feed Forward, Add & Norm etc.
        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # channel-wise
        # score matrix is ​​channel without mask and pe by default
        
        encoding_2 = self.embedding_input(x.transpose(-1, -2).type(torch.FloatTensor).to(DEVICE))
        channel_to_gather = encoding_2

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        # 3D to 2D
        #A torch.view might be the better option here, so that we don't have to store any new memory
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        self.fgate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        encoding = torch.cat([encoding_1 * self.fgate[:, 0:1], encoding_2 * self.fgate[:, 1:2]], dim=-1)


        # output
        #The reason I didn't apply the softmax layer, is that supposedly the torch crossentropyloss expects unnormalized logits for each class. 
        output = self.output_linear(encoding)
        return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, self.fgate
