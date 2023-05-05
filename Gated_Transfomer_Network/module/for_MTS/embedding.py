# @Time    : 2021/07/21 19:36
# @Author  : SY.M
# @FileName: embedding.py

import torch
import math


class Embedding(torch.nn.Module):
    def __init__(self,
                 d_feature: int,
                 d_timestep: int,
                 d_model: int,
                 wise: str = 'timestep' or 'feature'):
        super(Embedding, self).__init__()

        #wise is shoosing what kind of inputs. If it is univariate, then wise will be timestep, if it is multivariate, then it will be feature. 
        #it might also have to do with wich encoding we are doing. If we are doing the temporal encoding, it is timestep, if we are doing the channel encoding
        #it is going to be feature
        assert wise == 'timestep' or wise == 'feature', 'Embedding wise error!'
        self.wise = wise

        if wise == 'timestep':
            self.embedding = torch.nn.Linear(d_feature, d_model) # (9, 512)
        elif wise == 'feature':
            self.embedding = torch.nn.Linear(d_timestep, d_model) # (Window, 512) *right now the window is set at 100

    def forward(self,
                x: torch.Tensor):
        if self.wise == 'feature':
            #This takes the linear torch layer, and multiplies it by x, in an x @ self.embedding manner
             # (16,9,100)
            x = self.embedding(x)
             # (16,9,512)
        elif self.wise == 'timestep':
            #transpose the last two dimensions. The -1, and -2, allow for flexibility in number of dimensions
            x = self.embedding(x.transpose(-1, -2)) #Tensor comes in as a (batch, feature, timestep) -> (16, 9, 100) | it leaves as a (batch, timestep, d_model) -> (16,100,512)
            x = position_encode(x)
            #(16,100,512)
        return x, None

#This encodes the position so that the timestep encoder will have a sense of the temporal dimension
def position_encode(x):

    # torch.ones makes a 0 matrix identical to the last x dimensions of whatever matrix you put in. In this case it is the 0 row
    pe = torch.ones_like(x[0])
    # arange makes a list from 0 to x.shape[1], then the unsqueeze adds a deimension with size one insered at the specified position, which in this case is that last position
    position = torch.arange(0, x.shape[1]).unsqueeze(-1)
    # Creates a tensor, with a range from 0, to the shape of the last dimension of x. the 2 at the end signifies the size of steps to take with each stride
    temp = torch.Tensor(range(0, x.shape[-1], 2))
    temp = temp * -(math.log(10000) / x.shape[-1])
    temp = torch.exp(temp).unsqueeze(0)
    temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
    pe[:, 0::2] = torch.sin(temp)
    pe[:, 1::2] = torch.cos(temp)
    # (16, 100, 512)
    return x + pe