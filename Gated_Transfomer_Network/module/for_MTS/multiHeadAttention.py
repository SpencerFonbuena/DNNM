# @Time    : 2021/07/21 19:29
# @Author  : SY.M
# @FileName: multiHeadAttention.py


import torch


class MultiHeadAttention(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int):
        super(MultiHeadAttention, self).__init__()

        self.h = h
        self.q = q

        #Creating the weight matrices for the query key and value abstractions
        self.W_Q = torch.nn.Linear(in_features=d_model, out_features=q * h) #(512,64)
        self.W_K = torch.nn.Linear(in_features=d_model, out_features=q * h) #(512,64)
        self.W_V = torch.nn.Linear(in_features=d_model, out_features=v * h) #(512,64)
        self.W_out = torch.nn.Linear(in_features=v * h, out_features=d_model) #(64,512)

        self.inf = -2**32+1
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self,
                x: torch.Tensor,
                stage: str):
        #Multiplying the inputs by the weight matrices to come up with Q, K, V
        Q = torch.cat(self.W_Q(x).chunk(self.h, dim=-1), dim=0)
        K = torch.cat(self.W_K(x).chunk(self.h, dim=-1), dim=0)
        V = torch.cat(self.W_V(x).chunk(self.h, dim=-1), dim=0)

        #Multiplying Q by K to get the attention weights 
        score = torch.matmul(Q, K.transpose(-1, -2))  # / torch.sqrt(torch.Tensor(self.q)).to(self.device)

        #storing the scores to later create a heatmap. This has no effect on the computations of multi-head attention
        heatmap_score = score

        #If we are training, then we don't want the network to have access to future values, so we mask them.
        if stage == 'train':
            #This will create a tensor filled with ones that is the same size as the last x dimensions of score. Without the 0, ti would be the same size as the whole matrix
            mask = torch.ones_like(score[0])
            #Tril creates a lower triangular amtrix of 1s, and 0s, with 0s, being on the top. This is to mask out all future values
            mask = mask.tril(diagonal=0)
            # Torch.where is essentially an if statement. It's saying, if mask > 0, input score, otherwise, put in the the value at that same location found in the tensor filled - 
            #with ones that have the size of mask, multiplied by infinitity (not really infiinity, but a really big number. Because there is a negative in front however, it is actually)
            #a very negative number. The purpose of course being to nullify those connections to the network so that they don't learn. Why they don't just put 0 I'm not sure.
            score = torch.where(mask > 0, score, (torch.ones_like(mask) * self.inf).to(self.device))

        #Apply the softmax layer
        score = torch.nn.functional.softmax(score, dim=-1)
        #Multiply the softmaxed score by V, and then chunk it into heads
        weight_V = torch.cat(torch.matmul(score, V).chunk(self.h, dim=0), dim=-1)

        #multiply the concatenated scores by a Wo matrix to come out with the matrix that contains all the information from each different head
        out = self.W_out(weight_V) # (16,100,512) & (16,9,512)

        return out, heatmap_score