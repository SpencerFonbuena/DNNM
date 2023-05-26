import torch
import torch.nn as nn
from torch.nn import Module
import numpy as np
import random


from hyperparameters import HyperParameters as hp
#from dataset_process.dataset_process import Create_Dataset as cd

# [Initialize stat-tracking]
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# [End init]

class Embedding(Module):
    def __init__(self,
                 stage = str,
                 channel_in = str,
                 timestep_in = str,
                 d_model = str,
                 window_size = int):
        super(Embedding, self).__init__()

        # [Making init variables class-wide available]
        self.stage = stage
        self.d_model = d_model
        self.window_size = window_size
        # [End availability]

        '''-----------------------------------------------------------------------------------------------------'''
        '''====================================================================================================='''

        # [Init layers]
        self.ffchannelembedding = nn.Linear(timestep_in, d_model)
        self.fftimestepembedding = nn.Linear(channel_in, d_model)
        # positional encoding of some sort
        # [End Init]
    
    def forward(self, x, tower):
        pe = positional_encoding(max_position=self.window_size, d_model=self.d_model)
        if tower == 'channel':
            x = x.transpose(-1,-2)
            x = self.ffchannelembedding(x) #(16,9,512)
        if tower == 'timestep':
            x = self.fftimestepembedding(x) # (16,120,512)
            x = pe + x #(16,120,512)
        return x
    
def positional_encoding(max_position, d_model, min_freq=1e-4):
    print(max_position, d_model)
    '''max_position: window size
        d_model: out_size from the embedding class'''
    position = np.arange(max_position)
    freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
    
    '''Possible Error: Not sure what the memory is doing just casting this numpy array to a torch tensor'''
    pos_enc = torch.tensor(pos_enc)
    '''End Possible Error'''
    return pos_enc

# [Mock test the embedding]
'''
mockdata = torch.tensor(np.random.randn(16,120,9)).to(torch.float32)
test_emb = Embedding(stage='train', channel_in=mockdata.shape[2], timestep_in=mockdata.shape[1], d_model=512,window_size=mockdata.shape[1])
test_emb(mockdata, 'timestep')
test_emb(mockdata, 'channel')
'''
# [End mock embedding]