import torch
import torch.nn as nn
from module.embedding import Embedding
from module.layers import Projector, Ns_Transformer
import torchvision.ops.stochastic_depth as std
from module.embedding import Embedding
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    """
    Non-stationary Transformer
    """
    def __init__(self,
                 d_model = int,
                 heads = int,
                 dropout = float,
                 dim_feedforward = int,
                 stack = int, 
 
                 # [Embedding]
                 channel_in = int,
                 window_size = int,
                 batch_size = int
                 ):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.embedding = Embedding(channel_in=channel_in, window_size=window_size)

        # [Encoder]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu',batch_first=True, norm_first=True,)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=stack, norm=nn.LayerNorm(d_model))
        
        self.preout = nn.Linear(d_model * window_size, d_model)

        self.out = nn.Linear(d_model, 1)


    def forward(self, x):
        
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = F.gelu(self.preout(x))
        x = self.out(x)

        return x
    
    