import torch
import torch.nn as nn
from module.embedding import Embedding
from module.layers import Projector, Ns_Transformer
import torchvision.ops.stochastic_depth as std
from module.embedding import Embedding

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
                 window_size = int
                 ):
        super(Model, self).__init__()

        self.embedding = Embedding(channel_in=channel_in, window_size=window_size)

        # [Encoder]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu',batch_first=True, norm_first=True,)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=stack, norm=nn.LayerNorm(d_model))
        
        # [Decoder]
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=stack, norm=nn.LayerNorm(d_model))

        self.out = nn.Linear(d_model, 9)
    def forward(self, x, tgt):

        x = self.embedding(x)
        tgt = self.embedding(tgt)
        memory = self.encoder(x)
        out = self.decoder(tgt, memory)
        out = self.out(out)
        return out