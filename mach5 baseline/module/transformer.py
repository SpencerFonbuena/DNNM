import torch
import torch.nn as nn
from module.embedding import Embedding
from module.layers import Projector, Ns_Transformer
import torchvision.ops.stochastic_depth as std
from module.embedding import Embedding
from sklearn.preprocessing import StandardScaler
import torchvision.ops.stochastic_depth as std
from module.hyperparameters import HyperParameters as hp

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
                 pred_size = int
                 ):
        super(Model, self).__init__()
        
        self.stack = stack


        self.sourceembedding = Embedding(channel_in=channel_in, window_size=window_size)
        self.tgtembedding = Embedding(channel_in=1, window_size=window_size)


        # [Encoder]
        
        self.encoder_tower = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True, 
            norm_first=True,) for _ in range(stack)

        ])

        # [Decoder]
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True, norm_first=True,)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=stack, norm=nn.LayerNorm(d_model))

        '''self.decoder_tower = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            ) for _ in range(stack)
        ])'''

        self.out = nn.Linear(d_model, 1)


    def forward(self, x, tgt, mask):
        
        memory = self.sourceembedding(x, input='source')
        tgt = self.tgtembedding(tgt, input='target')

        for i, encoder in enumerate(self.encoder_tower):
            memory = std(memory, (i/self.stack) * hp.p, 'batch')
            memory = encoder(memory)
        
        out = self.decoder(tgt, memory, mask)
        
        '''
        for i, decoder in enumerate(self.decoder_tower):
            tgt = std(tgt, (i/self.stack) * hp.p, 'batch')
            tgt = encoder(tgt, memory, mask)
        '''
        out = self.out(out)

        return out
    
    