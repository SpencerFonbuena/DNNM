import torch
import torch.nn as nn
from module.embedding import Embedding
from module.layers import Projector, Ns_Transformer
import torchvision.ops.stochastic_depth as std

class Model(nn.Module):
    """
    Non-stationary Transformer
    """
    def __init__(self,
                 enc_in,
                 seq_len,
                 p_hidden_dims,
                 p_hidden_layers,
                 pred_len,
                 label_len,
                 d_model,
                 heads,
                 dropout,
                 device,
                 rbn,
                 stack
                 ):
        super(Model, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.label_len = label_len

        # [Destationary factors]
        self.tau_learner   = Projector(enc_in=enc_in, seq_len=seq_len, hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=enc_in, seq_len=seq_len, hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers, output_dim=seq_len)

        # [Encoder]
        self.encoder = nn.ModuleList ({Ns_Transformer(
                 d_model=d_model,
                 num_heads=heads,
                 dropout=dropout,
                 rbn=rbn
            )} for _ in range(stack)
        ) 

        # [Decoder]
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=heads)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=stack)


    def forward(self, x_enc, tgt):

        x_raw = x_enc.clone().detach() # Use the raw x to later approximate de-stationary features

        # [Normalization]
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E |
        x_enc = x_enc - mean_enc # Subtract Mean
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        x_enc = x_enc / std_enc # Divide by standard deviation
        #x_dec_new = torch.cat([x_enc[:, -self.label_len: , :], torch.zeros_like(x_dec[:, -self.pred_len:, :])], dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()     # B x S x E, B x 1 x E -> B x 1, positive scalar    
        delta = self.delta_learner(x_raw, mean_enc)      # B x S x E, B x 1 x E -> B x S

        # [Encoding]
        for i, encoder in enumerate(self.encoder):
            x_encoder = std(x_encoder, (i/self.stack) * self.p, 'batch')
            x_encoder = encoder(x_encoder, tau, delta)


        dec_out = self.decoder(tgt,x_encoder)

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc

        return dec_out