import torch
import torch.nn as nn



class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3) # (64, 120, 512) @ (512, 512*3) | This creates 3x the number of features. 3x because there is 1 query + 1 key + 1 value = 3 representations of our original dataset
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
    def forward(self, data):
        B, S, _ = data.shape
        qkv = self.qkv(data) # (64, 120, 1536) => (64 examples, 120 tokens (in the form of timesteps), 1536 features)
        qkv = qkv.reshape(B, S, 3, -1) # (64, 120, 3, 512) => (64 examples, 120 tokens, 3 attributes of each token (QKV), 512 features)
        qkv = qkv.reshape(B, S, 3, self.num_heads, -1) # (64, 120, 3, 8, 64) => (64 examples, 120 tokens, 3 attributes of each token, 8 versions of each attribute, 64 features )
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, 64, 8, 120, 64)
        q, k, v = qkv.reshape(3, B*self.num_heads, 120, -1).unbind(0) # each q, k, v has dimensions 3x(512,120,64) => 3 of (examples, tokens, features)

        attn = (q * self.scale) @ k.transpose(-2, -1) # (512, 120, 120)

        attn = attn.softmax(dim=-1)

        x = (attn @ v) # (512, 120, 64)
        x = x.view(B, self.num_heads, S, -1) # (64,8,120,64)
        x = x.permute(0, 2, 1, 3) # (64, 120, 8, 64)
        x = x.reshape(B, S, -1) # (64, 120, 512)


class Ns_Transformer(nn.Module):
    def __init__(self,
                 num_heads: int = 8,
                 dropout = float,
                 d_model = int,
                 rbn = int,) -> None:
        super().__init__()
        #MHA
        self.num_heads = num_heads
        self.qkv = nn.Linear(d_model, d_model * 3) # (64, 120, 512) @ (512, 512*3) | This creates 3x the number of features. 3x because there is 1 query + 1 key + 1 value = 3 representations of our original dataset
        head_dim = d_model // num_heads
        self.scale = head_dim**-0.5

        self.drop_mha = nn.Dropout(dropout)

        #FFN
        self.feedforward = nn.Sequential(
        nn.Linear(d_model, rbn),
        nn.GELU(),  
        nn.Linear(rbn, d_model),
        nn.Dropout(dropout)
        )

    def forward(self, data, tau, delta):

        # [MHA]
        B, S, _ = data.shape
        qkv = self.qkv(data) # (64, 120, 1536) => (64 examples, 120 tokens (in the form of timesteps), 1536 features)
        qkv = qkv.reshape(B, S, 3, -1) # (64, 120, 3, 512) => (64 examples, 120 tokens, 3 attributes of each token (QKV), 512 features)
        qkv = qkv.reshape(B, S, 3, self.num_heads, -1) # (64, 120, 3, 8, 64) => (64 examples, 120 tokens, 3 attributes of each token, 8 versions of each attribute, 64 features )
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, 64, 8, 120, 64)
        q, k, v = qkv.reshape(3, B*self.num_heads, 120, -1).unbind(0) # each q, k, v has dimensions 3x(512,120,64) => 3 of (examples, tokens, features)

        # [Non-Stationary]
        tau = tau.unsqueeze(1).unsqueeze(1)
        delta = delta.unsqueeze(1).unsqueeze(1)

        # [MHA]
        attn = (q * self.scale) @ k.transpose(-2, -1) * tau + delta # (512, 120, 120) ## I may want to look at the scaling, and where it is implemented to be sure that nothing weird is happening
        attn = attn.softmax(dim=-1)
        x = (attn @ v) # (512, 120, 64)
        x = x.view(B, self.num_heads, S, -1) # (64,8,120,64)
        x = x.permute(0, 2, 1, 3) # (64, 120, 8, 64)
        x = x.reshape(B, S, -1) # (64, 120, 512)
        x = self.drop_mha(x)
        # [Feed-Forward]
        x = self.feedforward(x)

        return x
    

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1): #hidden_layers - 1 because of we put a layers above. range is not 0 indexed, but the reason we defined the list was to take in a specific dimension
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)] # making the output of the linear layer have expected dimensions
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x O

        return y



class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
    

def mask(dim1: int, dim2: int):
 
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
