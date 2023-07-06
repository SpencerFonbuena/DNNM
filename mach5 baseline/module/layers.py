import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU

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


    

def mask(dim1: int, dim2: int):
 
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)



def run_encoder_decoder_inference(
    model: nn.Module, 
    src: torch.Tensor, 
    forecast_window: int,
    batch_size: int,
    batch_first: bool=True,
    scaler: object=None
    ) -> torch.Tensor:

    """
    NB! This function is currently only tested on models that work with 
    batch_first = False
    
    This function is for encoder-decoder type models in which the decoder requires
    an input, tgt, which - during training - is the target sequence. During inference,
    the values of tgt are unknown, and the values therefore have to be generated
    iteratively.  
    
    This function returns a prediction of length forecast_window for each batch in src
    
    NB! If you want the inference to be done without gradient calculation, 
    make sure to call this function inside the context manager torch.no_grad like:
    with torch.no_grad:
        run_encoder_decoder_inference()
        
    The context manager is intentionally not called inside this function to make
    it usable in cases where the function is used to compute loss that must be 
    backpropagated during training and gradient calculation hence is required.
    
    If use_predicted_tgt = True:
    To begin with, tgt is equal to the last value of src. Then, the last element
    in the model's prediction is iteratively concatenated with tgt, such that 
    at each step in the for-loop, tgt's size increases by 1. Finally, tgt will
    have the correct length (target sequence length) and the final prediction
    will be produced and returned.
    
    Args:
        model: An encoder-decoder type model where the decoder requires
               target values as input. Should be set to evaluation mode before 
               passed to this function.
               
        src: The input to the model
        
        forecast_horizon: The desired length of the model's output, e.g. 58 if you
                         want to predict the next 58 hours of FCR prices.
                           
        batch_size: batch size
        
        batch_first: If true, the shape of the model input should be 
                     [batch size, input sequence length, number of features].
                     If false, [input sequence length, batch size, number of features]
    
    """

    # Dimension of a batched model input that contains the target sequence values
    #src = src.permute(1,0,2)
    target_seq_dim = 0 if batch_first == False else 1 # This is saying that we want to predict sequence, and if the batch is first, then the sequence dimension is dimension 1, if not, it's 0

    # Take the last value of the target variable in all batches in src and make it tgt
    # as per the Influenza paper
    tgt = src[-1, :, 0] if batch_first == False else src[:, -1, 0] # shape [1, batch_size, 1] | Notice where the -1 is. It is saying, let's take the last value in the sequence, which is where we will start our forecasting


    # [I don't think this pertains to me]
     # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_size == 1 and batch_first == False:
        tgt = tgt.unsqueeze(0).unsqueeze(0) # change from [1] to [1, 1, 1]

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_first == False and batch_size > 1:
        tgt = tgt.unsqueeze(0).unsqueeze(-1)
    # [I don't think this pertains to me]
    tgt = tgt.unsqueeze(0).unsqueeze(0)
    
    # Iteratively concatenate tgt with the first element in the prediction
    for _ in range(forecast_window-1):
        
        # Create masks
        dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

        tgt_mask = mask(
            dim1=dim_a,
            dim2=dim_a,
            ).to(DEVICE)


        # Make prediction

        prediction = model(src, tgt, torch.zeros_like(tgt_mask))
        
         
        # If statement simply makes sure that the predicted value is 
        # extracted and reshaped correctly
        if batch_first == False:

            # Obtain the predicted value at t+1 where t is the last time step 
            # represented in tgt
            last_predicted_value = prediction[-1, :, :] 

            # Reshape from [batch_size, 1] --> [1, batch_size, 1]
            last_predicted_value = last_predicted_value.unsqueeze(0)

        else:

            # Obtain predicted value
            last_predicted_value = prediction[:, -1, :]

            # Reshape from [batch_size, 1] --> [batch_size, 1, 1]
            last_predicted_value = last_predicted_value.unsqueeze(-1)
            

        # Detach the predicted element from the graph and concatenate with 
        # tgt in dimension 1 or 0
        tgt = torch.cat((tgt, last_predicted_value.detach()), target_seq_dim)
        print(tgt)


    dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]



    tgt_mask = mask(
        dim1=dim_a,
        dim2=dim_a,
        ).to(DEVICE)


    
    final_prediction = model(src, tgt, tgt_mask)

    return final_prediction


def Scaler():
    scaler = StandardScaler()
    return scaler