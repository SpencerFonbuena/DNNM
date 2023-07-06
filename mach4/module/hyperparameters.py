from sklearn.preprocessing import StandardScaler

class HyperParameters:
    # hyperparameter settings
    EPOCH = 10
    batch_size = 150
    window_size = 120
    LR = .00001 #.00001,.00003,.00005,.00007,.00009
    d_model = 512
    d_hidden = 2048
    queries = 8 # Queries
    values = 8 # Values
    heads = 64 # Heads
    stack = 24 # multi head attention layers
    dropout = 0.7
    split = .85
    optimizer_name = 'AdamW'
    clip = .9
    p = 0.3
    fcnstack = 2
    logs = 5
    pred_size = 60
    scaler = StandardScaler()