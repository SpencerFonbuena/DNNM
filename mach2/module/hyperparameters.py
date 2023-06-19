class HyperParameters:
    
        # hyperparameter settings
    EPOCH = 8
    BATCH_SIZE = [16]
    WINDOW_SIZE = [120]
    LR = [.00001,.00003,.00005,.00007,.00009]
    d_model = [512]
    d_hidden = [2048]
    queries = 8 # Queries
    values = 8 # Values
    heads = [64] # Heads
    N = [32] # multi head attention layers
    dropout = [0.0]
    split = .85
    optimizer_name = 'AdamW'
    clip = .9
    p = [0.5, 0.6, 0.7, 0.8,]
    fcnstack = [2]
    logs = 5