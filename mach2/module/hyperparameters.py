class HyperParameters:
    # hyperparameter settings
    EPOCH = 8
    BATCH_SIZE = 16
    WINDOW_SIZE = 120
    LR = .00003
    d_model = 512
    d_hidden = 2048
    queries = 8 # Queries
    values = 8 # Values
    heads = 16 # Heads
    N = 8 # multi head attention layers
    dropout = 0.0
    split = .85
    optimizer_name = 'AdamW'
    clip = .9
    p = .7
    fcnstack = 2
    logs = 5