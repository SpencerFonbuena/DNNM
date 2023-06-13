class HyperParameters:

    # hyperparameter settings
    EPOCH = 1500
    BATCH_SIZE = 36
    WINDOW_SIZE = 120
    LR = .0003
    d_model = 512
    d_hidden = 2048
    queries = 8 # Queries
    values = 8 # Values
    heads = 16 # Heads
    N = 24 # multi head attention layers
    dropout = 0.0
    split = .85
    optimizer_name = 'AdamW'
    clip = .9
    p = .01
    fcnstack = 2
    logs = 5