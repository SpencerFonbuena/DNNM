class HyperParameters:

    # hyperparameter settings
    EPOCH = 1500
    BATCH_SIZE = 36
    WINDOW_SIZE = 120
    LR = .0003
    d_model = 512
    d_hidden = 2048
    queries = 16 # Queries
    values = 16 # Values
    heads = 32 # Heads
    N = 24 # multi head attention layers
    dropout = 0.2
    split = .85
    optimizer_name = 'AdamW'
    clip = .9
    p = .5
    fcnstack = 2
    logs = 5