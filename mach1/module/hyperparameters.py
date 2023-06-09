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
    heads = 8 # Heads
    N = 8 # multi head attention layers
    dropout = 0.3
    split = .85
    optimizer_name = 'AdamW'
    clip = .9
    p = .5
    fcnstack = 2
    logs = 5