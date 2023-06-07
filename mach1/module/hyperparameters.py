class HyperParameters:

    # hyperparameter settings
    EPOCH = 1500
    BATCH_SIZE = 32
    WINDOW_SIZE = 120
    LR = .3
    d_model = 512
    d_hidden = 2048
    queries = 8 # Queries
    values = 8 # Values
    heads = 8 # Heads
    N = 8 # multi head attention layers
    dropout = 0.0
    split = .85
    optimizer_name = 'Adam'
    clip = .9
    p = .5
    fcnstack = 1