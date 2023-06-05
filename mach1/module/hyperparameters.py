class HyperParameters:

    # hyperparameter settings
    EPOCH = 1500
    BATCH_SIZE = 9
    WINDOW_SIZE = 5
    LR = .0003
    d_model = 512
    d_hidden = 2048
    queries = 8 # Queries
    values = 8 # Values
    heads = 8 # Heads
    N = 1 # multi head attention layers
    dropout = 0.0
    split = .85
    optimizer_name = 'Adam'
    clip = .9