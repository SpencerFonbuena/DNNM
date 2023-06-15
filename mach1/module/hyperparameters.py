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
    heads = 96 # Heads
    N = 96 # multi head attention layers
    dropout = 0.2
    split = .85
    optimizer_name = 'AdamW'
    clip = .9
    p = .9
    fcnstack = 2
    logs = 5