class HyperParameters:
    # hyperparameter settings
    EPOCH = 25
    BATCH_SIZE = [64]
    WINDOW_SIZE = [240]
    LR = [.00003]#[.00001,.00003,.00005,.00007,.00009]
    d_model = [512]
    d_hidden = [2048]
    queries = 8 # Queries
    values = 8 # Values
    heads = [64] # Heads
    N = [64] # multi head attention layers
    dropout = [0.0]
    split = .85
    optimizer_name = 'AdamW'
    clip = .9
    p = [.7] # [0.5, 0.6, 0.7, 0.8,]
    fcnstack = [2]
    logs = 5