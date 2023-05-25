class HyperParameters:

    # hyperparameter settings
    EPOCH = 1500
    BATCH_SIZE = 64
    WINDOW_SIZE = 240
    LR = .0003
    d_model = 512
    d_hidden = 1024
    q = 8 # Queries
    v = 8 # Values
    h = 8 # Heads
    N = 2 # multi head attention layers
    dropout = 0.0
    split = .85
    pe = True # # The setting is in the twin towers score=pe score=channel has no pe by default
    mask = True # set the mask of score=input in the twin towers score=channel has no mask by default
    # optimizer selection
    optimizer_name = 'Adam'