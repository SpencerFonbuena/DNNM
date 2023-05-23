class HyperParameters:

    # hyperparameter settings
    EPOCH = 225
    BATCH_SIZE = 1
    WINDOW_SIZE = 16
    LR = .0003
    d_model = 512
    d_hidden = 1024
    q = 8
    v = 8
    h = 8
    N = 1
    dropout = 0.0
    split = .85
    pe = True # # The setting is in the twin towers score=pe score=channel has no pe by default
    mask = True # set the mask of score=input in the twin towers score=channel has no mask by default
    # optimizer selection
    optimizer_name = 'Adam'