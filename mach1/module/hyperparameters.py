class HyperParameters:

    # hyperparameter settings
    EPOCH = 225
    BATCH_SIZE = 64
    WINDOW_SIZE = 360
    LR = 3e-4
    d_model = 512
    d_hidden = 1024
    q = 16
    v = 16
    h = 16
    N = 16
    dropout = 0.2
    split = .97
    pe = True # # The setting is in the twin towers score=pe score=channel has no pe by default
    mask = True # set the mask of score=input in the twin towers score=channel has no mask by default
    # optimizer selection
    optimizer_name = 'Adam'