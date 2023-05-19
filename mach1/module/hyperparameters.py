class HyperParameters:

    # hyperparameter settings
    EPOCH = 225
    BATCH_SIZE = 32
    WINDOW_SIZE = 360
    LR = 5
    d_model = 512
    d_hidden = 1024
    q = 8
    v = 8
    h = 8
    N = 10
    dropout = 0.2
    split = .97
    pe = True # # The setting is in the twin towers score=pe score=channel has no pe by default
    mask = True # set the mask of score=input in the twin towers score=channel has no mask by default
    # optimizer selection
    optimizer_name = 'Adam'