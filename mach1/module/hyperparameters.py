class HyperParameters:
    # hyperparameter settings
    EPOCH = 8
    BATCH_SIZE = [16,32,64]
    WINDOW_SIZE = [120,150,180,210,240]
    LR = [.00001,.0001,.001, .00003,.0003, .00005,.0005, .00007,.0007, .00009,.0009,]
    d_model = [512]
    d_hidden = [2048]
    queries = 8 # Queries
    values = 8 # Values
    heads = [8,16,32,64,128] # Heads
    N = [8,16,32,64,96] # multi head attention layers
    dropout = [0.2,0.3,0.4]
    split = .85
    optimizer_name = 'AdamW'
    clip = .9
    p = [0.5, 0.6, 0.7, 0.8, 0.9]
    fcnstack = [2,4]
    logs = 5