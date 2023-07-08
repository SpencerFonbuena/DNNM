class HyperParameters:
    # hyperparameter settings
    EPOCH = 20
    batch_size = 64
    window_size = 120
    learning_rate = .00001
    d_model = 512
    d_hidden = 2048
    queries = 8 # Queries
    values = 8 # Values
    heads = 64 # Head
    stack = 8 # multi head attention layer
    dropout = 0.7
    split = .85
    optimizer_name = 'AdamW'
    clip = .9
    stoch_p = 0.3
    fcnstack = 2
    logs = 5
    pred_size = 30