from sklearn.preprocessing import StandardScaler

class HyperParameters:
    # hyperparameter settings
    EPOCH = 10
    batch_size = 128
    window_size = 256
    LR = .00001
    d_model = 512
    d_hidden = 2048
    heads = 64 
    stack = 8 
    dropout = 0.7
    split = .85
    clip = .9
    scaler = StandardScaler()