from sklearn.preprocessing import StandardScaler

class HyperParameters:
    # hyperparameter settings
    EPOCH = 10
    batch_size = 512
    window_size = 120
    LR = .000001
    d_model = 512
    d_hidden = 2048
    heads = 64 
    stack = 12
    dropout = 0.7
    split = .85
    clip = .9
    scaler = StandardScaler()