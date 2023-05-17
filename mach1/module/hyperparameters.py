class HyperParameters():
    def init(self):
        # hyperparameter settings
        self.EPOCH = 225
        self.BATCH_SIZE = 16
        self.LR = 1e-4

        self.d_model = 512
        self.d_hidden = 1024
        self.q = 16
        self.v = 16
        self.h = 16
        self.N = 32
        self.dropout = 0.2
        self.pe = True # # The setting is in the twin towers score=pe score=channel has no pe by default
        self.mask = True # set the mask of score=input in the twin towers score=channel has no mask by default
        # optimizer selection
        self.optimizer_name = 'Adam'