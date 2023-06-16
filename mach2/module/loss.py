import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import random

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


class Myloss(pl.LightningModule):
    def __init__(self):
        super(Myloss, self).__init__()
        self.loss_function = CrossEntropyLoss()

    def forward(self, y_pre, y_true):
        y_true = y_true.type(torch.LongTensor)
        loss = self.loss_function(y_pre, y_true)

        return loss