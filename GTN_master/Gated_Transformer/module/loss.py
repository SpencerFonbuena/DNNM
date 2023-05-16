from torch.nn import Module
import torch
from torch.nn import CrossEntropyLoss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
#print(f'use device: {DEVICE}')

class Myloss(Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.loss_function = CrossEntropyLoss()

    def forward(self, y_pre, y_true):
        y_true = y_true.type(torch.LongTensor).to(DEVICE)
        loss = self.loss_function(y_pre, y_true)

        return loss