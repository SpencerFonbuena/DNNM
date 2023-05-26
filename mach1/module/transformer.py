from torch.nn import Module
import torch
from torch.nn import ModuleList
from module.encoder import Encoder
import math
import torch.nn.functional as F
import numpy as np
import random
from module.hyperparameters import HyperParameters as hp

from module.embedding import Embedding
from module.encoder import Encoder
from module.gate import Gating

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
#print(f'use device: {DEVICE}')

class Transformer(Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def forward(self, x):
        x = Embedding(x)
        x = Encoder(x)
        x = Gating(x)
        return
