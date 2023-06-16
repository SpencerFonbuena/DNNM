# @Time    : 2021/01/22 25:16
# @Author  : SY.M
# @FileName: run.py

from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torchmetrics as tm
from torch.utils.data import DataLoader
from dataset_process.dataset_process import Create_Dataset
from torch.utils.data import WeightedRandomSampler as wrs
import torch.optim as optim
from time import time
from tqdm.auto import tqdm
import os
import numpy as np
import wandb
import random
import pandas as pd
import pytorch_lightning as pl
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall


from module.transformer import Transformer
from module.loss import Myloss
from module.hyperparameters import HyperParameters as hp


'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

# [General Initialization]

# Set random seed
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Make us of GPU
# [Create WandB sweeps]

config = {
    # [training hp]
    'EPOCH': hp.EPOCH,
    'BATCH_SIZE': hp.BATCH_SIZE,
    'WINDOW_SIZE': hp.WINDOW_SIZE,
    'LR': hp.LR,

    # [architecture hp]
    'd_model': hp.d_model,
    'd_hidden': hp.d_hidden,
    'queries': hp.queries, # Queries,
    'values': hp.values, # Values,
    'heads': hp.heads, # Heads,
    'N': hp.N, # multi head attention layers,
    'Conv Layers': 3,
    'Linear-Out Layers': 2,

    # [General]
    'split': hp.split,
    'optimizer_name': hp.optimizer_name,

    # [Regularizers]
    'dropout': hp.dropout,
    'clip': hp.clip,

}

'''sweep_config = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 
    }

}'''

# [End Sweeps]

# Log on Weights and Biases
'''wandb.init(
    project='garbage',
    name='changed validation set',
    #config=config
)'''

#switch datasets depending on local or virtual run
if torch.cuda.is_available():
    path = '/root/GTN/mach1/datasets/AAPL_1hour_correct.txt'
else:
    path = '/Users/spencerfonbuena/Documents/Python/Trading Models/models/mach1/datasets/AAPL_1hour.txt'


# Use this sleeper function if you want to look at the computational graph
#print(net)
#from torchviz import make_dot
#traind, label = next(iter(train_dataloader))
#y, _, _, _, _, _, _ = net(traind, 'train')
#make_dot(y.mean(), show_attrs=True, show_saved=True,  params=dict(net.named_parameters())).render("GTN_torchviz", format="png")

# [End General Init]

'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

class DataModule(pl.LightningDataModule):
    def setup(self, stage: str):
        return super().setup(stage)
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()

#[Create and load the dataset]

#create the datasets to be loaded
train_dataset = Create_Dataset(datafile=path, window_size=hp.WINDOW_SIZE, split=hp.split, mode='train')
val_dataset = Create_Dataset(datafile=path, window_size=hp.WINDOW_SIZE, split=hp.split, mode='validate')
test_dataset = Create_Dataset(datafile=path, window_size=hp.WINDOW_SIZE, split=hp.split, mode='test')

#create the samplers
samplertrain = wrs(weights=train_dataset.trainsampleweights, num_samples=len(train_dataset), replacement=True)
samplertest = wrs(weights=test_dataset.testsampleweights, num_samples=len(test_dataset), replacement=True)
samplertrainval = wrs(weights=val_dataset.trainvalsampleweights, num_samples=len(val_dataset), replacement=True)

#Load the data
train_dataloader = DataLoader(dataset=train_dataset, batch_size=hp.BATCH_SIZE, shuffle=False,  sampler=samplertrain)
validate_dataloader = DataLoader(dataset=val_dataset, batch_size=hp.BATCH_SIZE, shuffle=False,  sampler=samplertrainval)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=hp.BATCH_SIZE, shuffle=False,  sampler=samplertest)

DATA_LEN = train_dataset.training_len # Number of samples in the training set
d_input = train_dataset.input_len # number of time parts
d_channel = train_dataset.channel_len # feature dimension
d_output = train_dataset.output_len # classification category

# Dimension display
print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

# [End Dataset Init]


'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''


# [Initialize Training and Testing Procedures]

class PlClassifier(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()

        self.net = Transformer(window_size=hp.WINDOW_SIZE, timestep_in=d_input, channel_in=d_channel,
                heads=hp.heads,d_model=hp.d_model,qkpair=hp.queries,value_count=hp.values,
                inner_size=hp.d_hidden,class_num=d_output, stack=hp.N, layers=[128, 256, 512], kss=[7, 5, 3], p=hp.p, fcnstack=hp.fcnstack)
        
    def forward(self, x, stage):
        fcgtn = self.net(x, stage)
        return fcgtn
    
    

    def loss(self, logits, y):
        pre_loss_func = torch.nn.CrossEntropyLoss()
        y = y.type(torch.LongTensor)
        loss_func = pre_loss_func(logits, y)
        return loss_func

    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x, 'train')
        loss = self.loss(logits, y)

        if batch_idx % 100 == 0:
            accuracy = MulticlassAccuracy()
            precision = MulticlassPrecision()
            recall = MulticlassRecall()

            accuracy.update(logits, y)
            precision.update(logits, y)
            recall.update(logits, y)

            accuracy.compute
            precision.compute
            recall.compute 

        return loss
        
    
    def validation_step(self, trainval_batch, trainval_idx):
        x, y = trainval_batch
        logits = self.forward(x, 'test')
        loss = self.loss(logits, y)

        accuracy = MulticlassAccuracy()
        precision = MulticlassPrecision()
        recall = MulticlassRecall()

        accuracy.update(logits, y)
        precision.update(logits, y)
        recall.update(logits, y)

        accuracy.compute
        precision.compute
        recall.compute 
        
        return loss, accuracy, precision, recall
        
    
    
    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.net.parameters(), lr=hp.LR)

        return optimizer


model = PlClassifier()
trainer = pl.Trainer()

trainer.fit(model, train_dataloader, test_dataloader)

# [Run the model]
if __name__ == '__main__':
    pass
# [End experiment]