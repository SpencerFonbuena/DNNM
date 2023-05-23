import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_process.dataset_process import Create_Dataset
from tqdm import tqdm, trange

from torch.nn import Module
import torch
from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader
from dataset_process.dataset_process import Create_Dataset
from torch.utils.data import WeightedRandomSampler as wrs
import torch.optim as optim
from time import time
import os
import numpy as np
import wandb
import random

from module.transformer import Transformer
from module.loss import Myloss
from module.hyperparameters import HyperParameters as hp
from baselines.FCN import ConvNet

wandb.init(
    project='mach1 1hour',
    name='test'

)

if torch.cuda.is_available():
    path = '/root/GTN/mach1/datasets/AAPL_1hour_expand.txt'
else:
    path = 'gtn/mach1/datasets/AAPL_1hour_expand.txt'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
print(f'use device: {DEVICE}')

path = 'gtn/mach1/datasets/AAPL_1hour_expand.txt'
#create the dataset to be loaded
train_dataset = Create_Dataset(datafile=path, window_size=hp.WINDOW_SIZE, split=hp.split, mode='train')
val_dataset = Create_Dataset(datafile=path, window_size=hp.WINDOW_SIZE, split=hp.split, mode='validate')
test_dataset = Create_Dataset(datafile=path, window_size=hp.WINDOW_SIZE, split=hp.split, mode='test')

#create the sampler
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

fcnmodel = ConvNet(26,4)

print(fcnmodel)


def train(dataloader_train: DataLoader,
          dataloader_test: DataLoader,
          device: str,
          model: nn.Module,
          epochs: int,
          learning_rate: float,
          save: bool):

    optimiser = optim.Adam(model.parameters(),lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss().requires_grad_(True)
    history = []
    wandb.watch(model, log='all')
    epoch_bar = trange(epochs)
    for epoch in epoch_bar:

        # train
        model.train()
        for batch,data in enumerate(dataloader_train):
            x,y = data
            x,y = x.to(device),(y.view(-1)).to(device)
            optimiser.zero_grad()
            out = model(x)
            loss = criterion(out.type(torch.FloatTensor), y.type(torch.LongTensor).to(DEVICE))
            loss.requires_grad = True
            wandb.log({'loss': loss})
            loss.backward()
            optimiser.step()

        # test
        running_loss = 0
        running_acc  = 0
        for batch,data in enumerate(dataloader_test):
            x,y = data
            x,y = x.to(device),(y.view(-1)).to(device)
            outs = model(x)

            test_acc = ((torch.argmax(outs,1))== y).cpu().detach().numpy().sum()/len(y)*100
            test_loss = F.cross_entropy(outs,y).item()
            running_acc  += test_acc*x.size(0)
            running_loss += test_loss*x.size(0)

        test_size = len(dataloader_test.dataset)
        test_acc = running_acc/test_size
        test_loss = running_loss/test_size
        epoch_bar.set_description('acc={0:.2f}%\tcross entropy={1:.4f}'
                                  .format(test_acc, test_loss))

        history.append((test_acc,test_loss))

    if save:
        #save
        pass

    return model,history

train(train_dataloader, test_dataloader, DEVICE, fcnmodel, 100, .0003, False)