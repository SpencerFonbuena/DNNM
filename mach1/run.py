# @Time    : 2021/01/22 25:16
# @Author  : SY.M
# @FileName: run.py

import torch
import torchmetrics as tm
from torch.utils.data import DataLoader
from dataset_process.dataset_process import Create_Dataset
import torch.optim as optim
from time import time
from tqdm.auto import tqdm
import os
import numpy as np
import wandb
import random

from module.transformer import Transformer
from module.loss import Myloss
from module.hyperparameters import HyperParameters as hp

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
print(f'use device: {DEVICE}')

wandb.init(
    project='test vmbaseline',
    name='verify random seed'
)

#path = 'gtn/mach1/AAPL_1hour_expand.txt'
path = '/root/GTN/GTN_master/AAPL_1hour_expand.txt'

test_interval = 2  # Test interval unit: epoch
draw_key = 1  # Greater than or equal to draw_key will save the image
file_name = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]  # get file name



train_dataset = Create_Dataset(datafile=path, window_size=hp.WINDOW_SIZE, split=hp.split, mode='train')
test_dataset = Create_Dataset(datafile=path, window_size=hp.WINDOW_SIZE, split=hp.split, mode='test')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=hp.BATCH_SIZE, shuffle=False, num_workers=2)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=hp.BATCH_SIZE, shuffle=False, num_workers=2)

DATA_LEN = train_dataset.training_len # Number of samples in the training set
d_input = train_dataset.input_len # number of time parts
d_channel = train_dataset.channel_len # feature dimension
d_output = train_dataset.output_len # classification category

# Dimension display
print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

# Create a Transformer model
net = Transformer(d_model=hp.d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=hp.d_hidden,
                  q=hp.q, v=hp.v, h=hp.h, N=hp.N, dropout=hp.dropout, pe=hp.pe, mask=hp.mask, device=DEVICE).to(DEVICE)

#print the model summary
print(net)
#Print the number of parameters
#print(sum([param.nelement() for param in net.parameters()])) (Currently there are: 101M parameters)

# Create a loss function here using cross entropy loss
loss_function = Myloss()

if hp.optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=hp.LR)

# Used to record the accuracy rate change
correct_on_train = []
correct_on_test = []
# Used to record loss changes
loss_list = []
time_cost = 0


# test function
def test(dataloader, flag=str):
    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
            accuracy = correct / total * 100
        if flag == 'train':
            wandb.log({"training acc": accuracy})
        if flag == 'test':
            wandb.log({"test acc": accuracy})

# training function
def train():
    net.train()
    wandb.watch(net, log='all')
    for index in tqdm(range(hp.EPOCH)):
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')
            loss = loss_function(y_pre, y.to(DEVICE))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            wandb.log({'loss': loss})
        #validate training accuracy and test accuracy
        if ((index + 1) % test_interval) == 0:
            test(train_dataloader, 'train_set')
            test(test_dataloader, 'test_set')




if __name__ == '__main__':
    train()
