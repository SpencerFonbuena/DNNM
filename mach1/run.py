# @Time    : 2021/01/22 25:16
# @Author  : SY.M
# @FileName: run.py

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
    project='mach test',
    name='test'

)


if torch.cuda.is_available():
    path = '/root/GTN/mach1/datasets/AAPL_1hour_expand.txt'
else:
    path = 'gtn/mach1/datasets/AAPL_1hour_expand.txt'




test_interval = 2  # Test interval unit: epoch


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

# Dimension display
print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

# Create a Transformer model
net = Transformer(d_model=hp.d_model, d_timestep=d_input, d_channel=d_channel, d_output=d_output, d_hidden=hp.d_hidden,
                  q=hp.q, v=hp.v, h=hp.h, N=hp.N, dropout=hp.dropout, pe=hp.pe, mask=hp.mask, device=DEVICE).to(DEVICE)


print(net)

# [Beginning creating and visualizing computation graph]

#from torchviz import make_dot
#traind, label = next(iter(train_dataloader))
#y, _, _, _, _, _, _ = net(traind, 'train')
#make_dot(y.mean(), show_attrs=True, show_saved=True,  params=dict(net.named_parameters())).render("GTN_torchviz", format="png")

#[End of creating and visualizing computation graph]

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


# training function
def train():
    net.train()
    wandb.watch(net, log='all')
    for index in tqdm(range(hp.EPOCH)):
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pre = net(x.to(DEVICE), 'train')
            loss = loss_function(y_pre, y.to(DEVICE))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            wandb.log({'loss': loss})
            wandb.log({'index': index})
        #validate training accuracy and test accuracy
        test(validate_dataloader, 'train')
        test(test_dataloader, 'test')
        

# test function
def test(dataloader, flag = str):
    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
            accuracy = correct / total * 100
            if i % 500 == 0:
                print(net.fgate)
        if flag == 'train':
            wandb.log({"Train acc": accuracy})
        if flag == 'test':
            wandb.log({"Test acc": accuracy})



if __name__ == '__main__':
    train()
