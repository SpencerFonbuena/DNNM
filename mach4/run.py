import torch
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
import matplotlib.pyplot as plt
from module.hyperparameters import HyperParameters as hp
from module.transformer import Model
from module.loss import Myloss
from module.layers import run_encoder_decoder_inference
from sklearn.preprocessing import StandardScaler
'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

# [General Initialization]

# Set random seed
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Make us of GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
print(f'use device: {DEVICE}')


# Log on Weights and Biases

wandb.init(project='mach43', name='01')

#switch datasets depending on local or virtual run
if torch.cuda.is_available():
    path = '/root/DNNM/mach1/datasets/SPY_30mins_returns.txt'
else:
    path = 'DNNM/mach1/datasets/SPY_30mins_returns.txt'

# [End General Init]




#[Create and load the dataset]
def pipeline(batch_size, window_size, scaler):
    #create the datasets to be loaded
    train_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, scaler=scaler, mode='train', )
    test_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, scaler=scaler, mode='test', )
    inference_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, scaler=scaler, mode='inference', )



    #Load the data
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,pin_memory=True,  drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True)
    inference_dataloader = DataLoader(dataset=inference_dataset, batch_size=1, shuffle=False, num_workers=1,pin_memory=True)

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

    return train_dataloader, test_dataloader, inference_dataloader, d_channel


def network( heads, d_model, dropout, stack, d_hidden, channel_in, window_size, batch_size):
    net = Model(
                d_model=d_model,
                heads=heads,
                stack=stack,
                dim_feedforward=d_hidden,
                dropout=dropout,
                channel_in=channel_in,
                window_size=window_size,
                batch_size=batch_size
                ).to(DEVICE)
    return net
    
def train():



    train_dataloader, test_dataloader, _, d_channel = pipeline(batch_size=hp.batch_size, window_size=hp.window_size, pred_size=hp.pred_size, scaler=hp.scaler)
    
    net = network(d_model=hp.d_model,
                    heads=hp.heads,
                    stack=hp.stack,
                    d_hidden=hp.d_hidden,
                    dropout=hp.dropout,
                    channel_in=d_channel,
                    window_size=hp.window_size,
                    batch_size=hp.batch_size).to(DEVICE)
    # Create a loss function here using cross entropy loss
    loss_function = Myloss()

    #Select optimizer in an un-optimized way

    optimizer = optim.AdamW(net.parameters(), lr=hp.LR)

    # training function
    
    
    net.train()
    wandb.watch(net, log='all')
    for index in tqdm(range(hp.EPOCH)):
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pre = net(x)
            loss = loss_function(y_pre, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), .5)
            optimizer.step()

            if i % 200 == 0:
                print('Train Pred:',y_pre[0])
                print('Train Truth',y[0])

            wandb.log({'Loss': loss})
            wandb.log({'index': index})

        test(dataloader=test_dataloader, net=net, loss_function=loss_function)



def test(dataloader, net, loss_function):
    
    
    net.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre = net(x)
            
            if i % 100 == 0:
                print('Test Pred:',y_pre[0])
                print('Test Truth',y[0])
                

if __name__ == '__main__':
    train()