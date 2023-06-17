# @Time    : 2021/01/22 25:16
# @Author  : SY.M
# @FileName: run.py

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
import torcheval
from torcheval.metrics import MulticlassAUPRC, MulticlassRecall





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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
print(f'use device: {DEVICE}')



# [Create WandB sweeps]

sweep_config = {
    'method': 'random',


    'metric': {
        'goal': 'minimize',
        'name': 'test_loss'
    },


    'parameters': {
    # [training hp]
    'learning_rate': {
        'values': hp.LR},
    'batch_size':{
        'values': hp.BATCH_SIZE},
    'window_size':{
        'values': hp.WINDOW_SIZE},

    # [architecture hp]
    'd_model':{
        'values': hp.d_model},
    'd_hidden':{
        'values': hp.d_hidden},
    'heads':{
        'values': hp.heads}, # Heads
    'stack':{
        'values': hp.N}, # multi head attention layers
    'stoch_p':{
    'values': hp.p}, # multi head attention layers
    'fcnstack':{
    'values': hp.fcnstack}, # multi head attention layers

    # [Regularizers]
    'dropout':{
        'values': hp.dropout},
    }
}

# [End Sweeps]

# Log on Weights and Biases

sweep_id = wandb.sweep(sweep_config, project='mach9 sweep')

#switch datasets depending on local or virtual run
if torch.cuda.is_available():
    path = '/root/DNNM/mach1/datasets/SPY_30mins.txt'
else:
    path = 'models/mach1/datasets/SPY_30mins.txt'

# [End General Init]

'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''


#[Create and load the dataset]
def pipeline(batch_size, window_size):
    #create the datasets to be loaded
    train_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, mode='train')

    test_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, mode='test')

    #create the samplers
    samplertrain = wrs(weights=train_dataset.trainsampleweights, num_samples=len(train_dataset), replacement=True)
    samplertest = wrs(weights=test_dataset.testsampleweights, num_samples=len(test_dataset), replacement=True)


    #Load the data
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=24,pin_memory=True ,sampler=samplertrain)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=24,pin_memory=True,sampler=samplertest)

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

    return train_dataloader, test_dataloader, d_input, d_channel, d_output


'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''


def network(d_input, d_channel, d_output, window_size, heads, d_model, dropout, stack, p, fcnstack, d_hidden):
    net = Transformer(window_size=window_size, 
                    timestep_in=d_input, channel_in=d_channel,
                    heads=heads,
                    d_model=d_model,
                    device=DEVICE,
                    dropout = dropout,
                    inner_size=d_hidden,
                    class_num=d_output, 
                    stack=stack, 
                    layers=[128, 256, 512], 
                    kss=[7, 5, 3], 
                    p=p, 
                    fcnstack=fcnstack).to(DEVICE)

    def hiddenPrints():
        # [Printing summaries]
        '''print (
            sum(param.numel() for param in net.parameters())
        )
        print(net)'''
        # [End Summaries]
        # Use this sleeper function if you want to look at the computational graph
        #print(net)
        #from torchviz import make_dot
        #traind, label = next(iter(train_dataloader))
        #y, _, _, _, _, _, _ = net(traind, 'train')
        #make_dot(y.mean(), show_attrs=True, show_saved=True,  params=dict(net.named_parameters())).render("GTN_torchviz", format="png")

    return net
    # [Place computational graph code here if desired]


def train(config=None):

    with wandb.init(config=config):

        config = wandb.config

        train_dataloader, test_dataloader, d_input, d_channel, d_output = pipeline(batch_size=config.batch_size, window_size=config.window_size)
        net = network(d_input=d_input, d_channel=d_channel, d_output=d_output, window_size=config.window_size, heads=config.heads, d_model=config.d_model, 
                      dropout=config.dropout, stack=config.stack, p=config.stoch_p, fcnstack=config.fcnstack, d_hidden=config.d_hidden)
        # Create a loss function here using cross entropy loss
        loss_function = Myloss()

        #Select optimizer in an un-optimized way
        if hp.optimizer_name == 'AdamW':
            optimizer = optim.AdamW(net.parameters(), lr=config.learning_rate) #
        # [End Training and Test Init]


        # [Begin Training and Testing]

        # training function

        net.train()
        wandb.watch(net, log='all', log_freq=10)
        for index in tqdm(range(hp.EPOCH)):
            for i, (x, y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                y_pre = net(x.to(DEVICE))
                loss = loss_function(y_pre, y.to(DEVICE))
                '''for i in range(len(list(net.parameters()))):
                    print(list(net.parameters())[i])'''
                loss.backward()
                optimizer.step()
                if i % 500 == 0:
                    wandb.log({'Loss': loss})
            wandb.log({'index': index})
            #validate training accuracy and test accuracy
            test(test_dataloader, 'test', net, loss_function)


# test function
def test(dataloader, net, loss_function, flag = str):
    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre = net(x)
            if i % 100 == 0:
                if flag == 'train':
                    max_indices = torch.argmax(y_pre, dim=-1)
                    print('Train',torch.cat([max_indices, y]))
                if flag == 'test':
                    max_indices = torch.argmax(y_pre, dim=-1)
                    print('test',torch.cat([max_indices, y]))
            test_loss = loss_function(y_pre, y.to(DEVICE))
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
            accuracy = correct / total * 100
            metricAUPRC = MulticlassAUPRC(num_classes=4).to(DEVICE)
            metricrecall = MulticlassRecall(num_classes=4).to(DEVICE)
            metricAUPRC.update(y_pre, y).to(DEVICE)  # Add predictions and targets
            auprc = metricAUPRC.compute().to(DEVICE)  # Get the computed Multiclass AUPRC

            metricrecall.update(y_pre, y).to(DEVICE)
            recall = metricrecall.compute().to(DEVICE)
        if flag == 'train':
            wandb.log({"Train acc": accuracy})
            wandb.log({"Train precision": auprc})
            wandb.log({"Train recall": recall})

        if flag == 'test':
            wandb.log({"Test acc": accuracy})
            wandb.log({"test_loss": test_loss})
            wandb.log({"Test precision": auprc})
            wandb.log({"Test recall": recall})

# [End Training and Testing]

'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

# [Save Model]

# [End Save Model]

# [Run the model]
if __name__ == '__main__':
    wandb.agent(sweep_id, train, count=20)
# [End experiment]