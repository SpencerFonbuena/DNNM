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
from torcheval.metrics import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy




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
        'goal': 'maximize',
        'name': 'test_acc'
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

sweep_id = wandb.sweep(sweep_config, project='mach9 garbage')

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
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True ,sampler=samplertrain)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True,sampler=samplertest)

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
                    timestep_in=d_input, 
                    channel_in=d_channel,
                    heads=heads,
                    d_model=d_model,
                    device=DEVICE,
                    dropout = dropout,
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
                      dropout=config.dropout, stack=config.stack, p=config.stoch_p, fcnstack=config.fcnstack, d_hidden=config.d_hidden).to(DEVICE)
        # Create a loss function here using cross entropy loss
        loss_function = Myloss()

        #Select optimizer in an un-optimized way
        if hp.optimizer_name == 'AdamW':
            optimizer = optim.AdamW(net.parameters(), lr=config.learning_rate)

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
                optimizer.step()
                if i % 1000 == 0:
                    accuracy = MulticlassAccuracy().to(DEVICE)
                    specacc = MulticlassAccuracy(average=None, num_classes=4).to(DEVICE)
                    precision = MulticlassPrecision().to(DEVICE)
                    recall = MulticlassRecall().to(DEVICE)

                    accuracy.update(y_pre, y)
                    precision.update(y_pre, y)
                    recall.update(y_pre, y)
                    specacc.update(y_pre, y)

                    accuracy.compute()
                    precision.compute()
                    recall.compute()
                    specacc.compute()

                    wandb.log({"test_acc": accuracy.accuracy()})
                    wandb.log({"Test precision": precision.precision()})
                    wandb.log({"Test recall": recall.recall()})
                    wandb.log({'Loss': loss})
                    wandb.log({'index': index})
                    print(specacc.specacc())
                #validate training accuracy and test accuracy
            test(dataloader=test_dataloader, net=net, loss_function=loss_function)


# test function
def test(dataloader, net, loss_function):
    
    correct = 0
    total = 0  
    print('got through')

    net.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre = net(x)
            test_loss = loss_function(y_pre, y)
            
            accuracy = MulticlassAccuracy().to(DEVICE)
            specacc = MulticlassAccuracy(average=None, num_classes=4).to(DEVICE)
            precision = MulticlassPrecision().to(DEVICE)
            recall = MulticlassRecall().to(DEVICE)

            accuracy.update(y_pre, y)
            precision.update(y_pre, y)
            recall.update(y_pre, y)
            specacc.update(y_pre, y)

            accuracy.compute()
            precision.compute()
            recall.compute()
            specacc.compute()

            wandb.log({"test_acc": accuracy.accuracy()})
            wandb.log({"test_loss": test_loss})
            wandb.log({"Test precision": precision.precision()})
            wandb.log({"Test recall": recall.recall()})
            print(specacc.specacc())

# [End Training and Testing]

'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''

# [Save Model]

# [End Save Model]

# [Run the model]
if __name__ == '__main__':
    wandb.agent(sweep_id, train, count=200)
# [End experiment]