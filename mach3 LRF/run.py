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
from torcheval.metrics import  MeanSquaredError




from cross_models.loss import Myloss
from cross_models.hyperparameters import HyperParameters as hp
from cross_models.cross_former import Crossformer
from utils.metrics import metric


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
    'pred_size':{
    'values': hp.pred_size}, # multi head attention layers
    'seg_len':{
    'values': hp.seg_len}, # multi head attention layers

    # [Regularizers]
    'dropout':{
        'values': hp.dropout},
    }
}

# [End Sweeps]

# Log on Weights and Biases

sweep_id = wandb.sweep(sweep_config, project='mach29 forecast ')

#switch datasets depending on local or virtual run
if torch.cuda.is_available():
    path = '/root/DNNM/mach1/datasets/SPY_30mins_gaus.txt'
else:
    path = 'DNNM/mach1/datasets/SPY_30mins_gaus.txt'

# [End General Init]




#[Create and load the dataset]
def pipeline(batch_size, window_size, pred_size):
    #create the datasets to be loaded
    train_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, mode='train', pred_size=pred_size)
    test_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, mode='test', pred_size=pred_size)



    #Load the data
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12,pin_memory=True,  drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=12,pin_memory=True)

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

    return train_dataloader, test_dataloader, d_channel



def network( d_channel, window_size, heads, d_model, dropout, stack, d_hidden, pred_size, seg_len):
    net = Crossformer(data_dim=d_channel,
                    in_len=window_size,
                    out_len=pred_size,
                    seg_len=seg_len,
                    d_model=d_model,
                    n_heads=heads,
                    e_layers=stack,
                    d_ff=d_hidden,
                    dropout=dropout,
                    device=DEVICE,
                    ).to(DEVICE)

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



def train(config=None):

    with wandb.init(config=config):

        config = wandb.config

        train_dataloader, test_dataloader, d_channel = pipeline(batch_size=config.batch_size, window_size=config.window_size, pred_size=config.pred_size)
        
        net = network( d_channel=d_channel, window_size=config.window_size, heads=config.heads, d_model=config.d_model, 
                      dropout=config.dropout, stack=config.stack, d_hidden=config.d_hidden, pred_size=config.pred_size, seg_len=config.seg_len).to(DEVICE)
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
                torch.nn.utils.clip_grad_norm_(net.parameters(), .5)
                optimizer.step()
                
    
                wandb.log({'Loss': loss})
                wandb.log({'index': index})
            mae,mse,rmse,mape,mspe = metric(y_pre.cpu().detach().numpy(), y.cpu().detach().numpy())
                
            print(mae,mse,rmse,mape,mspe)

            
            
            

            #wandb.log({"train_mse": mse})
            
            test(dataloader=test_dataloader, net=net, loss_function=loss_function)


# test function
def test(dataloader, net, loss_function):
    
    
    net.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre = net(x)
        mae,mse,rmse,mape,mspe = metric(y_pre.cpu().detach().numpy(), y.cpu().detach().numpy())
                
        print(mae,mse,rmse,mape,mspe)
        
        
        #wandb.log({"test_mse": tmse})



# [Save Model]

# [End Save Model]

# [Run the model]
if __name__ == '__main__':
    wandb.agent(sweep_id, train, count=200)
# [End experiment]