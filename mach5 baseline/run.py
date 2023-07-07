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
from module.layers import mask


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
'''sweep_config = {
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

    # [Regularizers]
    'dropout':{
        'values': hp.dropout},
    }
}'''
# [End Sweeps]

# Log on Weights and Biases
wandb.init(project='mach40', name='10')

#switch datasets depending on local or virtual run
if torch.cuda.is_available():
    path = '/root/DNNM/mach1/datasets/SPY_30mins_returns.txt'
else:
    path = 'DNNM/mach1/datasets/SPY_30mins_returns.txt'

# [End General Init]




#[Create and load the dataset]
def pipeline(batch_size, window_size,  pred_size, scaler):
    
    # [create the datasets to be loaded]
    train_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, scaler=scaler, mode='train', pred_size=pred_size)
    test_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, scaler=scaler, mode='test', pred_size=pred_size)
    inference_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, scaler=scaler, mode='inference', pred_size=pred_size)

    # [Load the data]
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,pin_memory=True,  drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True)
    inference_dataloader = DataLoader(dataset=inference_dataset, batch_size=1, shuffle=False, num_workers=1,pin_memory=True)

    DATA_LEN = train_dataset.training_len # Number of samples in the training set
    d_input = train_dataset.input_len # number of time parts
    d_channel = train_dataset.channel_len # feature dimension
    d_output = train_dataset.output_len # classification category

    # [Dimension display]
    print('data structure: [lines, timesteps, features]')
    print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
    print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
    print(f'Number of classes: {d_output}')

    # [End Dataset Init]

    return train_dataloader, test_dataloader, inference_dataloader, d_channel


def network( heads, d_model, dropout, stack, d_hidden, channel_in, window_size, pred_size):
    net = Model(
                    d_model=d_model,
                    heads=heads,
                    stack=stack,
                    dim_feedforward=d_hidden,
                    dropout=dropout,
                    channel_in=channel_in,
                    window_size=window_size,
                    pred_size=pred_size
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


def train():
    # [Define Pipeline]
    train_dataloader, test_dataloader, _, d_channel = pipeline(batch_size=hp.batch_size, window_size=hp.window_size, pred_size=hp.pred_size, scaler=hp.scaler)
    
    # [Define Model]
    net = network(d_model=hp.d_model,
                    heads=hp.heads,
                    stack=hp.stack,
                    d_hidden=hp.d_hidden,
                    dropout=hp.dropout,
                    channel_in=d_channel,
                    window_size=hp.window_size,
                    pred_size=hp.pred_size).to(DEVICE)
    
    # [Define Loss]
    loss_function = Myloss()

    # [Define Optimizer]
    optimizer = optim.AdamW(net.parameters(), lr=hp.LR)

    train_mask = mask(hp.pred_size, hp.pred_size).to(DEVICE)

    # [Training function]
    net.train()
    wandb.watch(net, log='all')
    for index in tqdm(range(hp.EPOCH)):
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pre = net(x, y, train_mask)
            loss = loss_function(y_pre, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), .5)
            optimizer.step()
            
            # [Log Evaluation Metrics]
            wandb.log({'Loss': loss})
            wandb.log({'index': index})
        
            # [Logging the graph]
            if i % 200 == 0:
                pre = y_pre.cpu().detach().numpy()[0]
                ys = y.cpu().detach().numpy()[0]
                fig, ax = plt.subplots()

                ax.plot(pre, label='predictions')
                ax.plot(ys, label ='actual')
                plt.legend()
                wandb.log({'through plots': wandb.Image(fig)})

        # [Save the model]
        path = '/root/DNNM/model_3.pth'
        torch.save(net.state_dict(), path)
        
        # [Validate Model]
        infer(dataloader=test_dataloader, net=net)


# test function
def infer(dataloader, net):
        
    
        
        # [Load Model]
        net.load_state_dict(torch.load('DNNM/model_3.pth') ) # , map_location=torch.device('cpu')


        net.eval()
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(DEVICE)
                predictions = run_encoder_decoder_inference(model=net, 
                                                            src=x,
                                                            forecast_window = hp.pred_size,
                                                            batch_size = hp.batch_size,
                                                            scaler=hp.scaler
                                                            )
        
        predictions = hp.scaler.inverse_transform(predictions.reshape(1, hp.pred_size).cpu())
        
        # [Log Graph]
        pre = torch.tensor(predictions).cpu().detach().numpy()[0].squeeze()
        fig, ax = plt.subplots()

        ax.plot(pre, label='prediction')
        plt.legend()
        wandb.log({"test plot": wandb.Image(fig)})

        

# [End Save Model]

# [Run the model]
if __name__ == '__main__':
    train()
# [End experiment]