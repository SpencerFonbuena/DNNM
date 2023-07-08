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
from torcheval.metrics import  MulticlassAccuracy, MulticlassPrecision, MulticlassRecall




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


# Log on Weights and Biases

wandb.init(project='mach46', name='01')

#switch datasets depending on local or virtual run
if torch.cuda.is_available():
    path = '/root/DNNM/mach1/datasets/SPY_30mins_class_returns.txt'
else:
    path = 'DNNM/mach1/datasets/SPY_30mins_class_returns.txt'

# [End General Init]

'''-----------------------------------------------------------------------------------------------------'''
'''====================================================================================================='''


#[Create and load the dataset]
def pipeline(batch_size, window_size):
    #create the datasets to be loaded
    train_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, mode='train', pred_size=hp.pred_size)
    test_dataset = Create_Dataset(datafile=path, window_size=window_size, split=hp.split, mode='test', pred_size=hp.pred_size)

    #create the samplers
    samplertrain = wrs(weights=train_dataset.trainsampleweights, num_samples=len(train_dataset), replacement=True)
    samplertest = wrs(weights=test_dataset.testsampleweights, num_samples=len(test_dataset), replacement=True)

    #Load the data
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True ,sampler=samplertrain, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True,sampler=samplertest, drop_last=True)

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


def network(d_input, d_channel, d_output, window_size, heads, d_model, dropout, stack, p, d_hidden):
    net = Transformer(window_size=window_size, 
                    timestep_in=d_input, 
                    channel_in=d_channel,
                    heads=heads,
                    d_model=d_model,
                    device=DEVICE,
                    dropout = dropout,
                    class_num=d_output, 
                    stack=stack, 
                    p=p, 
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
    # [Place computational graph code here if desired]


def train():


    train_dataloader, test_dataloader, d_input, d_channel, d_output = pipeline(batch_size=hp.batch_size, window_size=hp.window_size)
    net = network(d_input=d_input, d_channel=d_channel, d_output=d_output, window_size=hp.window_size, heads=hp.heads, d_model=hp.d_model, 
                    dropout=hp.dropout, stack=hp.stack, p=hp.stoch_p, d_hidden=hp.d_hidden).to(DEVICE)
    # Create a loss function here using cross entropy loss
    loss_function = Myloss()

    #Select optimizer in an un-optimized way
    if hp.optimizer_name == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=hp.learning_rate)

    # training function
    trainmetricaccuracy = MulticlassAccuracy().to(DEVICE)
    specacc = MulticlassAccuracy(average=None, num_classes=3).to(DEVICE)
    trainprecision = MulticlassPrecision(average=None, num_classes=3).to(DEVICE)
    trainrecall = MulticlassRecall(average=None, num_classes=3).to(DEVICE)
    
    net.train()
    wandb.watch(net, log='all')
    for index in tqdm(range(hp.EPOCH)):
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pre = net(x, True)
            loss = loss_function(y_pre, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), .5)
            optimizer.step()

            trainmetricaccuracy.update(y_pre, y)
            specacc.update(y_pre.to(torch.int64), y.to(torch.int64))
            trainprecision.update(y_pre.to(torch.int64), y.to(torch.int64))
            trainrecall.update(y_pre.to(torch.int64), y.to(torch.int64))


            wandb.log({'Loss': loss})
            wandb.log({'index': index})

        
        trainaccuracy = trainmetricaccuracy.compute()
        print('TrainAcc',specacc.compute())
        print('TrainPrecision',trainprecision.compute())
        print('TrainRecall ',trainrecall.compute())


        wandb.log({"train_acc": trainaccuracy})
        
        test(dataloader=test_dataloader, net=net, loss_function=loss_function)


# test function
def test(dataloader, net, loss_function):
    metricaccuracy = MulticlassAccuracy().to(DEVICE)
    testspecacc = MulticlassAccuracy(average=None, num_classes=3).to(DEVICE)
    testprecision = MulticlassPrecision(average=None, num_classes=3).to(DEVICE)
    testrecall = MulticlassRecall(average=None, num_classes=3).to(DEVICE)
    net.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre = net(x, False)
            metricaccuracy.update(y_pre, y)
            testspecacc.update(y_pre.to(torch.int64), y.to(torch.int64))
            testprecision.update(y_pre.to(torch.int64), y.to(torch.int64))
            testrecall.update(y_pre.to(torch.int64), y.to(torch.int64))
        
        accuracy = metricaccuracy.compute()
        wandb.log({"test_acc": accuracy})
        print('test',testspecacc.compute())
        print('TestPrecision',testprecision.compute())
        print('TestRecall',testrecall.compute())


# [Save Model]

# [End Save Model]

# [Run the model]
if __name__ == '__main__':
    train()
# [End experiment]