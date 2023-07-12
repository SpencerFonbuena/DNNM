import torch
import wandb
import glob
import os
import colossalai
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import pandas as pd


from torch.utils.data import DataLoader
from modules.data_process import Create_Dataset
from modules.crossformer import Crossformer
from sklearn.preprocessing import StandardScaler
from modules.loss import Myloss
from modules.hyperparameters import Hyperparameters as hp


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
print(f'use device: {DEVICE}')

'''
datafile, window_size, pred_size, split, scaler, mode = str
'''

wandb.init(project='mark Garbage', name="01")
if torch.cuda.is_available():
    path = '/root/DNNM/datasets/ETTh1.csv'
    path1 = '/root/DNNM/datasets/ETTm1.csv'
    path3 = '/mnt/blockstorage/'
else:
    path = 'CF/datasets/ETTh1.csv'
    path1 = 'CF/datasets/SPY_30mins_returns.txt'
    path3 = '/Users/spencerfonbuena/Documents/Python/Trading Models/CF/dataset'
scaler = StandardScaler()


'''colossalai.launch_from_torch(
    config='DNNM/modules/hyperparameters.py',
)'''


def pipeline(datafile):
    train_dataset = Create_Dataset(datafile=datafile, window_size=hp.lookback, pred_size=hp.pred_size, split=hp.split, scaler=scaler, mode='train')
    test_dataset = Create_Dataset(datafile=datafile, window_size=hp.lookback, pred_size=hp.pred_size, split=hp.split, scaler=scaler, mode='train')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=hp.num_workers,pin_memory=True,  drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=hp.num_workers,pin_memory=True)


    return train_dataloader, test_dataloader

'''
d_ff, n_heads, e_layers, 
                dropout, baseline, device):

'''

def model():
    net = Crossformer(data_dim=hp.data_dim, in_len=hp.lookback, out_len=hp.pred_size, seg_len=hp.seg_len,
                      win_size=hp.win_size, d_model=hp.d_model, baseline=hp.baseline, d_ff=hp.d_ff, n_heads=hp.n_heads, 
                      e_layers=hp.e_layers, dropout=hp.dropout, factor=hp.factor, device=DEVICE).to(DEVICE)
    return net

def main():
    net = model()
    loss_function = Myloss()
    optimizer = optim.AdamW(net.parameters(), lr = hp.learning_rate)
    
    net.train()
    for epochs in range(10):
        for datafile in glob.glob(os.path.join(path3, '*.txt')):
            with open(os.path.join(os.getcwd(), datafile), 'r') as f:
                df = pd.read_csv(datafile, delimiter=',', index_col=0)
                train_dataloader, test_dataloader = pipeline(df)
                for i, (x,y) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    y_pred = net(x)
                    loss = loss_function(y_pred, y)
                    loss.backward()
                    optimizer.step()
                    wandb.log({'Loss': loss})
                    wandb.log({'Epoch': epochs})

                    if i % 50 == 0:
                        pre = y_pred.cpu().detach().numpy()[0,:,0]
                        ys = y.cpu().detach().numpy()[0,:,0]
                        fig, ax = plt.subplots()
                        ax.plot(pre, label='predictions')
                        ax.plot(ys, label ='actual')
                        plt.legend()
                        wandb.log({'train plot': wandb.Image(fig)})
                        plt.close()

        test(net=net, dataloader=test_dataloader, optimizer=optimizer, loss_function=loss_function)

def test(net, dataloader, optimizer, loss_function):
    net.eval()
    for epochs in range(10):
        for i, (x,y) in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = net(x)
            loss = loss_function(y_pred, y)
            wandb.log({'test loss': loss})
            if i % 20 == 0:
                pre = y_pred.cpu().detach().numpy()[0,:,0]
                ys = y.cpu().detach().numpy()[0,:,0]
                fig, ax = plt.subplots()
                ax.plot(pre, label='predictions')
                ax.plot(ys, label ='actual')
                plt.legend()
                wandb.log({'test plot': wandb.Image(fig)})
                plt.close()

if __name__ == '__main__':
    main()