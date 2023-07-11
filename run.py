import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import wandb

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

wandb.init(project='mark LI', name="01")
if torch.cuda.is_available():
    path = '/root/DNNM/datasets/ETTh1.csv'
    path1 = '/root/DNNM/datasets/SPY_30mins_returns.txt'
else:
    path = 'CF/datasets/ETTh1.csv'
    path1 = 'CF/datasets/SPY_30mins_returns.txt'
scaler = StandardScaler()
def pipeline():
    train_dataset = Create_Dataset(datafile=path, window_size=hp.lookback, pred_size=hp.pred_size, split=hp.split, scaler=scaler, mode='train')
    test_dataset = Create_Dataset(datafile=path, window_size=hp.lookback, pred_size=hp.pred_size, split=hp.split, scaler=scaler, mode='train')

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

def train():

    train_dataloader, test_dataloader = pipeline()

    net = model()

    loss_function = Myloss()

    optimizer = optim.AdamW(net.parameters(), lr = hp.learning_rate)
    
    net.train()
    for epochs in range(10):
        for i, (x,y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = net(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            wandb.log({'Loss': loss})
            wandb.log({'Epoch': epochs})
 
        pre = y_pred.cpu().detach().numpy()[0,:,0]
        ys = y.cpu().detach().numpy()[0,:,0]
        fig, ax = plt.subplots()
        ax.plot(pre, label='predictions')
        ax.plot(ys, label ='actual')
        plt.legend()
        wandb.log({'train plot': wandb.Image(fig)})
        plt.close()

        test(net=net, dataloader=train_dataloader, optimizer=optimizer)

def test(net, dataloader, optimizer):
    net.eval()
    for epochs in range(10):
        for i, (x,y) in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = net(x)
            
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
    train()