import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

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
data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline = False, device=torch.device('cuda:0')):
'''
def model():
    net = Crossformer(data_dim=hp.data_dim, in_len=hp.lookback, out_len=hp.pred_size, seg_len=hp.seg_len)
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
            print(loss)
            loss.backward()
            optimizer.step()

            
        print(epochs)
        pre = y_pred.cpu().detach().numpy()[0]
        ys = y.cpu().detach().numpy()[0]
        fig, ax = plt.subplots()

        ax.plot(pre, label='predictions')
        ax.plot(ys, label ='actual')
        plt.legend()


if __name__ == '__main__':
    train()