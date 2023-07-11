import torch
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.utils.data import DataLoader
from modules.data_process import Create_Dataset
from modules.crossformer import Crossformer
from sklearn.preprocessing import StandardScaler
from modules.loss import Myloss


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
print(f'use device: {DEVICE}')

'''
datafile, window_size, pred_size, split, scaler, mode = str
'''

path = 'CF/datasets/ETTh1.csv'
path1 = 'CF/datasets/SPY_30mins_returns.txt'
scaler = StandardScaler()
def pipeline():
    train_dataset = Create_Dataset(datafile=path, window_size=120, pred_size=10, split=.7, scaler=scaler, mode='train')
    test_dataset = Create_Dataset(datafile=path, window_size=120, pred_size=10, split=.7, scaler=scaler, mode='train')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=1,pin_memory=True,  drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False, num_workers=1,pin_memory=True)

    return train_dataloader, test_dataloader


'''
data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline = False, device=torch.device('cuda:0')):
'''
def model():
    net = Crossformer(data_dim=7, in_len=10, out_len=10, seg_len=10)
    return net

def train():

    train_dataloader, test_dataloader = pipeline()

    net = model()

    loss_function = Myloss()

    optimizer = optim.AdamW(net.parameters(), lr = .0001)
    
    net.train()
    for epochs in range(10):
        for i, (x,y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)

            print(x.shape, y.shape)
            y_pred = net(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    train()