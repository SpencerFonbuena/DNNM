# @Time    : 2021/01/22 25:16
# @Author  : SY.M
# @FileName: run.py

import torch
import torchmetrics as tm
from torch.utils.data import DataLoader
from dataset_process.dataset_process import Create_Dataset
import torch.optim as optim
from time import time
from tqdm import tqdm
import os
import numpy as np

from module.transformer import Transformer
from module.loss import Myloss

# from mytest.gather.main import draw

reslut_figure_path = 'result_figure'  # Result image save path

path = '/root/GTN/GTN_master/AAPL_1hour_expand.txt'
#path = '/Users/spencerfonbuena/Desktop/AAPL_1hour_expanded_test 3.txt'

test_interval = 2  # Test interval unit: epoch
draw_key = 1  # Greater than or equal to draw_key will save the image
file_name = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]  # get file name

# hyperparameter settings
EPOCH = 225
BATCH_SIZE = 16
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
print(f'use device: {DEVICE}')
d_model = 512
d_hidden = 1024
q = 16
v = 16
h = 16
N = 32
dropout = 0.2
pe = True # # The setting is in the twin towers score=pe score=channel has no pe by default
mask = True # set the mask of score=input in the twin towers score=channel has no mask by default
# optimizer selection
optimizer_name = 'Adam'

train_dataset = Create_Dataset(datafile=path, window_size=120, split=.85, mode='train')
test_dataset = Create_Dataset(datafile=path, window_size=120, split=.85, mode='test')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

DATA_LEN = train_dataset.training_len # Number of samples in the training set
d_input = train_dataset.input_len # number of time parts
d_channel = train_dataset.channel_len # feature dimension
d_output = train_dataset.output_len # classification category

# Dimension display
print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

# Create a Transformer model
net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)

#print the model summary
#print(net)
#Print the number of parameters
#print(sum([param.nelement() for param in net.parameters()])) (Currently there are: 101M parameters)

# Create a loss function here using cross entropy loss
loss_function = Myloss()

if optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)

# Used to record the accuracy rate change
correct_on_train = []
correct_on_test = []
# Used to record loss changes
loss_list = []
time_cost = 0


# test function
def test(dataloader, flag=str):
    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            #print(y_pre[:15], y[:15])
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            #print(total)
            correct += (label_index == y.long()).sum().item()
            #print(correct)
        if flag == "train_set":
            print(f"Train Accuracy: {correct / total * 100}")
        if flag == "test_set":
            print(f"Test Accuracy: {correct / total * 100}")

# training function
def train():
    net.train()
    begin = time()
    for index in range(EPOCH):
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')
            loss = loss_function(y_pre, y.to(DEVICE))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        #validate training accuracy and test accuracy
        if ((index + 1) % test_interval) == 0:
            #current_accuracy = test(test_dataloader)
            print(loss)
            test(train_dataloader, 'train_set')
            test(test_dataloader, 'test_set')
            #print(f'current maximum accuracy\t test set: {max(correct_on_test)}%\t training set: {max(correct_on_train)}%')

            #if current_accuracy > max_accuracy:
                #max_accuracy = current_accuracy
                #torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')
                #torch.save(net, '/root/GTN/GTN_master/mach0.txt')

    os.rename(f'saved_model/{file_name} batch={BATCH_SIZE}.pkl',
              f'saved_model/{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl')

    end = time()
    time_cost = round((end - begin) / 60, 2)

if __name__ == '__main__':
    train()
