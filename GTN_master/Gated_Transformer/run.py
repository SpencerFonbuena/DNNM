# @Time    : 2021/01/22 25:16
# @Author  : SY.M
# @FileName: run.py

import torch
from torch.utils.data import DataLoader
from dataset_process.dataset_process import Create_Dataset
import torch.optim as optim
from time import time
from tqdm import tqdm
import os

from module.transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization

# from mytest.gather.main import draw

reslut_figure_path = 'result_figure'  # Result image save path

# Dataset path selection
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\AUSLAN\\AUSLAN.mat'  # lenth=1140  input=136 channel=22 output=95
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\CharacterTrajectories\\CharacterTrajectories.mat'
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\CMUsubject16\\CMUsubject16.mat'  # lenth=29,29  input=580 channel=62 output=2
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\ECG\\ECG.mat'  # lenth=100  input=152 channel=2 output=2
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\JapaneseVowels\\JapaneseVowels.mat'  # lenth=270  input=29 channel=12 output=9
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\Libras\\Libras.mat'  # lenth=180  input=45 channel=2 output=15
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\UWave\\UWave.mat'  # lenth=4278  input=315 channel=3 output=8
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\KickvsPunch\\KickvsPunch.mat'  # lenth=10  input=841 channel=62 output=2
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\NetFlow\\NetFlow.mat'  # lenth=803  input=997 channel=4 output=只有1和13
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\ArabicDigits\\ArabicDigits.mat'  # lenth=6600  input=93 channel=13 output=10
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\PEMS\\PEMS.mat'
# path = 'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\Wafer\\Wafer.mat'
path = '/root/GTN/GTN_master/AAPL_1hour_expanded_test.txt'
#path = '/Users/spencerfonbuena/Documents/Python/Trading Models/gtn/GTN_master/AAPL_1hour_expanded_test.txt'

test_interval = 5  # Test interval unit: epoch
draw_key = 1  # Greater than or equal to draw_key will save the image
file_name = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]  # get file name

# hyperparameter settings
EPOCH = 12
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
print(f'use device: {DEVICE}')

d_model = 512
d_hidden = 1024
q = 8
v = 8
h = 8
N = 8
dropout = 0.2
pe = True # # The setting is in the twin towers score=pe score=channel has no pe by default
mask = True # set the mask of score=input in the twin towers score=channel has no mask by default
# optimizer selection
optimizer_name = 'Adagrad'

train_dataset = Create_Dataset(datafile=path, window_size=100, split=.8, mode='train')
test_dataset = Create_Dataset(datafile=path, window_size=100, split=.8, mode='test')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

DATA_LEN = train_dataset.training_len # Number of samples in the training set
d_input = train_dataset.input_len # number of time parts
d_channel = train_dataset.channel_len # time series dimension
d_output = train_dataset.output_len # classification category

# Dimension display
print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

# Create a Transformer model
net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
# Create a loss function here using cross entropy loss
loss_function = Myloss()
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)

# Used to record the accuracy rate change
correct_on_train = []
correct_on_test = []
# Used to record loss changes
loss_list = []
time_cost = 0


# test function
def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
        if flag == 'test_set':
            correct_on_test.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))

        return round((100 * correct / total), 2)


# training function
def train():
    net.train()
    max_accuracy = 0
    pbar = tqdm(total=EPOCH)
    begin = time()
    for index in range(EPOCH):
        print(index)
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')

            loss = loss_function(y_pre, y.to(DEVICE))
            
            
            #print(f'Epoch:{i + 1}:\t\tloss:{loss.item()}')
            loss_list.append(loss.item())

            loss.backward()

            optimizer.step()

        if ((index + 1) % test_interval) == 0:
            current_accuracy = test(test_dataloader)
            test(train_dataloader, 'train_set')
            print(f'current maximum accuracy\t test set: {max(correct_on_test)}%\t training set: {max(correct_on_train)}%')

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                #torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')
                torch.save(net, '/root/GTN/GTN_master/mach0.txt')

        pbar.update()

    os.rename(f'saved_model/{file_name} batch={BATCH_SIZE}.pkl',
              f'saved_model/{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl')

    end = time()
    time_cost = round((end - begin) / 60, 2)

    # result graph
    result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
                         test_interval=test_interval,
                         d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
                         time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
                         file_name=file_name,
                         optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask)


if __name__ == '__main__':
    train()
