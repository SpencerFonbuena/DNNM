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
from torch.utils.data import WeightedRandomSampler as wrs
import wandb

from module.transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization

# from mytest.gather.main import draw

# Log on Weights and Biases
wandb.init(
    project='garbage',
    name='changed validation set',
    #config=config
)

setup_seed(30)  # 设置随机数种子
reslut_figure_path = 'result_figure'  # 结果图像保存路径

#switch datasets depending on local or virtual run
if torch.cuda.is_available():
    path = '/root/GTN/mach1/datasets/AAPL_1hour_expand.txt'
else:
    path = 'models/mach1/datasets/AAPL_1hour_expand.txt'

test_interval = 5  # 测试间隔 单位：epoch
draw_key = 1  # 大于等于draw_key才会保存图像
file_name = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]  # 获得文件名字

# 超参数设置
EPOCH = 100
BATCH_SIZE = 3
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
print(f'use device: {DEVICE}')

d_model = 512
d_hidden = 1024
q = 8
v = 8
h = 8
N = 8
dropout = 0.2
pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask
# 优化器选择
optimizer_name = 'Adagrad'

#create the datasets to be loaded
train_dataset = Create_Dataset(datafile=path, window_size=120, split=.85, mode='train')
val_dataset = Create_Dataset(datafile=path, window_size=120, split=.85, mode='validate')
test_dataset = Create_Dataset(datafile=path, window_size=120, split=.85, mode='test')

#create the samplers
samplertrain = wrs(weights=train_dataset.trainsampleweights, num_samples=len(train_dataset), replacement=True)
samplertest = wrs(weights=test_dataset.testsampleweights, num_samples=len(test_dataset), replacement=True)
samplertrainval = wrs(weights=val_dataset.trainvalsampleweights, num_samples=len(val_dataset), replacement=True)

#Load the data
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False,  sampler=samplertrain)
validate_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,  sampler=samplertrainval)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,  sampler=samplertest)

DATA_LEN = train_dataset.training_len # Number of samples in the training set
d_input = train_dataset.input_len # number of time parts
d_channel = train_dataset.channel_len # feature dimension
d_output = train_dataset.output_len # classification category

# 维度展示
print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

# 创建Transformer模型
net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
# 创建loss函数 此处使用 交叉熵损失
loss_function = Myloss()
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)

# 用于记录准确率变化
correct_on_train = []
correct_on_test = []
# 用于记录损失变化
loss_list = []
time_cost = 0


# 测试函数
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

            wandb.log({'test acc': round((100 * correct / total), 2)})
        elif flag == 'train_set':
            correct_on_train.append(round((100 * correct / total), 2))

            wandb.log({'train acc': round((100 * correct / total), 2)})
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))

        return round((100 * correct / total), 2)


# 训练函数
def train():
    net.train()
    wandb.watch(net, log='all')
    max_accuracy = 0
    pbar = tqdm(total=EPOCH)
    begin = time()
    for index in range(EPOCH):
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')

            loss = loss_function(y_pre, y.to(DEVICE))
            wandb.log({'Loss': loss})
            #print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}')
            loss_list.append(loss.item())

            loss.backward()

            optimizer.step()

        if ((index + 1) % test_interval) == 0:
            current_accuracy = test(test_dataloader)
            test(train_dataloader, 'train_set')
            print(f'当前最大准确率\t测试集:{max(correct_on_test)}%\t 训练集:{max(correct_on_train)}%')

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')

        pbar.update()

    os.rename(f'saved_model/{file_name} batch={BATCH_SIZE}.pkl',
              f'saved_model/{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl')

    end = time()
    time_cost = round((end - begin) / 60, 2)

    # 结果图
    result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
                         test_interval=test_interval,
                         d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
                         time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
                         file_name=file_name,
                         optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask)


if __name__ == '__main__':
    train()
