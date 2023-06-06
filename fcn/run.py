from dataset_process.dataset_process import Create_Dataset
from modules.hyperparameters import HyperParameters as hp
from torch.utils.data import WeightedRandomSampler as wrs
from torch.utils.data import DataLoader
from modules.loss import Myloss
from modules.fcn import FN

import torch
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import wandb
import random
import numpy as np


# [Initialize stat-tracking]
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
print(f'use device: {DEVICE}')

wandb.init(
    project='garbage',
    name='bigger batch and window'
)
# [End Initialization]


# [Initialize functions for dataset]
if torch.cuda.is_available():
    path = '/root/GTN/fcn/datasets/AAPL_1hour_expand.txt'
else:
    path = 'models/fcn/datasets/AAPL_1hour_expand.txt'
# [End Initialization]



#create the dataset to be loaded
train_dataset = Create_Dataset(datafile=path, window_size=hp.WINDOW_SIZE, split=hp.split, mode='train')
val_dataset = Create_Dataset(datafile=path, window_size=hp.WINDOW_SIZE, split=hp.split, mode='validate')
test_dataset = Create_Dataset(datafile=path, window_size=hp.WINDOW_SIZE, split=hp.split, mode='test')

#create the sampler
samplertrain = wrs(weights=train_dataset.trainsampleweights, num_samples=len(train_dataset), replacement=True)
samplertest = wrs(weights=test_dataset.testsampleweights, num_samples=len(test_dataset), replacement=True)
samplertrainval = wrs(weights=val_dataset.trainvalsampleweights, num_samples=len(val_dataset), replacement=True)

#Load the data
train_dataloader = DataLoader(dataset=train_dataset, batch_size=hp.BATCH_SIZE, shuffle=False,  sampler=samplertrain)
validate_dataloader = DataLoader(dataset=val_dataset, batch_size=hp.BATCH_SIZE, shuffle=False,  sampler=samplertrainval)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=hp.BATCH_SIZE, shuffle=False,  sampler=samplertest)




# [Initialize functions for training]
loss_function = Myloss()
model = FN(data_in=9, data_out=4, layers=[128,256,512], kss=[7,5,3],  stack=124,p=.5).to(DEVICE)

#Choose Optimizer
if hp.optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=hp.LR)
# [End training initialization]



# Used to record the accuracy rate change
correct_on_train = []
correct_on_test = []
# Used to record loss changes
loss_list = []
time_cost = 0


# training function
def train():
    model.train()
    wandb.watch(model, log='all')
    for index in tqdm(range(hp.EPOCH)):
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pre = model(x.to(DEVICE))
            loss = loss_function(y_pre, y.to(DEVICE))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            wandb.log({'Loss': loss})
            wandb.log({'index': index})
        #validate training accuracy and test accuracy
        test(validate_dataloader, 'train')
        test(test_dataloader, 'test')
        

# test function
def test(dataloader, flag = str):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre = model(x)
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
            accuracy = correct / total * 100
        if flag == 'train':
            wandb.log({"Train acc": accuracy})
        if flag == 'test':
            wandb.log({"Test acc": accuracy})



if __name__ == '__main__':
    train()