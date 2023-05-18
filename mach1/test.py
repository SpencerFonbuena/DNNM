import pandas as pd
import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn import Module
import random

#set random seed for reproducibility
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)



wandb.init(
    project='test modelacc',
    name='randomseedtestchangerelu'
)

path = '/root/GTN/mach1/AAPL_1hour_expand.txt'

#Hyperparameters
hiddenlayers = 1024
epoch = 1000
testime = 5

class Create_Dataset(Dataset):


    def __init__(self, datafile, split, mode = str): 
        
        self.mode = mode
        
        df = pd.read_csv(datafile, index_col=0, delimiter=',')

        #Create the training and label datasets
        labeldata = df['Labels']
        trainingdata = df.drop(columns='Labels')

        #create a split value to separate valadate from training
        self.split = int(len(df) * split)
        
        #create the training data and labels
        self.trainingdata = torch.tensor(trainingdata[:self.split].to_numpy()).to(torch.float32)
        self.traininglabels = torch.tensor(labeldata[:self.split].to_numpy()).to(torch.float32)
        

        #create the validation data and labels
        self.valdata = torch.tensor(trainingdata[self.split:].to_numpy()).to(torch.float32)
        self.vallabels = torch.tensor(labeldata[self.split:].to_numpy()).to(torch.float32)
        #print('labels', self.valdata.shape, self.vallabels.shape, self.valdata[0])

        
    
    def __getitem__(self, index):
        if self.mode == 'train':
            return  self.trainingdata[index], self.traininglabels[index]
        elif self.mode == 'validate':
            return self.valdata[index], self.vallabels[index]

    
    def __len__(self):
        if self.mode == 'train':
            return len(self.trainingdata)
        elif self.mode == 'validate':
            return len(self.valdata)

train_dataset = Create_Dataset(datafile=path, split=.6, mode='train')
test_dataset = Create_Dataset(datafile=path, split=.6, mode='validate')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=False, num_workers=2)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False, num_workers=2)


class My_Loss(Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, y_predict, y_true):
        y_true = y_true.type(torch.LongTensor)
        loss = self.loss(y_predict, y_true)

        return loss

class Base_Model(Module):
    def __init__(self,
                 d_input = int,
                 d_output = int,
                 class_num = int):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_input, d_output)
        self.linear2 = torch.nn.Linear(d_output, d_output)
        self.linear3 = torch.nn.Linear(d_output, d_output)
        self.linear_out = torch.nn.Linear(d_output, class_num)
    
    def forward(self, input_data):
        x = F.relu(self.linear1(input_data))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        layerout = self.linear_out(x)
        
        return layerout


#instantiate the model
model = Base_Model(d_input=9, d_output=hiddenlayers, class_num=4)
#instantiate loss function
loss = My_Loss()
#instantiate optimizer
optimizer = torch.optim.Adam(model.parameters())

def train():
    model.train()
    wandb.watch(model, log='all')
    for index in range(epoch):
        for i, (x, y) in enumerate(train_dataloader):
            
            #before each step, set your gradients to 0
            optimizer.zero_grad()
            #make prediction
            y_predict = model(x)
            #test loss
            loss_function = loss(y_predict, y)
            loss_function.backward()
            optimizer.step()
        
        if index % testime == 0:
            test(train_dataloader, flag='train')
            test(test_dataloader, flag='test')
            wandb.log({"loss": loss_function})

    pass

def test(dataloader, flag = str):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x, y
            y_predict = model(x)
            _, label_index = torch.max(y_predict.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
            accuracy = correct / total * 100
        if flag == 'train':
            wandb.log({"training accuracy": accuracy})
        if flag == 'test':
            wandb.log({"test accuracy": accuracy})
    pass
if __name__ == '__main__':
    train()