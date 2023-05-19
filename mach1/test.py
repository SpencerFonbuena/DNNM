import pandas as pd
import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn import Module
import random
from torch.utils.data import WeightedRandomSampler as wrs
import matplotlib.pyplot as plt

#set random seed for reproducibility
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)



wandb.init(
    project='test vmdataq',
    name='machbetterdata'
)

path = '/Users/spencerfonbuena/Desktop/AAPL_1hour_expand.txt'

#Hyperparameters
hiddenlayers = 1024
epoch = 1000
testime = 5
window_size = 10

class Create_Dataset(Dataset):


    def __init__(self, datafile, split, window_size, mode = str): 
        
        self.mode = mode
        df = pd.read_csv(datafile, delimiter=',', index_col=0)
        self.df = df
        #Create the training and label datasets
        labeldata = df['Labels'].to_numpy()[window_size -1:]
        rawtrainingdata = df.drop(columns='Labels').to_numpy()
        

        #create a split value to separate valadate from training
        self.split = int(len(df) * split)
        
       #window the datasets
        window_array = np.array([np.arange(window_size)])
        dataset_array = np.array(np.arange(len(rawtrainingdata)-window_size + 1)).reshape(len(rawtrainingdata)-window_size + 1, 1)
        indexdata = window_array + dataset_array

        trainingdata = rawtrainingdata[indexdata]

        #create the training data and labels
        self.trainingdata = torch.tensor(trainingdata[:self.split]).to(torch.float32)
        self.traininglabels = torch.tensor(labeldata[:self.split]).to(torch.float32)
        self.normtraininglabels = labeldata[:self.split]

        #Find the distributions of each label in the training set
        self.distlabel = 1 / (pd.DataFrame(labeldata).value_counts())
        self.trainsampleweights = [self.distlabel[i] for i in self.normtraininglabels]

        #create the validation data and labels
        self.valdata = torch.tensor(trainingdata[self.split:]).to(torch.float32)
        self.vallabels = torch.tensor(labeldata[self.split:]).to(torch.float32)
        self.normvallabels = labeldata[self.split:]

        #Find the distributions of each label in the validation set
        self.testsampleweights = [self.distlabel[i] for i in self.normvallabels]

        
    
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

train_dataset = Create_Dataset(datafile=path, split=.6, window_size=window_size, mode='train')
test_dataset = Create_Dataset(datafile=path, split=.6, window_size=window_size, mode='validate')

samplertrain = wrs(weights=train_dataset.trainsampleweights, num_samples=len(train_dataset.df), replacement=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=False, sampler= samplertrain )
#This is there to provide a baseline for the sampler
#train_ubdataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=False )

samplertest = wrs(weights=test_dataset.testsampleweights, num_samples=len(test_dataset.df), replacement=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False, sampler=samplertest)

#This is there to check and make sure that the sampling is working as intended. Uncomment as needed
#labelhist = np.array([])
#ublibalehist = np.array([])
#for i, (data, labels) in enumerate(train_dataloader):
#    labelhist = np.append(labelhist, labels)
#for i, (data, labels) in enumerate(train_ubdataloader):
#    ublibalehist = np.append(ublibalehist, labels)
#print(pd.DataFrame(labelhist).value_counts(), pd.DataFrame(ublibalehist).value_counts())


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