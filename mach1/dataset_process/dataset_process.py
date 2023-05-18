# @Time    : 2021/07/16 19:50
# @Author  : SY.M
# @FileName: create_dataset.py

import  torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#print(f'Use device {DEVICE}')

#Create_Dataset class that inherits the attributes and methods of torch.utils.data.Dataset
class Create_Dataset(Dataset):
    def __init__(self, datafile, window_size, split, mode = str): # datafile -> csv file | window_size -> # of timesteps in each example | split -> The percent of data you want for training
        
        self.mode = mode
        
        df = pd.read_csv(datafile, delimiter=',', index_col=0)
        print(len(df))
        #Create the training and label datasets
        labeldata = df['Labels'].to_numpy()[window_size -1:]
        rawtrainingdata = df.drop(columns='Labels').to_numpy()
        
        #create a split value to separate valadate from training
        self.split = int(len(df) * split)
        
       #window the datasets
        window_array = np.array([np.arange(window_size)])
        dataset_array = np.array(np.arange(len(rawtrainingdata)-window_size + 1)).reshape(len(rawtrainingdata)-window_size + 1, 1)
        indexdata = window_array + dataset_array
        print(indexdata)

        trainingdata = rawtrainingdata[indexdata]
        print(trainingdata.shape)

        #create the training data and labels
        self.trainingdata = torch.tensor(trainingdata[:self.split]).to(torch.float32)
        self.traininglabels = torch.tensor(labeldata[:self.split]).to(torch.float32)
        

        #create the validation data and labels
        self.valdata = torch.tensor(trainingdata[self.split:]).to(torch.float32)
        self.vallabels = torch.tensor(labeldata[self.split:]).to(torch.float32)


        
        self.training_len = self.trainingdata.shape[0] # Number of samples in the training set
        self.input_len = window_size# number of time parts
        self.channel_len = self.trainingdata.shape[2]# Number of features (Channels)
        self.output_len = 4 # classification category
        self.test_len = self.valdata.shape[0]
        
    
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.trainingdata[index], self.traininglabels[index]
        elif self.mode == 'test':
            return self.valdata[index], self.vallabels[index]
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.trainingdata)
        if self.mode == 'test':
            return len(self.vallabels)
        

