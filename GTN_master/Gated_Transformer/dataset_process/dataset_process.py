# @Time    : 2021/07/16 19:50
# @Author  : SY.M
# @FileName: create_dataset.py

import  torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Use device {DEVICE}')

#Create_Dataset class that inherits the attributes and methods of torch.utils.data.Dataset
class Create_Dataset(Dataset):
    def __init__(self, datafile, window_size, split, mode = str): # datafile -> csv file | window_size -> # of timesteps in each example | split -> The percent of data you want for training
        
        # Gives either the training or the validation set depending on what is requested
        self.mode = mode
        self.window_size = window_size
        #reading in the entire un windowed dataset with the labels still part
        df = torch.nn.functional.normalize(pd.read_csv(datafile, index_col=0, delimiter=',').to_numpy())
        #creating the training dataset without the labels in the file
        labeldf = df.drop(columns='Labels')

        #Take the lengthof the data, and multiply it by the percentage of the data you want to train on (split)
        splitlocation = int(len(labeldf) * split)

       #temporarily store the windowed data
        window_set = []

        #window the data
        for i in range(len(labeldf) - self.window_size):
            example = labeldf[i: self.window_size + i]
            window_set.append(np.expand_dims(example, 0))
        
        #Training Dataset
        #The reason it is splitlocation - windowsize is because the if the dataset goes till the same as the labels, it will add in 100 extra examples, whose set contains values 
        #beyond that of the labels. Similar to the long descripiton of self.trainlabels.
        self.traindataset = torch.tensor(np.vstack(window_set))[:(splitlocation - self.window_size), :]
        #print(self.traindataset.shape)
        self.valdataset = torch.tensor(np.vstack(window_set))[(splitlocation - self.window_size):, :]
        #print(self.valdataset.shape)
        
        #Create the training labels. The reason it is starting from window size, is because there is technically labels for what happened after each timestep: however,
        #We created windows of data, so we want to know what is happening at the end of our window. If we started at the beginning, our labels would be off by the size of self.window_size
        self.trainlabels = torch.tensor(df['Labels'][self.window_size: splitlocation].to_numpy())
        print(self.trainlabels.shape)
        self.vallabels = torch.tensor(df['Labels'][splitlocation:].to_numpy())
        #print('labels dimensions', self.trainlabels.shape)
        #print('labels val dimensions', self.vallabels.shape)


        
        self.training_len = self.traindataset.shape[0] # Number of samples in the training set
        self.input_len = self.window_size# number of time parts
        self.channel_len = self.traindataset.shape[2]# Number of features (Channels)
        self.output_len = 4 # classification category
        self.test_len = self.valdataset.shape[0]
        
    
    def __getitem__(self, index):
        if self.mode == 'train':
            #print(self.traindataset[index].shape, self.trainlabels[index + self.window_size])
            return self.traindataset[index], self.trainlabels[index]
        elif self.mode == 'test':
            return self.valdataset[index], self.vallabels[index]
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.traindataset)
        if self.mode == 'test':
            return len(self.valdataset)
        

