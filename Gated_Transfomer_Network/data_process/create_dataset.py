# @Time    : 2021/07/16 19:50
# @Author  : SY.M
# @FileName: create_dataset.py

import  torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from config.param_config import Config as c
import pandas as pd

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Use device {DEVICE}')

#Create_Dataset class that inherits the attributes and methods of torch.utils.data.Dataset
class Create_Dataset(Dataset):
    def __init__(self, datafile, window_size):
        #reading in the entire un windowed dataset with the labels still part
        self.df = pd.read_csv(datafile, index_col=0, delimiter=',')
        #creating the training dataset without the labels in the file
        self.traindf = self.df.drop(columns='Labels')

       #temporarily store the windowed data
        window_set = []

        #window the data
        for i in range(len(self.traindf) - window_size):
            example = self.traindf[i: window_size + i]
            window_set.append(np.expand_dims(example, 0))
        
        #Training Dataset  
        self.dataset = torch.tensor(np.vstack(window_set)).transpose(-1,-2).to(DEVICE)
        
        #Create the training labels. The reason it is starting from window size, is because there is technically labels for what happened after each timestep: however,
        #We created windows of data, so we want to know what is happening at the end of our window. If we started at the beginning, our labels would be off by the size of window_size
        self.labels = torch.tensor(self.df['Labels'][window_size:].to_numpy()).to(DEVICE)

        #This is the stratified k fold index so that we can train and test on the whole set. The startified will also keep the label distribution constant in train and test
        self.sfk_indexset = StratifiedKFold(n_splits=c.k_fold, shuffle=True) \
            .split(self.dataset, self.labels)

    
    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]
    
    def __len__(self):
        return len(self.dataset)

#No idea if this is working, but thought I would use this to create two seperate training sets from our main big dataset
class Stratified_Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        # print(self.X.shape) Shape for the training set: (54805, 9, 100) | Shape for the val set: (54805)
        # print(self.Y.shape) Shape for the training labels : (13702, 9, 100) | Shape for the val labels: (13702)
    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return len(self.X)
