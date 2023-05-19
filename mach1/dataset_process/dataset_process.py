# @Time    : 2021/07/16 19:50
# @Author  : SY.M
# @FileName: create_dataset.py

import  torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import random

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#print(f'Use device {DEVICE}')

#Create_Dataset class that inherits the attributes and methods of torch.utils.data.Dataset
class Create_Dataset(Dataset):
    def __init__(self, datafile, window_size, split, mode = str): # datafile -> csv file | window_size -> # of timesteps in each example | split -> The percent of data you want for training
        
        self.mode = mode
        
        df = pd.read_csv(datafile, delimiter=',', index_col=0)

        
        #Create the training and label datasets
        labeldata = df['Labels'].to_numpy()[window_size -1:]
        prerawtrainingdata = torch.nn.functional.normalize(df.drop(columns='Labels'))
        rawtrainingdata = pd.DataFrame(prerawtrainingdata).to_numpy()
        
        #Find the distributions of each label
        distlabel = pd.DataFrame(labeldata).value_counts()
        print(distlabel)
        
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
        
        #can't call the iterate through a torch, so this is to create all of the weights
        self.normvallabels = labeldata[self.split:]

        #Find the distributions of each label in the validation set
        self.testsampleweights = [self.distlabel[i] for i in self.normvallabels]

        
        self.training_len = self.trainingdata.shape[0] # Number of samples in the training set
        self.input_len = window_size# number of time parts
        self.channel_len = self.trainingdata.shape[2]# Number of features (Channels)
        self.output_len = 2 # classification category
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
        

