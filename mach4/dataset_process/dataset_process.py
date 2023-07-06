# @Time    : 2021/07/16 19:50
# @Author  : SY.M
# @FileName: create_dataset.py

import  torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from module.layers import Scaler
import matplotlib.pyplot as plt
import wandb

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#print(f'Use device {DEVICE}')

#wandb.init(project='mach37', name='01')

#Create_Dataset class that inherits the attributes and methods of torch.utils.data.Dataset
class Create_Dataset(Dataset):
    def __init__(self, datafile, window_size, split, scaler, mode = str ): # datafile -> csv file | window_size -> # of timesteps in each example | split -> The percent of data you want for training
        
        # [Reading in and pre-processing data]

        df = pd.read_csv(datafile, delimiter=',', index_col=0)
  
        #Create the training data
        rawtrainingdata = pd.DataFrame(df['Close']).to_numpy()
        rawtrainingdata = pd.DataFrame(scaler.fit_transform(rawtrainingdata)).to_numpy()

        #create the labels
        rawtraininglabels = pd.DataFrame(df['Close']).to_numpy()
        rawtraininglabels = pd.DataFrame(scaler.fit_transform(rawtraininglabels)).to_numpy()
        # [End pre-processing]



        # [General Functionalities]

        #The data we want will depend if we are in train, validate, or test mode.
        self.mode = mode
        #create a split value to separate validate from training
        self.split = int(len(df) * split)

        # [End Functionalities]




  
        
        # [Window the dataset]
        
        #First array is a row vector, which broadcasts with dataset_array
        window_array = np.array([np.arange(window_size)])
        #This is a column vector (as shown by the reshape, to have a 1 in the column dimension) that broadcasts with window_array
        dataset_array = np.array(np.arange(len(rawtrainingdata)-window_size)).reshape(len(rawtrainingdata)-window_size, 1)
        #broadcast the data together
        indexdata = window_array + dataset_array
        # Index into the raw training data with our preset windows to create datasets quickly
        trainingdata = rawtrainingdata[indexdata]

        #[End windowing dataset]


        # [Window the dataset]
        labelindexdata = np.array([np.arange(start=window_size, stop=len(rawtraininglabels-1))]).reshape(-1,1)
        # Index into the raw training data with our preset windows to create datasets quickly
        labeldata = rawtraininglabels[labelindexdata].squeeze(1)
        

        #[End windowing dataset]
        
        

        #[beginning of creating test data and labels]

        #create the training data and labels
        self.trainingdata = torch.tensor(trainingdata[:self.split]).to(torch.float32)
        self.traininglabels = torch.tensor(labeldata[:self.split]).to(torch.float32)
        #can't call the iterate through a torch, so this is to create all of the weights
        

        #[end of creating training data and labels]




        #[beginning of creating validation data and labels]
        
        #create the validation data and labels
        self.valdata = torch.tensor(trainingdata[self.split:]).to(torch.float32)
        self.vallabels = torch.tensor(labeldata[self.split:]).to(torch.float32)

        #[end of creating validation data and labels]
        
        # [Inference Data]
        '''self.inference_data = torch.tensor(rawtrainingdata[len(rawtrainingdata)-window_size -1000: -1000].reshape(1,window_size,1)).to(torch.float32)
        self.inference_labels = torch.tensor(rawtrainingdata[len(rawtrainingdata)-window_size -1000: -1000].reshape(1,window_size,1)).to(torch.float32) # This is really a throw away, we just need it for the dataloaders sake
        demo = torch.tensor(rawtrainingdata[-1001: -941])


        fig, ax = plt.subplots()

        ax.plot(demo, label='prediction')
        plt.legend()
        wandb.log({"mock plot": wandb.Image(fig)})'''
        # [Creating dimension variables for easy computing on other sheets]
        
        self.training_len = self.trainingdata.shape[0] # Number of samples in the training set
        self.input_len = window_size# number of time parts
        self.channel_len = self.trainingdata.shape[2]# Number of features (Channels)
        self.output_len = 4 # classification category
        self.test_len = self.valdata.shape[0]

        
        #[End dimension variables]

    
    
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.trainingdata[index], self.traininglabels[index]
        elif self.mode == 'test':
            return self.valdata[index], self.vallabels[index]
        elif self.mode == 'inference':
            return self.inference_data[index], self.inference_labels[index]
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.trainingdata)
        if self.mode == 'test':
            return len(self.vallabels)
        if self.mode == 'inference':
            return len(self.inference_data)
        
        
