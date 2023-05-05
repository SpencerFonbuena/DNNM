# @Time    : 2021/07/15 20:20
# @Author  : SY.M
# @FileName: train_and_validate.py


import torch
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import KFold

from config.param_config import Config as c
from utils.random_seed import setup_seed
from module.loss import Myloss
from data_process.create_dataset import Create_Dataset, Stratified_Dataset
from utils.update_csv import update_validate
from module.for_MTS.transformer import Transformer

# from utils.save_model import save_model


#This is here to make use of a GPU if it is available, and then print out what GPU or device we are using
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Use device {DEVICE}')
# set random seed
setup_seed(30)


def train_and_validate(net_class: torch.nn.Module,
                       all_data: Create_Dataset):
    """
    Use all the default training set data to build k-fold cross-validation, use the divided training set for training, and use the divided verification set for verification for hyperparameter tuning
    :param net_class: the name of the class of the model to use
    :param all_data: object containing all required data
    :return: None
    """
    # Create a loss function
    loss_function = Myloss()

    #Make the progress bar pretty
    pbar = tqdm(total=c.EPOCH * c.k_fold)
    # record max accuracy on validate data set of each fold
    max_validate_acc_list = []  

    #bring in the splits?
    sfk = all_data.sfk_indexset
    begin_time = time()
    # k-fold cross-validation loop
    for n_fold, (train_index, val_index) in enumerate(sfk):

        # Create training and validation sets based on the stratified k fold splits
        train_X, val_X = all_data.dataset[train_index], all_data.dataset[val_index]
        train_Y, val_Y = all_data.labels[train_index], all_data.labels[val_index]
        #Create the two separate datasets for training and validation
        train_dataset = Stratified_Dataset(X=train_X, Y=train_Y)
        val_dataset = Stratified_Dataset(X=val_X, Y=val_Y)
        #train_dataset = Create_Dataset(datafile=f'{c.datafile}', window_size=100)[:(len(Create_Dataset) * .7)]
        #val_dataset = Create_Dataset(datafile=f'{c.datafile}', window_size=100)[(len(Create_Dataset) * .7):]
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=c.BATCH_SIZE, shuffle=False, drop_last=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=c.BATCH_SIZE, shuffle=False, drop_last=True)
        # create network. This represents the transformer
        net = net_class(q=c.q, v=c.v, h=c.h, N=c.N, d_model=c.d_model, d_hidden=c.d_hidden,
                        d_feature=c.d_feature, d_timestep=c.window_size, class_num=4).to(DEVICE)

        # create optimizer
        optimizer = None
        if c.optimizer_name == 'Adagrad':
            optimizer = torch.optim.Adagrad(net.parameters(), lr=c.LR)
        elif c.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=c.LR)
        assert optimizer is not None, 'optimizer is None!'

        loss_list = []
        accuracy_on_train = []
        accuracy_on_validate = []


        net.train()
        loss_sum_min = 99999
        best_net = None
        counter = 0
        for index in range(c.EPOCH):
            loss_sum = 0.0
            for x, y in train_dataloader: #This loop right now takes 45 seconds to complete
                optimizer.zero_grad()
                y_pre = net(x.to(DEVICE), 'train')
                loss = loss_function(y_pre, y.to(DEVICE))
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
                counter += 1
                print(counter)
            if loss_sum < loss_sum_min:
                loss_sum_min = loss_sum
                best_net = copy.deepcopy(net)

            print(f'No.{1+n_fold}fold EPOCH:{index + 1}\t\tLoss:{round(loss_sum, 5)}')
            loss_list.append(loss_sum)
            pbar.update()

            '''
            if (index + 1) % c.test_interval == 0:
                trian_acc, val_acc = validate(net=net, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
                accuracy_on_train.append(trian_acc)
                accuracy_on_validate.append(val_acc)
                print(f' {1+n_fold} fold current maximum accuracy validation set: {max(accuracy_on_validate)}\t training set: {max(accuracy_on_train)}')
            '''

        trian_acc, val_acc = validate(net=best_net, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
        print(f' {1 + n_fold} fold accuracy verification set: {trian_acc}\t training set: {val_acc}')
        # max_validate_acc_list.append(max(accuracy_on_validate))
        max_validate_acc_list.append(val_acc)

    end_time = time()
    time_cost = round((end_time - begin_time) / 60, 2)

    print(f'\033[0;34m acc in each fold:{max_validate_acc_list}\033[0m')
    print(f'\033[0;34m max_acc:{max(max_validate_acc_list)}    mean_acc:{np.mean(max_validate_acc_list)}    variance:{np.var(max_validate_acc_list)}\033[0m')
    print(f'\033[0;34m total time cost:{time_cost} min\033[0m')

    update_validate(max_validate_acc_list=max_validate_acc_list)

def validate(net,
             train_dataloader: torch.utils.data.DataLoader,
             val_dataloader: torch.utils.data.DataLoader):
    """
    calculate acc
    :param net: trained model
    :param train_dataloader: train_dataloader
    :param val_dataloader: validate_dataloader
    :return: training set acc, validation set acc
    """
    with torch.no_grad():
        net.eval()

        # test validation set
        correct = 0
        total = 0
        for x_val, y_val in val_dataloader:
            pre = net(x_val.to(DEVICE), 'test')
            _, pre_index = torch.max(pre.data, dim=-1)
            total += pre.shape[0]
            correct += torch.sum(torch.eq(pre_index, y_val.long().to(DEVICE))).item()
        accuracy_on_validate = round(correct / total, 4) * 100
        print(f'accuracy on validate:{accuracy_on_validate}%')

        # test training set
        correct = 0
        total = 0
        for x_train, y_train in train_dataloader:
            pre = net(x_train.to(DEVICE), 'test')
            _, pre_index = torch.max(pre.data, dim=-1)
            total += pre.shape[0]
            correct += torch.sum(torch.eq(pre_index, y_train.long().to(DEVICE))).item()
        accuracy_on_train = round(correct / total, 4) * 100
        print(f'accuracy on train:{accuracy_on_train}%')

        return accuracy_on_train, accuracy_on_validate

train_and_validate(Transformer, Create_Dataset(datafile='/Users/spencerfonbuena/Desktop/AAPL_1hour_expanded_test.txt', window_size=c.window_size))