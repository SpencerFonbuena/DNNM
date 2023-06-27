import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

df = pd.read_csv('/root/DNNM/mach1/datasets/SPY_full_1min_adjsplit.txt', sep=',', index_col=0, header=None, names=["Date", 'Open', 'High', 'low', 'Close', 'Volume'])

#50 period moving average
df['50SMA'] = df['Close'].rolling(50).mean()

#200 period moving average
df['200SMA'] = df['Close'].rolling(200).mean()

def RSI(df, lookback):
    deltas = np.diff(df)
    seed = deltas[:lookback+1]
    up = seed[seed>= 0].sum()/lookback
    down = -seed[seed < 0].sum()/lookback
    rs = up/down
    rsi = np.zeros_like(df)
    
    for i in range(lookback, len(df)):
        delta = deltas[i-1]

        if delta > 0:
            upval = delta
            downval = 0.
        if delta < 0:
            upval = 0
            downval=abs(delta)
        up = (up * (lookback - 1) + upval) / lookback
        down = (down * (lookback - 1) + downval) / lookback

        rs = up/down
        rsi[i] = 100. - 100./(1. +rs)

    return rsi
df['RSI'] = RSI(df['Close'], 14)
    
def create_labels(df):

    A = 0
    C = 0
    labels = np.array([])
    
    print(df)
    for i in range(0, (len(df))):


        #find 1 percent and 2 percent above and below
        #print(df[A])
        one_low = df[A] * .99
        two_low = df[A] * .98
        one_high = df[A] * 1.01
        two_high = df[A] * 1.02

        #print(f'1 low: {one_low} | 2 low: {two_low} | 1 high: {one_high} | 2 high: {two_high}')
        #initialize the label counter
        label_counter = A

        #this is to make sure that once it either enters the "gone up by one percent" or "gone down by 1 percent"
        #it doesn't enter the other while loops
        pathway = 0

        try:
            #look for the instance when the price increases or decreases by 1 percent
            while df[label_counter] >= one_low and df[label_counter] <= one_high:
                label_counter += 1
                #print(df[label_counter])
            #If the price moved up 1 pecent first, this while loop will trigger and check if it is a two to one, or a one to one trade
            while df[label_counter] >= one_low and df[label_counter] <= two_high:
                label_counter += 1
                pathway = 1
                #print(df[label_counter])
            #Check if price has increased two percent
            if df[label_counter] >= two_high:
                labels = np.append(labels, 2)
                pathway = 1
                #print(df[label_counter])
            #check if price has reversed back down to the one percent marker
            if df[label_counter] <= one_low and pathway == 1:
                labels = np.append(labels, 1)
                #print(df[label_counter])
            
            #if the price moved down 1 pecent first, this will check if it is a two to one, or a one to one trade
            while df[label_counter] <= one_high and df[label_counter] >= two_low and pathway != 1:
                label_counter += 1
                pathway = 2
                #print(df[label_counter])
        
            #check if the price has continued down two percent
            if df[label_counter] <= two_low and pathway != 1:
                labels = np.append(labels, 0)
                #print(df[label_counter])
            #check if price reversed back up to the 1 percent above marker
            if df[label_counter] >= one_high and pathway != 1:
                labels = np.append(labels, 1)
                #print(df[label_counter])
            
            #temporarily store the last label that was added to "labels=[]"
            C = labels[-1]

        except:
            break
        #increment the graph by one time interval
        A += 1 

    #Create an array with the last value before the classification algorithm stopped
    array_append = []
    while A < len(df):
        array_append = np.append(array_append, C)
        A += 1
        

    labels = np.append(labels, array_append)
    return labels
df['Labels'] = create_labels(df['Close'])

dataframe = pd.DataFrame()

dataframe['Open'] = df['Open'].pct_change()
dataframe['High'] = df['High'].pct_change()
dataframe['Low'] = df['low'].pct_change()
dataframe['Close'] = df['Close'].pct_change()
dataframe['Volume'] = df['Volume']
dataframe['50SMA'] = df['50SMA'].pct_change()
dataframe['200SMA'] = df['200SMA'].pct_change()
dataframe['RSI'] = df['RSI'].pct_change()
dataframe['Labels'] = df['Labels']

print(dataframe['Labels'].value_counts())

dataframe.to_csv('/root/DNNM/mach1/datasets/ES_1min_dataset')