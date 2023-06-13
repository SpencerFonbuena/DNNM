import numpy as np
from sklearn.linear_model import RidgeClassifierCV
import pandas as pd
import torch

from rocket_functions import generate_kernels, apply_kernels

# [Hyperparameters]

window_size = 120

# [End]

# [Import the dataset]


'''if torch.cuda.is_available():
    path = '/root/GTN/mach1/datasets/SPY_30mins.txt'
else:
    path = 'models/mach1/datasets/SPY_30mins.txt'''

if torch.cuda.is_available():
    path = '/root/GTN/mach1/datasets/SPY_30mins.txt'
else:
    path = '/Users/spencerfonbuena/Documents/Python/Trading Models/models/mach1/datasets/AAPL_1hour_expand_batch.txt'

df = pd.read_csv(path, delimiter=',', index_col=0)
#Create the training and label datasets
labeldata = df['Labels'].to_numpy()[:15]
labeltest = df['Labels'].to_numpy()[15:]
#normalize the data inputs
prerawtrain = torch.nn.functional.normalize(torch.tensor(df.drop(columns=['Labels']).to_numpy()))
#prerawtrain = torch.tensor(df.drop(columns=['Labels']).to_numpy())
#recasting data as pandas dataframe. I couldn't find a way to normalize with pandas, so I cast it first to torch, then back to pandas.
trainingdata = pd.DataFrame(prerawtrain).to_numpy()[:15]
testdata = pd.DataFrame(prerawtrain).to_numpy()[15:]



# [End import]

# [Begin Rocket]

    # [Training loop]
kernels = generate_kernels(trainingdata.shape[1], 100)

x_training_transform = apply_kernels(trainingdata, kernels)

classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))

classifier.fit(x_training_transform, labeldata)

print(end = "") # suppress print output of classifier.fit(...)

    #[End Training]

    # [Test Loop]
x_test_transform = apply_kernels(testdata, kernels)

predictions = classifier.predict(x_test_transform)

print(f"predictions = {', '.join(predictions.astype(str))}")
print(f"accuracy    = {(predictions == labeltest).mean()}")