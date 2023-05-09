#Gated-Transformer-on-MTS
Based on Pytorch, the improved Transformer model is applied to the classification task of multidimensional time series


## lab environment
Environment|Description|
---|---------|
Language|Python3.7|
Framework|Pytorch1.6|
IDE|Pycharm and Colab|
Device|CPU and GPU|

## data set
Multivariate time series data set, the file is in .mat format, the training set and the test set are in one file, and it is pre-defined as the test set data, the test set label, the training set data and the training set label. <br>
The data set is downloaded using Baidu cloud disk, the link is as follows:<br>
  Link: https://pan.baidu.com/s/1u2HN6tfygcQvzuEK5XBa2A <br>
  Extraction code: dxq6 <br>
---

 
## Data preprocessing
For the detailed dataset processing process, refer to the dataset_process.py file. <br>
- Create a torch.utils.data.Dataset object, process the dataset in the class, and its member variables define training set data, training set labels, test set data, test set labels, etc. Create a torch.utils.data.DataLoader object to generate a random shuffle of the mini-batch and dataset during training<br>
- The Time series Length of different samples in the data set is different. When processing, use the **longest** time step of all samples (test set and training set) as the Time series Length, and fill it with **0**. <br>
- During the data set processing, the training set data and test set data without padding are saved, and the sample list of the longest time step in the test set is used for the exploration model. <br>
- The labels in the NetFlow dataset are **1 and 13**, and the returned label values ​​must be processed when using this dataset. <br>

## Model Description

<img src="https://github.com/SY-Ma/Gated-Transformer-on-MTS/blob/main/images/GTN%20structure.png" style="zoom:50%">

- Only use Encoder: Since it is a classification task, the model deletes the decoder in the traditional Transformer, **only uses Encoder** for classification
- Two Tower: There are obviously many connections between different Steps or between different Channels. The traditional Transformer uses the Attentino mechanism to focus on the degree of correlation between different steps or channels, but only selects one for calculation. Unlike the CNN model processing time series, it can use two-dimensional convolution kernels to focus on step-wise and channel-wise at the same time. Here we use the **Two Towers** model, which calculates step-wise Attention and channel-wise Attention at the same time .
- Gate mechanism: For different data sets, different Attention mechanisms are good or bad. For the results of the feature extraction of the two towers, the simple method is to splicing the output of the two towers. But here, we use The model learns two weight values, and assigns weights to the output of each tower, the formula is as follows. <br>
    `h = W Concat(C, S) + b` <br>
    `g1, g2 = Softmax(h)` <br>
    `y = Concat(C · g1, S · g2)` <br>
- In step-wise, the model adds position encoding and mask mechanism like the traditional Transformer, while in channel-wise, the model discards position encoding and mask, because these two mechanisms have no practical significance for channels without time characteristics .


## Hyperparameter description
Hyperparameter|Description|

d_model|The model deals with time series rather than natural language, so the encoding of words in NLP is omitted, and only one linear layer is used to map into a dense vector of d_model dimension. In addition, d_model guarantees the dimension of the place where each module connects the same |

d_hidden|Dimensions of hidden layers in Position-wise FeedForword|

d_input|The length of the time series is actually the dimension of the longest time step in a data set **fixed**, directly determined by the data set preprocessing|

d_channel|The number of time channels of the multivariate time series, that is, the time series of several dimensions **fixed**, directly determined by the data set preprocessing|

d_output|Number of classification categories **fixed**, directly determined by dataset preprocessing|

q,v|Linear layer mapping dimension in Multi-Head Attention|

h|Number of heads in Multi-Head Attention|

N|Number of Encoders in the Encoder stack|

dropout|random dropout|

EPOCH|Number of training iterations|

BATCH_SIZE|mini-batch size|

LR|learning rate is defined as 1e-4|

optimizer_name|optimizer choice recommends **Adagrad** and Adam|

## File Description
file name|description|
-------|----|
dataset_process|dataset processing|
font|Storage font, used for the text in the result image|
gather_figure|clustering result figure|
heatmap_figure_in_test|Heat map of the score matrix drawn when testing the model|
module|Each module of the model|
mytest|Various test codes|
reslut_figure|Accuracy result figure|
saved_model|saved pkl file|
utils|utility files|
run.py|training model|
run_with_saved_model.py|Test results using the trained model (saved as a pkl file)|

## utils tool description
Briefly introduce a few
- random_seed: Used to set **random seed**, so that the results of each experiment can be reproduced.
- heatMap.py: **heat map** used to draw the score matrix of the twin towers, used to analyze the degree of correlation between channel and channel or between step and step, and **DTW** for comparison Matrix and Euclidean distance matrix, used to analyze the factors that determine the weight distribution.
- draw_line: It is used to draw a line chart, and it is generally necessary to customize a new function for drawing as needed.
- visualization: It is used to draw the loss change curve and accuracy change curve of the training model to judge whether it is converged and overfitting.
- TSNE: **Dimensionality reduction clustering algorithm** and draw a cluster diagram to evaluate the effect of model feature extraction or the similarity between time series.

## Tips
- The .pkl file needs to be trained first and saved at the end of the training (set the parameter to True). Due to github's limitation on the file size, the uploaded file does not include the trained .pkl file.
- The .pkl file is saved with pytorch version 1.6 on pycharm and pytorch 1.7 on colab. If you want to load the model directly for testing, you need to use a version of pytorch higher than or equal to version 1.6 as much as possible.
- Root directory files such as saved_model, reslut_figure are the default path for saving, please do not delete or modify the name, unless the path is directly modified in the source code.
- Please use the dataset provided by Baidu Cloud Disk. Different MTS datasets have different file formats. This dataset deals with .mat files.
- A tool class in utils. When drawing colored curves and clustering diagrams, for the division of colors in the diagrams, since the requirements cannot be generalized, please write the code definition in the function.
- The .pkl file saved by save model is continuously updated during the iterative process. At the end, the model with the highest accuracy is saved and named. Do not modify the naming format, because in run_with_saved_model.py, the information in the file naming will be used. The naming of the drawing results will also refer to the information in it.
- GPU is preferred, CPU is used if none.

## refer to
```
[Wang et al., 2017] Z. Wang, W. Yan, and T. Oates. Time series classification from scratch with deep neural networks: A strong baseline. In 2017 International Joint Conference on Neural Networks (IJCNN), pages 1578– 1585, 2017.
```

## My knowledge is shallow, if the code and text are inappropriate, please criticize and correct me!
## Contact: masiyuan007@qq.com
