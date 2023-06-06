import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.ops.stochastic_depth as std


class FCN(nn.Module):
    def __init__(self, data_in, data_out, layers=[128, 256, 512], kss=[7, 5, 3]):
        super().__init__()

        self.conv1 = nn.Conv1d(data_in, layers[1], kss[2], 1, 3)
        self.conv2 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv3 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv4 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv5 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv6 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv7 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv8 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv9 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv10 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv11 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv12 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv13 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv14 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv15 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv16 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.conv17 = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        self.convout = nn.Conv1d(layers[1], layers[1], kss[2], 1, 1)
        
        self.bn1 = nn.BatchNorm1d(layers[1])
        self.bn2= nn.BatchNorm1d(layers[1])
        self.bn3 = nn.BatchNorm1d(layers[1])
        self.bn4 = nn.BatchNorm1d(layers[1])
        self.bn5= nn.BatchNorm1d(layers[1])
        self.bn6 = nn.BatchNorm1d(layers[1])
        self.bn7 = nn.BatchNorm1d(layers[1])
        self.bn8= nn.BatchNorm1d(layers[1])
        self.bn9 = nn.BatchNorm1d(layers[1])
        self.bn10 = nn.BatchNorm1d(layers[1])
        self.bn11 = nn.BatchNorm1d(layers[1])
        self.bn12= nn.BatchNorm1d(layers[1])
        self.bn13 = nn.BatchNorm1d(layers[1])
        self.bn14 = nn.BatchNorm1d(layers[1])
        self.bn15= nn.BatchNorm1d(layers[1])
        self.bn16 = nn.BatchNorm1d(layers[1])
        self.bn17 = nn.BatchNorm1d(layers[1])
        self.bnout = nn.BatchNorm1d(layers[1])
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(layers[1], data_out)


        
    def forward(self, x):
        print(x.shape)
        x = x.transpose(-1,-2)
        print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        identity = x

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x + identity
        identity = x

        x = std(x, .1, 'batch')
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x + identity
        identity = x

        x = std(x, .3, 'batch')
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = x + identity
        identity = x

        x = std(x, .3, 'batch')
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = x + identity
        identity = x
        
        x = std(x, .5, 'batch')
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = x + identity
        identity = x

        x = std(x, .5, 'batch')
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = x + identity
        identity = x

        x = std(x, .5, 'batch')
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = x + identity
        identity = x
        
        x = std(x, .5, 'batch')
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.relu(self.bn17(self.conv17(x)))
        x = x + identity
        identity = x
        

        x = F.relu(self.bnout(self.convout(x)))
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = F.softmax(x, 1)
        return x
    