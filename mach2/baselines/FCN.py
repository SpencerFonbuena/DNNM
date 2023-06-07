import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self,n_in,n_classes):
        super(ConvNet,self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes

        self.conv1 = nn.Conv1d(16, 128, 8,1,4)
        self.bn1   = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, 5,1,2)
        self.bn2   = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, 3,2)
        self.bn3   = nn.BatchNorm1d(128)


        #nn.Conv2d(in_channels=1,out_channels=128,kernel_size=(7,1),stride=1,padding=(3,0))
        
    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        print(x.shape)

        x = F.avg_pool2d(x,2)
        print(x.shape)
        print(x.shape, x)
        x = x.reshape(1,4).type(torch.LongTensor)

        return x
    