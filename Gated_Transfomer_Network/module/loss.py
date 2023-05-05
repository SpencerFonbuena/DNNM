# @Time    : 2021/07/15 20:33
# @Author  : SY.M
# @FileName: loss.py


import torch

#creating a subclass "MyLoss" of base class torch.nn.module. This will give it added functionality
class Myloss(torch.nn.Module):
    def __init__(self):
        #the (Myloss, self) is not necessaryt technically, you could just put in super()
        super(Myloss, self).__init__()
        #At initializiation, this is creating an attribute of MyLoss and instantiating it to be crossentropyloss
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, pre, target):
        
        #This is assigning attributes, which must be a part of the defintion of crossentropyloss
        loss = self.loss_function(pre, target.long())

        return loss