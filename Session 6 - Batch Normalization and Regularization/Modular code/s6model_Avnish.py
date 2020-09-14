import torch.nn as nn
import torch.nn.functional as F
from regularization import GhostBatchNorm
import config


class Net(nn.Module):
    def __init__(self, batchnorm):
        super(Net, self).__init__()
        
        if (batchnorm == "GBN"):
          self.conv1 = nn.Sequential(
              
              nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), padding=0, bias = True), # output 26X26X4 : RF- 3X3
              nn.ReLU(),
              GhostBatchNorm(4, config.num_splits, weight=False), # Ghost Batch Normalization after each convolution.
              nn.Dropout2d(0.05), # dropout of 5% at each layer
              
              nn.Conv2d(4, 8, 3), # output 24X24X8 : RF - 5x5
              nn.ReLU(),
              GhostBatchNorm(8, config.num_splits, weight=False),  
              nn.Dropout2d(0.05),  # dropout of 5% at each layer

              nn.Conv2d(8, 16, 3), # output 22X22X16 : RF - 7X7
              nn.ReLU(),
              GhostBatchNorm(16, config.num_splits, weight=False),  
              nn.Dropout2d(0.05),  # dropout of 5% at each layer

              nn.MaxPool2d(2, 2)       # output 11X11X16 : RF - 8x8 
              )
          self.conv2 = nn.Sequential(
              
            
              nn.Conv2d(16, 16, 3), # output 9X9X16 : RF - 12x12
              nn.ReLU(),
              GhostBatchNorm(16, config.num_splits, weight=False),
              nn.Dropout2d(0.05), # 5% dropout

              nn.Conv2d(16, 16, 3, padding=1), # output 9X9X16 : RF - 16x16
              nn.ReLU(),
              GhostBatchNorm(16, config.num_splits, weight=False),
              nn.Dropout2d(0.05), # 5% dropout

              )
          self.conv3 = nn.Sequential(
            
              nn.Conv2d(16, 16, 3,padding=1), # output 9X9X16 : RF - 20 X 20
              nn.ReLU(),
              GhostBatchNorm(16, config.num_splits, weight=False),
              nn.Dropout2d(0.05), # 5% dropout

              nn.Conv2d(16, 10, 1), # output 7X7X10 : RF- 20 X 20
              nn.AvgPool2d(7) # output 1x1x10 : RF - 32x32

              )
        else: # batchnorm == "BN"
          self.conv1 = nn.Sequential(
              
              nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), padding=0, bias = True), # output 26X26X4 : RF- 3X3
              nn.ReLU(),
              nn.BatchNorm2d(4), # Batch Normalization after each convolution.
              nn.Dropout2d(0.05), # dropout of 5% at each layer
              
              nn.Conv2d(4, 8, 3), # output 24X24X8 : RF - 5x5
              nn.ReLU(),
              nn.BatchNorm2d(8),  # Batch Normalization after each convolution.
              nn.Dropout2d(0.05),  # dropout of 5% at each layer

              nn.Conv2d(8, 16, 3), # output 22X22X16 : RF - 7X7
              nn.ReLU(),
              nn.BatchNorm2d(16),  # Batch Normalization after each convolution.
              nn.Dropout2d(0.05),  # dropout of 5% at each layer

              nn.MaxPool2d(2, 2)       # output 11X11X16 : RF - 8x8 
              )
          self.conv2 = nn.Sequential(
              
            
              nn.Conv2d(16, 16, 3), # output 9X9X16 : RF - 12x12
              nn.ReLU(),
              nn.BatchNorm2d(16),
              nn.Dropout2d(0.05), # 5% dropout

              nn.Conv2d(16, 16, 3, padding=1), # output 9X9X16 : RF - 16x16
              nn.ReLU(),
              nn.BatchNorm2d(16),
              nn.Dropout2d(0.05), # 5% dropout

              )
          self.conv3 = nn.Sequential(
            
              nn.Conv2d(16, 16, 3,padding=1), # output 9X9X16 : RF - 20 X 20
              nn.ReLU(),
              nn.BatchNorm2d(16),
              nn.Dropout2d(0.05), # 5% dropout

              nn.Conv2d(16, 10, 1), # output 7X7X10 : RF- 20 X 20
              nn.AvgPool2d(7) # output 1x1x10 : RF - 32x32

              )

    def forward(self, x):
        
        x = self.conv1(x)
        x= self.conv2(x)
        x= self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, -1)