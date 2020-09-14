import torch.nn as nn
import torch.nn.functional as F
from regularization import GhostBatchNorm
import config



class Net(nn.Module):
    def __init__(self, batchnorm):
        super(Net, self).__init__()
        if (batchnorm == "GBN"):
            # Input Block
            self.convblock1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
                nn.ReLU(),
                #nn.BatchNorm2d(10),
                GhostBatchNorm(num_features =10, num_splits=config.num_splits),
                nn.Dropout(config.dropout)
            ) # output_size = 28  RF : 3

            # CONVOLUTION BLOCK 1
            self.convblock2 = nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=24, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                #nn.BatchNorm2d(24),
                GhostBatchNorm(num_features =24, num_splits=config.num_splits),
                nn.Dropout(config.dropout)
            ) # output_size = 26   RF : 5

            # TRANSITION BLOCK 1
            self.convblock3 = nn.Sequential(
                nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
                nn.ReLU() 
            )  # output_size = 26  RF 5
            self.pool1 = nn.MaxPool2d(2, 2) # output_size = 13  RF 6

            self.convblock4 = nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
                nn.ReLU(),
            # nn.BatchNorm2d(10),
                GhostBatchNorm(num_features =10, num_splits=config.num_splits),
                nn.Dropout(config.dropout)
            ) # output_size = 13   RF : 10

            self.convblock41 = nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                #nn.BatchNorm2d(16),
                GhostBatchNorm(num_features =16, num_splits=config.num_splits),
                nn.Dropout(config.dropout)
            ) #output_size = 11  RF : 14
        
            self.convblock5 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
            # nn.BatchNorm2d(16),
                GhostBatchNorm(num_features =16, num_splits=config.num_splits),
                nn.Dropout(config.dropout)
            ) # output_size = 9  RF :18

            self.convblock6 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), 
                nn.ReLU(),
                #nn.BatchNorm2d(16),
                GhostBatchNorm(num_features =16, num_splits=config.num_splits),
                nn.Dropout(config.dropout)
            ) # output_size = 7 RF 22

            
            self.gap1 = nn.Sequential(
                nn.AvgPool2d(kernel_size =(7,7))
            ) # output_size = 1  RF 34
            #  # Output BLOCK 
            self.convblock7 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            
            ) # output_size = 1  RF 34

        else:
            # Input Block
            self.convblock1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(10),
                #GhostBatchNorm(num_features =10, num_splits=2),
                nn.Dropout(config.dropout)
            ) # output_size = 28  RF : 3

            # CONVOLUTION BLOCK 1
            self.convblock2 = nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=24, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(24),
                #GhostBatchNorm(num_features =24, num_splits=2),
                nn.Dropout(config.dropout)
            ) # output_size = 26   RF : 5

            # TRANSITION BLOCK 1
            self.convblock3 = nn.Sequential(
                nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
                nn.ReLU() 
            )  # output_size = 26  RF 5
            self.pool1 = nn.MaxPool2d(2, 2) # output_size = 13  RF 6

            self.convblock4 = nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(10),
                #GhostBatchNorm(num_features =10, num_splits=2),
                nn.Dropout(config.dropout)
            ) # output_size = 13   RF : 10

            self.convblock41 = nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                #GhostBatchNorm(num_features =16, num_splits=2),
                nn.Dropout(config.dropout)
            ) #output_size = 11  RF : 14
        
            self.convblock5 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Dropout(config.dropout)
            ) # output_size = 9  RF :18

            self.convblock6 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), 
                nn.ReLU(),
                nn.BatchNorm2d(16),
                #GhostBatchNorm(num_features =16, num_splits=2),
                nn.Dropout(config.dropout)
            ) # output_size = 7 RF 22

            
            self.gap1 = nn.Sequential(
                nn.AvgPool2d(kernel_size =(7,7))
            ) # output_size = 1  RF 34
            #  # Output BLOCK 
            self.convblock7 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            
            ) # output_size = 1  RF 34
       

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock41(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap1(x)
        x = self.convblock7(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

