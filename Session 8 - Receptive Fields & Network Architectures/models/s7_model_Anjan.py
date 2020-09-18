import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.Layer1 = nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=2, bias=False),  # output_size = 34  RF : 3
            nn.ReLU(),
            nn.BatchNorm2d(32), 

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False), # output_size = 36  RF : 5
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), groups=64, padding=1,bias=False),   # output_size = 36  RF : 7
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1),bias=False),   # output_size = 36  RF : 7
            nn.ReLU(),
            nn.BatchNorm2d(128),

              # Dilated convolution
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=2, dilation=2,bias=False),  # output_size = 36  RF : 11
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1,bias=False),  # output_size = 36  RF : 13
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),   # output_size = 36  RF : 13
            nn.ReLU(),

            nn.MaxPool2d(2, 2)  # # output_size = 18  RF : 14
           
        )
      
      self.Layer2 = nn.Sequential(
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False), # output_size = 20  RF : 18
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), groups=64, padding=1,bias=False),   # output_size = 20  RF : 22
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1),bias=False),   # output_size = 20  RF : 22
            nn.ReLU(),
            nn.BatchNorm2d(128),

              # Dilated convolution
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=2, dilation=2,bias=False),  # output_size = 20  RF : 30
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1,bias=False),  # output_size = 20  RF : 34
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),   # output_size = 20  RF : 34

            nn.MaxPool2d(2, 2)  # # output_size = 10  RF : 35
           
        )
      
      self.Layer3 = nn.Sequential(
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False), # output_size = 12  RF : 43
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), groups=64, padding=1,bias=False),   # output_size = 12  RF : 51
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1),bias=False),   # output_size = 12  RF : 51
            nn.ReLU(),
            nn.BatchNorm2d(128),

              # Dilated convolution
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=2, dilation=2,bias=False),  # output_size = 12  RF : 67
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1,bias=False),  # output_size = 12  RF : 75
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),   # output_size = 12  RF : 75
            nn.ReLU(),

            nn.MaxPool2d(2, 2)  # # output_size = 6  RF : 150
           
        )
      
      self.gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size =(6,6))  # # output_size = 6  RF : 190
       )
         
        #  # Output BLOCK
       
      self.conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )
      
       
 

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x =  self.gap1(x)
        x = self.conv(x)
        x = x.view(-1, 10)
        return x

