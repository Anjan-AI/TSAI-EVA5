import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05)
        ) # In: 32x32x3 | Out: 32x32x16 | RF: 3

        # CONVOLUTION BLOCK 1 With Dilation 
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), dilation=2, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05)
        ) # In: 32x32x16 | Out: 32x32x32 | RF: 7



         # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # In: 32x32x32 | Out: 16x16x32 | RF: 8

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05)
        ) # In: 16x16x32 | Out: 16x16x64 | RF: 12

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1)
        ) # In: 16x16x64 | Out: 16x16x64 | RF: 16

         # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # In: 16x16x64 | Out: 8x8x64 | RF: 18 

        # CONVOLUTION BLOCK 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05)
        ) # In: 8x8x64 | Out: 8x8x128 | RF: 26

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1)
        ) # In: 8x8x128 | Out: 8x8x128 | RF: 34

        self.pool3 = nn.MaxPool2d(2, 2) # In: 8x8x128 | Out: 4x4x128  | RF: 38 

        # Depthwise Seperable Convolution
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),  padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05)
        ) # In: 4x4x128 | Out: 4x4x128 | RF: 54

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, groups=128,  kernel_size=(3, 3), padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.1)
        ) # In: 4x4x128 | Out: 4x4x256 | RF: 72

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1)
        ) # In: 4x4x256 | output_size = 1x1x256 | RF = 96
        self.fc1 = nn.Linear(256, 10) #output_size = 10
            
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool3(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = x.view(-1,256)
        x = self.fc1(x)
        return x