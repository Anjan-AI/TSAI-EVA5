import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input: 32x32x3 Out: 32x32x64 RF:3
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) 
        # Input: 32x32x64 Out: 32x32x128 RF:5        
        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Input: 32x32x128 Out: 32x32x128 RF:9
        self.res_block_1 = self.return_conv_block(128)

        # Input: 32x32x128 Out: 16x16x256 RF:11+1=12
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                    kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # Input: 16x16x256 Out: 8x8x512 RF:(12+4) -> (12+4)+2 = 18 
        self.x3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                    kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # Input: 8x8x512 Out: 8x8x512 RF:18+6+6 = 30  
        self.res_block_2 = self.return_conv_block(512)
        # Input: 8x8x512 Out: 2x2x512 RF:30+4 = 34  
        self.pool4 = nn.MaxPool2d(4, 4)

        self.fc = nn.Linear(in_features=512, out_features=10, bias=False)

    def return_conv_block(self, num_kernels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels,
                        kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(num_kernels),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels,
                        kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(num_kernels),
            nn.ReLU(),
        )

    def forward(self, x):
        prep_layer = self.prep_layer(x)
        x1 = self.x1(prep_layer)
        r1 = self.res_block_1(x1)
        layer1 = x1+r1
        layer2 = self.layer2(layer1)
        x3 = self.x3(layer2)
        r2 = self.res_block_2(x3)
        layer3 = x3+r2
        pool4 = self.pool4(layer3)
        x = pool4.view(-1, 512)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        return x
