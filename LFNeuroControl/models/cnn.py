import torch
from torch import nn

class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SimpleCNNLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        #self.pool = nn.MaxPool2d(pool_size, stride=pool_stride if pool_stride else pool_size)
        self.match_dimensions = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.batchnorm1(x)

        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)

        x += self.match_dimensions(res)

        x = self.activation2(x)


        return x


