import torch
from torch import nn
import pdb

class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(CNNLayer, self).__init__()
        #pdb.set_trace()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
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

        res = self.match_dimensions(res)
        #pdb.set_trace()
        x = x + res

        x = self.activation2(x)


        return x

# Deconvolutional layer
class DeCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DeCNNLayer, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1)
        #self.match_dimensions = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        #pdb.set_trace()
        #res = x

        x = self.deconv1(x)
        x = self.batchnorm1(x)

        x = self.activation1(x)


        x = self.deconv2(x)
        x = self.batchnorm2(x)
        #pdb.set_trace()
        #x += self.match_dimensions(res)

        x = self.activation2(x)

        return x

