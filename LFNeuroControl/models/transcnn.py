import torch
from torch import nn
from LFNeuroControl.models.cnn import CNNLayer
from LFNeuroControl.models.transformer import STBTransformer

class TransCNN(nn.Module):
    def __init__(self, num_input_frames, image_n, dim, out_dim, num_trans_layers, cnn_kernel_size=3):
        super(CNNTrans, self).__init__()
        assert kernel_size % 2 != 0, "kernel size must be odd"

        self.enCNN = nn.Sequential(
            CNNLayer(num_input_frames, 16, kernel_size),
            CNNLayer(16, 128, kernel_size),
            CNNLayer(128, 256, kernel_size),
            CNNLayer(256, 64, kernel_size),
            CNNLayer(64, 16, kernel_size),
        )

        predim = (image_n-(len(self.enCNN)*(kernel_size-1)))**2

        self.flat = nn.Flatten()

        self.transformer = STBTransformer(predim, dim, 32, dim*2, out_dim, num_trans_layers)



    def forward(self, x):

        x = self.enCNN(x)
        x = self.flat(x)
        x = self.transformer(x)

        return x



