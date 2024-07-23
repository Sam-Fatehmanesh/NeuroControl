import torch
from torch import nn
from NeuroControl.models.cnn import CNNLayer
from NeuroControl.models.transformer import Transformer

class TransCNN(nn.Module):
    def __init__(self, num_input_frames, image_n, dim, out_dim, num_trans_layers, cnn_kernel_size=3):
        super(TransCNN, self).__init__()
        assert cnn_kernel_size % 2 != 0, "kernel size must be odd"

        self.enCNN = nn.Sequential(
            CNNLayer(num_input_frames, 16, cnn_kernel_size),
            CNNLayer(16, 128, cnn_kernel_size),
            CNNLayer(128, 256, cnn_kernel_size),
            CNNLayer(256, 256, cnn_kernel_size),
        )

        predim = (image_n**2)*256

        self.flat = nn.Flatten()

        self.transformer = Transformer(predim, dim, 32, dim*2, out_dim, num_trans_layers)



    def forward(self, x):

        x = self.enCNN(x)
        x = self.flat(x)
        x = self.transformer(x)

        return x



