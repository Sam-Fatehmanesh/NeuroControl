import torch
from torch import nn
import torch.nn.functional as F
import pdb
from NeuroControl.models.cnn import CNNLayer, DeCNNLayer
from NeuroControl.models.mlp import MLP
from NeuroControl.custom_functions.stfuncs import STMNsampler 

# An autoencoder for neural video
class NeuralAutoEncoder(nn.Module):
    def __init__(self, frame_count, discrete_latent_size_sqrt=32, cnn_kernel_size=3):#neuron_count, frame_count):
        super(NeuralAutoEncoder, self).__init__()

        # self.latent_size = neuron_count * frame_count
        #self.discrete_latent_catagories_num = discrete_latent_size_sqrt 
        self.discrete_latent_side_size = discrete_latent_size_sqrt
        self.latent_size = discrete_latent_size_sqrt**2

        self.loss = nn.MSELoss()
        
        # Encoder
        self.cnn_encoder = nn.Sequential(
            # 280x280 initial image input
            CNNLayer(frame_count, 32, cnn_kernel_size),
            nn.MaxPool2d(5, stride=5),
            # Pooled down to 56x56
            CNNLayer(32, 128, cnn_kernel_size),
            nn.MaxPool2d(4, stride=4),
            # Pooled down to 14x14
            CNNLayer(128, 64, cnn_kernel_size),
            nn.MaxPool2d(4, stride=4),
            # Pooled down to 3x3
            CNNLayer(64, self.latent_size, cnn_kernel_size),
            nn.MaxPool2d(3, stride=3),
            # Pooled down to 1x1
            nn.Flatten(),
            # Flattened to self.latent_size
            MLP(2, self.latent_size, self.latent_size, self.latent_size),
            nn.Unflatten(1, (self.discrete_latent_side_size, self.discrete_latent_side_size)),
            nn.Softmax(dim=1),
            STMNsampler(),
            nn.Flatten(),
        )





        # Decoder
        self.decoder = nn.Sequential(
            MLP(2, self.latent_size, self.latent_size, self.latent_size),
            # Start with 1x1xlatent_size
            nn.Unflatten(1, (self.latent_size, 1, 1)),
            
            # Upsample to 3x3
            DeCNNLayer(self.latent_size, 12, kernel_size=3, stride=3, padding=0),
            
            # Upsample to 14x14
            DeCNNLayer(64, 128, kernel_size=4, stride=4, padding=0),
            
            # Upsample to 56x56
            DeCNNLayer(128, 8, kernel_size=4, stride=4, padding=0),
            
            # Upsample to 280x280
            DeCNNLayer(8, 1, kernel_size=5, stride=5, padding=0),
    
        )


    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        x = self.encode(x)

        latents = x

        x = self.decode(x)
        return x, latents

    # def loss(self, x):
    #     x_hat, lats = self.forward(x)
    #     return F.mse_loss(x_hat, x), lats


    # def train_step(self, batch, optimizer):
        
    #     loss = self.loss(batch)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     return loss.item()
