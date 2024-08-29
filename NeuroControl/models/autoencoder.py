import torch
from torch import nn
import torch.nn.functional as F
import pdb
from NeuroControl.models.cnn import CNNLayer, DeCNNLayer
from NeuroControl.models.mlp import MLP
from NeuroControl.custom_functions.stfuncs import STMNsampler 
from tqdm import tqdm

# def to_log_space(x, epsilon=1e-10):
#     return torch.log(torch.abs(x) + epsilon)

# An autoencoder for neural video
class NeuralAutoEncoder(nn.Module):
    def __init__(self, frame_count, per_image_discrete_latent_size_sqrt=32, cnn_kernel_size=3):#neuron_count, frame_count):
        super(NeuralAutoEncoder, self).__init__()

        # self.per_image_latent_size = neuron_count * frame_count
        #self.discrete_latent_catagories_num = discrete_latent_size_sqrt 
        self.image_n = 280
        self.per_image_discrete_latent_side_size = per_image_discrete_latent_size_sqrt
        self.per_image_latent_size = per_image_discrete_latent_size_sqrt**2
        self.frame_count = frame_count


        self.frame_seq_latent_size = self.frame_count*self.per_image_latent_size

        # make sure that self.per_image_latent_size is divisible by self.frame_count
        # assert self.per_image_latent_size % self.frame_count == 0, "self.per_image_latent_size must be divisible by self.frame_count"
        # self.channel_out = self.per_image_latent_size // self.frame_count



        self.KLloss = nn.KLDivLoss(reduction='mean') 
        self.Rloss = nn.MSELoss()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 280x280 initial image input
            CNNLayer(1, 32, cnn_kernel_size),
            nn.MaxPool2d(5, stride=5),
            # Pooled down to 56x56
            CNNLayer(32, 64, cnn_kernel_size),
            nn.MaxPool2d(4, stride=4),
            # Pooled down to 14x14
            CNNLayer(64, 256, cnn_kernel_size),
            nn.MaxPool2d(4, stride=4),
            # Pooled down to 3x3
            CNNLayer(256, self.per_image_latent_size, cnn_kernel_size),
            nn.MaxPool2d(3, stride=3),
            # Pooled down to 1x1
            nn.Flatten(),
            MLP(2, self.per_image_latent_size, self.per_image_latent_size, self.per_image_latent_size),
            # Flattened to self.channel_out
            
            # nn.Unflatten(1, (self.per_image_discrete_latent_side_size, self.per_image_discrete_latent_side_size)),
            #nn.Softmax(dim=1),
            # STMNsampler(),
            # nn.Flatten(),
        )


        self.discretizer = nn.Sequential(
            nn.Softmax(dim=1),
            STMNsampler(),
        )


        


        # Decoder
        self.decoder = nn.Sequential(
            
            MLP(2, self.per_image_latent_size, self.per_image_latent_size, self.per_image_latent_size),
            # Start with 1x1xlatent_size
            nn.Unflatten(1, (self.per_image_latent_size, 1, 1)),
            
            # Upsample to 4x4
            DeCNNLayer(self.per_image_latent_size, 128, kernel_size=4, stride=4, padding=0),
            
            # Upsample to 14x14
            DeCNNLayer(128, 32, kernel_size=4, stride=4, padding=1),
            
            # Upsample to 56x56
            DeCNNLayer(32, 8, kernel_size=4, stride=4, padding=0),
            
            # Upsample to 280x280
            DeCNNLayer(8, 1, kernel_size=5, stride=5, padding=0),
        )

        self.act = nn.Sigmoid()


    def encode(self, x):
        batch_dim = x.shape[0]

        x = x.view(batch_dim * self.frame_count, 1, self.image_n, self.image_n)
        x = self.encoder(x)

        noise = torch.rand(x.shape).to(x.device) - 0.5
        
        x = 0.99 * x + 0.01 * noise
        

        x = x.view(batch_dim * self.frame_count * self.per_image_discrete_latent_side_size, self.per_image_discrete_latent_side_size)
        x = self.discretizer(x)
        x = x.view(batch_dim, self.frame_count, self.per_image_latent_size)

        

        return x

    def decode(self, z):
        batch_dim = z.shape[0]

        z = z.view(batch_dim*self.frame_count, self.per_image_latent_size)
        z = self.decoder(z)
        z = z.view(batch_dim, self.frame_count, self.image_n, self.image_n)

        z = self.act(z)

        return z
    
    def forward(self, x):
        batch_dim = x.shape[0]
        
        x = self.encode(x)


        latent = x.view(batch_dim, self.frame_count*self.per_image_latent_size)

        

        x = self.decode(x)




        return x, latent

    def lossFunc(self, x):
        x_hat, lats = self.forward(x)
        
        return self.KLloss(torch.log(x_hat), x) + self.Rloss(x_hat, x)


    # def train_step(self, batch, optimizer):
        
    #     loss = self.loss(batch)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     return loss.item()
