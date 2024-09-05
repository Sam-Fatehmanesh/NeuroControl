import torch
from torch import nn
import torch.nn.functional as F
import pdb
from NeuroControl.models.cnn import CNNLayer, DeCNNLayer
from NeuroControl.models.mlp import MLP
from NeuroControl.custom_functions.utils import STMNsampler, symlog, symexp
from tqdm import tqdm

# def to_log_space(x, epsilon=1e-10):
#     return torch.log(torch.abs(x) + epsilon)

# An autoencoder for neural video
class NeuralAutoEncoder(nn.Module):
    def __init__(self, frame_count, image_n, hidden_state_size, per_image_discrete_latent_size_sqrt=32, cnn_kernel_size=3):#neuron_count, frame_count):
        super(NeuralAutoEncoder, self).__init__()

        # self.per_image_latent_size = neuron_count * frame_count
        #self.discrete_latent_catagories_num = discrete_latent_size_sqrt 
        self.image_n = image_n
        self.per_image_discrete_latent_side_size = per_image_discrete_latent_size_sqrt
        self.per_image_latent_size = per_image_discrete_latent_size_sqrt**2
        self.frame_count = frame_count
        self.hidden_state_size = hidden_state_size


        self.frame_seq_latent_size = self.frame_count*self.per_image_latent_size

        # make sure that self.per_image_latent_size is divisible by self.frame_count
        # assert self.per_image_latent_size % self.frame_count == 0, "self.per_image_latent_size must be divisible by self.frame_count"
        # self.channel_out = self.per_image_latent_size // self.frame_count



        self.KLloss = nn.KLDivLoss(reduction='mean') 
        self.Rloss = nn.MSELoss()

        self.post_cnn_encoder_size = 8**2#14**2

        #self.pre_dcnn_decoder_size = 14**2

        # Encoder
        self.cnn_encoder = nn.Sequential(
            # 280x280 initial image input
            CNNLayer(1, 16, cnn_kernel_size),
            nn.AvgPool2d(3, stride=3),
            # Pooled down to 56x56
            CNNLayer(16, 128, cnn_kernel_size),
            nn.AvgPool2d(4, stride=4),
            # Pooled down to 14x14
            CNNLayer(128, 1, cnn_kernel_size),
            #CNNLayer(128, 1, cnn_kernel_size),

            nn.Flatten(),
            

        )

        self.mlp_encoder = MLP(3, self.post_cnn_encoder_size + hidden_state_size, self.per_image_latent_size, self.per_image_latent_size)


        self.discretizer = nn.Sequential(
            nn.Softmax(dim=1),
            STMNsampler(),
        )


        


        # Decoder
        self.decoder = nn.Sequential(
            
            MLP(3, self.per_image_latent_size + hidden_state_size, self.per_image_latent_size, self.per_image_latent_size),

            nn.Unflatten(1, (1, self.per_image_discrete_latent_side_size, self.per_image_discrete_latent_side_size)),

            DeCNNLayer(1, 128, kernel_size=4, stride=4, padding=1),
            
            DeCNNLayer(128, 32, kernel_size=4, stride=4, padding=0),
            
            DeCNNLayer(32, 1, kernel_size=5, stride=5, padding=0),
            
            nn.Upsample(size=(self.image_n, self.image_n), mode='bilinear'),
        )

        #self.act = nn.Sigmoid()


    def encode(self, x, h_t):
        batch_dim = x.shape[0]

        x = x.view(batch_dim * self.frame_count, 1, self.image_n, self.image_n)

        x = self.cnn_encoder(x)
        h_t = torch.tile(h_t, (self.frame_count, 1))
        x = torch.cat( (x, h_t), dim=1)
        x = self.mlp_encoder(x)


        # Adding noise to make non deterministic
        x = 0.99 * x + 0.01 * (torch.rand(x.shape).to(x.device) - 0.5)


        x = x.view(batch_dim * self.frame_count * self.per_image_discrete_latent_side_size, self.per_image_discrete_latent_side_size)
        x = self.discretizer(x)
        x = x.view(batch_dim, self.frame_count, self.per_image_latent_size)

        
        
        return x

    def decode(self, z, h_t):
        batch_dim = z.shape[0]
        #pdb.set_trace()
        z = z.view(batch_dim*self.frame_count, self.per_image_latent_size)
        h_t = torch.tile(h_t, (self.frame_count, 1))
        #pdb.set_trace()
        z = torch.cat( (z, h_t), dim=1)

        z = self.decoder(z)
        z = z.view(batch_dim, self.frame_count, self.image_n, self.image_n)


        return z
    
    def forward(self, x, h_t):
        batch_dim = x.shape[0]
        
        #pdb.set_trace()
        x = self.encode(x, h_t)


        latent = x.view(batch_dim, self.frame_count*self.per_image_latent_size)

        

        x = self.decode(x, h_t)


        #x = self.act(x)

        return x, latent

    def loss(self, x, h_t):
        x_hat, lats = self.forward(x, h_t)
        #x_hat = x_hat.view(x_hat.shape[0]*x_hat.shape[1], x_hat.shape[2], x_hat.shape[3])
        #x = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
        loss = self.Rloss((x_hat), (x))

        return loss, lats


    # def train_step(self, batch, optimizer):
        
    #     loss = self.loss(batch)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     return loss.item()
