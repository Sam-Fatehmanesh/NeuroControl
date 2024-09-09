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


        self.post_cnn_encoder_size_sqrt = 8
        self.post_cnn_encoder_size = self.post_cnn_encoder_size_sqrt**2#14**2

        #self.pre_dcnn_decoder_size = 14**2

        # Encoder
        self.cnn_encoder = nn.Sequential(
            CNNLayer(1, 16, cnn_kernel_size),
            nn.MaxPool2d(3, stride=3),

            CNNLayer(16, 64, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),

            CNNLayer(64, 1, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),

            nn.Flatten(),
        )

        self.mlp_encoder = MLP(3, self.post_cnn_encoder_size + hidden_state_size, self.per_image_latent_size, self.per_image_latent_size)

        self.softmax_act = nn.Softmax(dim=1)
        self.sampler = STMNsampler()

        # self.discretizer = nn.Sequential(
        #     ,
        # )


        


        # Decoder
        self.decoder = nn.Sequential(
            
            MLP(3, self.per_image_latent_size + hidden_state_size, self.per_image_latent_size, self.post_cnn_encoder_size),
            
            nn.Unflatten(1, (1, self.post_cnn_encoder_size_sqrt, self.post_cnn_encoder_size_sqrt)),

            DeCNNLayer(1, 64, kernel_size=3, stride=3, padding=0),

            DeCNNLayer(64, 16, kernel_size=2, stride=2, padding=0),

            DeCNNLayer(16, 1, kernel_size=2, stride=2, padding=0),
            
            
            #nn.Sigmoid(),
        )

        #self.decode_upsample = nn.Upsample(size=(self.image_n, self.image_n), mode='bilinear')

        #self.act = nn.Sigmoid()


    def encode(self, x, h_t):
        batch_dim = x.shape[0]

        #pdb.set_trace()

        #x = x.view(batch_dim * self.frame_count, 1, self.image_n, self.image_n)
        x = torch.reshape(x, (batch_dim * self.frame_count, 1, self.image_n, self.image_n))

        x = self.cnn_encoder(x)
        #pdb.set_trace()
        h_t = torch.tile(h_t, (self.frame_count, 1))
        x = torch.cat( (x, h_t), dim=1)
        x = self.mlp_encoder(x)


        # Adding noise to make non deterministic
        x = 0.99 * x + 0.01 * (torch.rand(x.shape).to(x.device) - 0.5)


        x = x.view(batch_dim * self.frame_count * self.per_image_discrete_latent_side_size, self.per_image_discrete_latent_side_size)
        #pdb.set_trace()
        distributions = self.softmax_act(x)

        sample = self.sampler(distributions)

        sample = sample.view(batch_dim, self.frame_count, self.per_image_latent_size)
        distributions = distributions.view(batch_dim, self.frame_count, self.per_image_latent_size)

        
        
        return sample, distributions

    def decode(self, z, h_t):
        batch_dim = z.shape[0]
        #pdb.set_trace()
        z = z.view(batch_dim*self.frame_count, self.per_image_latent_size)
        h_t = torch.tile(h_t, (self.frame_count, 1))
        #pdb.set_trace()
        z = torch.cat( (z, h_t), dim=1)

        z = self.decoder(z)
        #pdb.set_trace()
        #z = self.decode_upsample(z)
        z = z.view(batch_dim, self.frame_count, self.image_n, self.image_n)


        return z
    
    def forward(self, x, h_t):
        batch_dim = x.shape[0]
        
        #pdb.set_trace()
        latent_sample, latent_distribution = self.encode(x, h_t)


        latent_sample = latent_sample.view(batch_dim, self.frame_count*self.per_image_latent_size)
        latent_distribution = latent_distribution.view(batch_dim, self.frame_count*self.per_image_latent_size)

        

        decoded_out = self.decode(latent_sample, h_t)


        #x = self.act(x)

        return decoded_out, latent_sample, latent_distribution

    # def loss(self, x, h_t):
    #     x_hat, lats = self.forward(x, h_t)
    #     #x_hat = x_hat.view(x_hat.shape[0]*x_hat.shape[1], x_hat.shape[2], x_hat.shape[3])
    #     #x = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
    #     loss = self.Rloss((x_hat), (x))

    #     return loss, lats


    # def train_step(self, batch, optimizer):
        
    #     loss = self.loss(batch)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     return loss.item()
