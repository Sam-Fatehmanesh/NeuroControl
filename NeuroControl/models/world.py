import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.neuraloperator import FNOPredictor
from NeuroControl.models.transcnn import TransCNN
from NeuroControl.models.cnn import CNNLayer, DeCNNLayer
from NeuroControl.models.transformer import Transformer
from NeuroControl.models.encoder import SpikePositionEncoding
from NeuroControl.models.moe import MoEPredictor
from NeuroControl.models.mlp import MLP
from NeuroControl.models.critic import NeuralControlCritic
from NeuroControl.models.dynamics_predictor import NeuralRecurrentDynamicsModel
from NeuroControl.models.autoencoder import NeuralAutoEncoder
import pdb
import csv
from soft_moe_pytorch import SoftMoE, DynamicSlotsSoftMoE
from mamba_ssm import Mamba2 as Mamba
from NeuroControl.custom_functions.utils import STMNsampler, symlog, symexp


class NeuralWorldModel(nn.Module):
    def __init__(self, num_frames_per_step, action_dims, image_n, hidden_state_size, image_latent_size_sqrt):
        super(NeuralWorldModel, self).__init__()

        assert hidden_state_size % num_frames_per_step == 0, "Hidden state size must be divisible by number of frames per step"

        self.image_n = image_n
        self.action_dims = action_dims
        self.action_size = np.prod(action_dims) * num_frames_per_step

        self.seq_size = num_frames_per_step

        self.image_n = image_n

        self.hidden_state_size = hidden_state_size



        self.per_image_discrete_latent_size_sqrt = image_latent_size_sqrt
        self.seq_obs_latent = self.per_image_discrete_latent_size_sqrt**2 * num_frames_per_step

        self.autoencoder = NeuralAutoEncoder(num_frames_per_step, self.image_n, hidden_state_size, self.per_image_discrete_latent_size_sqrt)

        self.state_predictor = NeuralRecurrentDynamicsModel(self.hidden_state_size, self.seq_obs_latent, self.action_size, self.seq_size, self.per_image_discrete_latent_size_sqrt)

        self.critic = NeuralControlCritic(self.hidden_state_size, self.seq_size, self.seq_size)
        


    # def encode_obs(self, obs):
    #     return self.obs_autoencoder.encode(obs)

    # def decode_state(self, state):
    #     return self.obs_autoencoder.decode(state)

    # def auto_pred(self, obs):
    #     return self.obs_autoencoder(obs)

    def encode_obs(self, obs, hidden_state):
        batch_dim = obs.shape[0]
        z = self.autoencoder.encode(obs, hidden_state)
        z = z.view(batch_dim, self.seq_obs_latent)
        return z

    def forward(self, obs, action, hidden_state):
        batch_dim = obs.shape[0]

        #pdb.set_trace()
        #pdb.set_trace()
        predicted_rewards = self.critic.forward(hidden_state)

        decoded_obs, obs_lats = self.autoencoder(obs, hidden_state)
        decoded_obs = decoded_obs.view(batch_dim, self.seq_size, self.image_n, self.image_n)
        obs_lats = obs_lats.view(batch_dim, self.seq_obs_latent)

        #print(obs_lats.size(), action.size(), hidden_state.size())
        
        pred_next_obs_lat, hidden_state = self.state_predictor.forward(obs_lats, hidden_state, action)
        
        


        return decoded_obs, pred_next_obs_lat, obs_lats, hidden_state, predicted_rewards


    


class WorldModelMamba(nn.Module):
    def __init__(self, image_n, num_frames_per_step, latent_size, action_size, cnn_kernel_size=3):
        super(WorldModelMamba, self).__init__()
        
        self.critic_loss_func = nn.MSELoss()
        self.pred_loss = nn.MSELoss()

        self.action_size = action_size
        self.num_frames_per_step = num_frames_per_step
        self.image_n = image_n
        self.latent_size = latent_size
        state_size = latent_size
        self.state_size = state_size

        self.encoder_CNN_out_ch = 1

        num_2pools = 5

        self.encoder_CNN_0 = nn.Sequential(
            CNNLayer(1, 8, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(8, 128, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
        )

        self.encoder_CNN_1 = nn.Sequential(
            CNNLayer(128, 64, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(64, 32, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(32, 8, cnn_kernel_size),
            nn.MaxPool2d(2, stride=2),
            CNNLayer(8, self.encoder_CNN_out_ch, cnn_kernel_size),
        )
        self.postCNN_image_n = int(image_n/(2**num_2pools))
        pixles_num = int(self.postCNN_image_n**2)
        self.per_image_dim = self.encoder_CNN_out_ch * pixles_num
        self.predim = self.per_image_dim * num_frames_per_step
        self.transformer_in_dim = self.predim + state_size + action_size

        self.latent_vid_size = self.num_frames_per_step*latent_size
        
        # self.state_encoder = nn.Linear(latent_size, state_size)

        self.action_flat = nn.Flatten(start_dim=0)

        self.flat = nn.Flatten()

        # Latent Representation Transformer
        #self.lr_transformer = Transformer(self.per_image_dim, latent_size, heads=8, ff_dim=latent_size*2, out_dim=latent_size, layer_num=8)
        self.lr_mlp = MLP(4, self.transformer_in_dim, latent_size, self.latent_vid_size)#MoEPredictor(self.per_image_dim, latent_size, latent_size, num_experts=8, num_layers=8)

        self.mamba = nn.Sequential(
            Mamba(latent_size),
            Mamba(latent_size),
            Mamba(latent_size),
            Mamba(latent_size),
        )

        #self.decoder_transformer = Transformer(latent_size, latent_size, heads=8, ff_dim=latent_size*2, out_dim=pixles_num, layer_num=4)
        
        #!!!!!!!!!!!!!!!!! TEMP SETTING OUTPUT AS 35 WAS pixles_num

        pixles_num = int(35**2)
        self.decoder_mlp = MLP(2, latent_size, latent_size, pixles_num)
        self.dcnn_input_size = 35


        self.critic_0 = NeuralControlCritic(self.num_frames_per_step, self.latent_size)
        self.critic_1 = NeuralControlCritic(self.num_frames_per_step, self.latent_size)
        
        # Uses deconvolutions to generate an image
        # Upscales from 35x35 to 280x280
        self.decoder_DCNN_0 = nn.Sequential(
            DeCNNLayer(1, 128, kernel_size=4, stride=2, padding=1),
        )
        self.decoder_DCNN_1 = nn.Sequential(
            DeCNNLayer(128, 64, kernel_size=4, stride=2, padding=1),
            DeCNNLayer(64, 1, kernel_size=4, stride=2, padding=1),
        )


    def forward(self, critic_mode, observation_t, action_t, state_t, steps_left, current_r):

        
        xt = self.encoder_CNN_0(observation_t)
        image_res = xt
        xt = self.encoder_CNN_1(xt)
        xt = self.flat(xt)
        
        xt = xt.view(1,1, 1, self.predim)
        state_t = state_t.view(1, 1, 1, self.state_size)
        action_t = self.action_flat(action_t).view(1, 1, 1, self.action_size)

        obs_state_cat = torch.cat((xt, state_t, action_t), dim=3)
        #obs_state_cat=obs_state_cat.view(1, batch_dim + 1, self.per_image_dim, 1)
        
       # xt = xt.view(1,1,1,self.predim)
        zt = self.lr_mlp(obs_state_cat)
        zt = zt.view(1,self.num_frames_per_step, self.latent_size)

        zt_hat = self.mamba(zt.clone())
        
        # last_z_state = zt[:,-1]
        state_t_hat = zt_hat[:,-1]


        zt_hat = torch.transpose(zt_hat, 0, 1)

        # critic_in_state = zt_hat
        if not critic_mode:
            critic_in_state = zt_hat.detach()
        else:
            critic_in_state = zt_hat


        reward_hat_0 = self.critic_0(critic_in_state, steps_left, current_r)
        reward_hat_1 = self.critic_1(critic_in_state, steps_left, current_r)

        if not critic_mode:
            imglat = self.decoder_mlp(zt_hat)

            imglat = imglat.view(self.num_frames_per_step, 1, self.dcnn_input_size, self.dcnn_input_size)
            observation_t_hat = self.decoder_DCNN_0(imglat)
            observation_t_hat = observation_t_hat + image_res
            observation_t_hat = self.decoder_DCNN_1(observation_t_hat)

            return observation_t_hat, state_t_hat, reward_hat_0, reward_hat_1



        return None, None, reward_hat_0, reward_hat_1