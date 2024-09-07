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
        


    def encode_obs(self, obs, hidden_state):
        batch_dim = obs.shape[0]
        z, dist = self.autoencoder.encode(obs, hidden_state)
        z = z.view(batch_dim, self.seq_obs_latent)
        return z

    def forward(self, obs, action, hidden_state):
        batch_dim = obs.shape[0]

        #pdb.set_trace()
        #pdb.set_trace()
        

        decoded_obs, obs_lats_sample, obs_lats_dist = self.autoencoder(obs, hidden_state)
        decoded_obs = decoded_obs.view(batch_dim, self.seq_size, self.image_n, self.image_n)
        obs_lats_sample = obs_lats_sample.view(batch_dim, self.seq_obs_latent)

        #print(obs_lats.size(), action.size(), hidden_state.size())
        
        pred_obs_lat_sample, pred_obs_lat_dist, hidden_state = self.state_predictor.forward(obs_lats_sample, hidden_state, action)

        predicted_rewards = self.critic.forward(hidden_state)

        


        return decoded_obs, pred_obs_lat_sample, obs_lats_sample, hidden_state, predicted_rewards, obs_lats_dist, pred_obs_lat_dist


    