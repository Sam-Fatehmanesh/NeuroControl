import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NeuroControl.models.cnn import CNNLayer, DeCNNLayer
from NeuroControl.models.mlp import MLP
from NeuroControl.models.critic import NeuralControlCritic
from NeuroControl.models.dynamics_predictor import NeuralRecurrentDynamicsModel, NeuralRepModel, NeuralSeqModel
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

        self.frames_per_obs = num_frames_per_step

        self.image_n = image_n

        self.hidden_state_size = hidden_state_size



        self.per_image_discrete_latent_size_sqrt = image_latent_size_sqrt
        self.seq_obs_latent = self.per_image_discrete_latent_size_sqrt**2 * self.frames_per_obs

        self.autoencoder = NeuralAutoEncoder(num_frames_per_step, self.image_n, hidden_state_size, self.per_image_discrete_latent_size_sqrt)

        #self.state_predictor = NeuralRecurrentDynamicsModel(self.hidden_state_size, self.seq_obs_latent, self.action_size, self.frames_per_obs, self.per_image_discrete_latent_size_sqrt)
        self.seq_model = NeuralSeqModel(self.hidden_state_size, self.seq_obs_latent, self.action_size, self.frames_per_obs, self.per_image_discrete_latent_size_sqrt)
        self.rep_model = NeuralRepModel(self.hidden_state_size, self.seq_obs_latent, self.frames_per_obs, self.per_image_discrete_latent_size_sqrt)

        self.critic = NeuralControlCritic(self.hidden_state_size, self.frames_per_obs, self.frames_per_obs)



    def encode_obs(self, obs, hidden_state):
        batch_dim = obs.shape[0]
        z, dist = self.autoencoder.encode(obs, hidden_state)
        z = z.view(batch_dim, self.seq_obs_latent)
        return z

    def forward(self, obs, action, hidden_state):
        batch_dim = obs.shape[0]

        #pdb.set_trace()
        #pdb.set_trace()
        pred_obs_lat_sample, pred_obs_lat_dist = self.rep_model.forward(hidden_state)
        predicted_rewards_logits, predicted_rewards_logits_ema = self.critic.forward(hidden_state)
        
        decoded_obs, obs_lats_sample, obs_lats_dist = self.autoencoder(obs, hidden_state)
        decoded_obs = decoded_obs.view(batch_dim, self.frames_per_obs, self.image_n, self.image_n)
        obs_lats_sample = obs_lats_sample.view(batch_dim, self.seq_obs_latent)
        
        
        
        #, hidden_state = self.state_predictor.forward(obs_lats_sample, hidden_state, action)
        
        hidden_state = self.seq_model.forward(obs_lats_sample, hidden_state, action)


        

        return decoded_obs, pred_obs_lat_sample, obs_lats_sample, hidden_state, predicted_rewards_logits, predicted_rewards_logits_ema, obs_lats_dist, pred_obs_lat_dist    