import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from NeuroControl.models.world import NeuralWorldModel
from NeuroControl.models.actor import NeuralControlActor
from NeuroControl.custom_functions.laprop import LaProp
from NeuroControl.custom_functions.utils import *


import pdb


class NeuralAgent:
    def __init__(self, num_neurons, frames_per_step, state_latent_size, steps_per_ep, env):
        self.action_dims = env.action_dims
        #print(self.action_dims)
        #pdb.set_trace()
        self.steps_per_ep = steps_per_ep
        self.seq_size = frames_per_step
        self.state_latent_size = state_latent_size
        self.num_neurons = num_neurons
        self.frames_per_step = frames_per_step

        self.image_n = 96
        self.reward_dims = (5,)
        #self.action_dims = action_dims

        self.image_latent_size_sqrt = 16

        self.seq_obs_latent = self.image_latent_size_sqrt**2 * self.seq_size

        # Initialize the world and actor models
        print("Initializing world and actor models.")
        self.world_model = NeuralWorldModel(frames_per_step, self.action_dims, self.image_n, state_latent_size, self.image_latent_size_sqrt)
        self.actor_model = NeuralControlActor(state_latent_size + self.image_latent_size_sqrt**2, state_latent_size, self.action_dims)

        # Loss function and optimizer
        print("Setting up optimizers.")
        self.optimizer_w = LaProp(self.world_model.parameters())
        self.optimizer_a = LaProp(self.actor_model.parameters())

    def act(self, state):
        return self.actor_model(state.detach())
    
    def act_learn_forward(self, state):
        return self.actor_model(state)
    
    def pre_training_loss(self, obs_list, actions_list, rewards_list):
        total_loss = torch.zeros(1)

        batch_length = obs_list.shape[1] // self.frames_per_step
        batch_size = obs_list.shape[0]

        hidden_state = torch.zeros(batch_size, self.state_latent_size)

        for i in range(batch_length):
            # Takes 
            obs = obs_list[:, self.frames_per_step*i:self.frames_per_step*(i+1) ]
            actions = actions_list[:, self.frames_per_step*i:self.frames_per_step*(i+1)]
            rewards = rewards_list[:, self.frames_per_step*i:self.frames_per_step*(i+1)]

            # Forward pass through the world model
            #pdb.set_trace()
            decoded_obs, pred_next_obs_lat, obs_lats, hidden_state, predicted_rewards = self.world_model.forward(obs, actions, hidden_state)

            # Compute the loss
            
            representation_loss = F.mse_loss(obs, decoded_obs)
            reward_prediction_loss = F.mse_loss(predicted_rewards, rewards)
            kl_loss = kl_divergence_with_free_bits(pred_next_obs_lat.detach(), obs_lats) + kl_divergence_with_free_bits(pred_next_obs_lat, obs_lats.detach())

            #pdb.set_trace()
            total_loss += representation_loss + reward_prediction_loss + kl_loss

        return total_loss



    def predict_image_latents(self, steps, lat_0):

        batch_dim = lat_0.shape[0]

        pred_image_latents = torch.empty((batch_dim, 0, self.seq_obs_latent))
        latent = lat_0

        h_state = torch.zeros(batch_dim, self.state_latent_size)

        for _ in range(steps):
            action = torch.rand(batch_dim, self.seq_size, *self.action_dims)
            #pdb.set_trace()
            latent, h_state = self.world_model.state_predictor.forward(latent, h_state, action)
            pred_image_latents = torch.cat((pred_image_latents, latent.unsqueeze(1)), dim=1)

        return pred_image_latents

    def predict_obs(self, latents):
        batch_dim = latents.shape[0]

        current_seq_length = latents.shape[1]

        #pdb.set_trace()

        latents = latents.view(batch_dim*current_seq_length*self.seq_size, self.image_latent_size_sqrt**2)
        obs = self.world_model.autoencoder.decode(latents)
        obs = obs.view(batch_dim * current_seq_length, self.image_n, self.image_n)

        return obs
        
