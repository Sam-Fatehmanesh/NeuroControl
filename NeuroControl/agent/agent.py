import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from NeuroControl.models.world import NeuralWorldModel
from NeuroControl.models.actor import NeuralControlActor
from custom_functions.laprop import LaProp

class NeuralAgent:
    def __init__(self, num_neurons, frames_per_step, state_latent_size, steps_per_ep, env):

        self.action_dims = env.action_dims
        self.steps_per_ep = steps_per_ep
        self.state_latent_size = state_latent_size
        self.num_neurons = num_neurons
        self.frames_per_step = frames_per_step

        self.image_n = 280
        self.reward_dims = (5,)

        self.image_latent_size_sqrt = 16

        # Initalize the world and actor models
        print("Initializing world and actor models.")
        self.world_model = NeuralWorldModel(frames_per_step, self.reward_dims, self.image_n, state_latent_size, image_latent_size_sqrt)
        self.actor_model = NeuralControlActor(state_latent_size + self.image_latent_size_sqrt**2, state_latent_size, action_dims)

        # Loss function and optimizer
        print("Setting up optimizers.")
        optimizer_w = LaProp(self.world_model.parameters())
        optimizer_a = LaProp(self.actor_model.parameters())




    def act(self, state):
        return self.actor_model(state.detach())
    
    def act_learn_forward(self, state):
        return self.actor_model(state)
    
    def pre_training_loss(self, obs_list, actions_list, rewards_list):
        total_loss = torch.zeros(1)

        batch_length = len(obs_list)
        batch_size = obs_list[0].shape[0]

        hidden_state = torch.zeros(batch_size, self.state_latent_size)

        for i in range(batch_length):
            obs = obs_list[i]
            actions = actions_list[i]
            rewards = rewards_list[i]

            # Forward pass through the world model
            decoded_obs, pred_next_obs_lat, obs_lats, hidden_state, predicted_rewards = self.world_model.forward(obs, actions, hidden_state)

            # Compute the loss
            representation_loss = F.mse_loss(obs, decoded_obs)
            reward_prediction_loss = F.mse_loss(predicted_rewards, rewards)
            klloss += F.kl_div(torch.log(pred_next_obs_lat).detach(), obs_lats, reduction='mean') + F.kl_div(torch.log(pred_next_obs_lat), obs_lats.detach(), reduction='mean')

            total_loss += representation_loss + reward_prediction_loss + klloss


        
        return total_loss


        







        # for i in range(len(its)):
        #     # Get the data for this iteration
        #     obs_i = obs[i*self.steps_per_ep:(i+1)*self.steps_per_ep]
        #     actions_i = actions[i*self.steps_per_ep:(i+1)*self.steps_per_ep]
        #     rewards_i = rewards[i*self.steps_per_ep:(i+1)*self.steps_per_ep]

        #     hat_obs_i = 
