import torch
from torch import nn
import numpy as np
from NeuroControl.models.world import NeuralWorldModel
from NeuroControl.models.actor import NeuralControlActor

class NeuralAgent:
    def __init__(self, num_neurons, frames_per_step, state_latent_size, steps_per_ep):

        self.action_dims = env.action_dims
        self.steps_per_ep = steps_per_ep
        
        self.world_model = NeuralWorldModel(frames_per_step, state_latent_size)
        self.actor_model = NeuralControlActor(state_latent_size, state_latent_size, action_dims)

        # Loss function and optimizer
        print("Setting up optimizers.")
        optimizer_w = optim.Adam(self.world_model.parameters(), lr=0.00001)
        optimizer_a = optim.Adam(self.actor_model.parameters(), lr=0.00001)


        

    def act(self, state):
        return self.actor_model(state.detach())
    
    def act_learn_forward(self, state):
        return self.actor_model(state)
    
    def world_learn(self, obs, actions, rewards):
        its = len(obs) - (len(obs) % self.steps_per_ep)

        # Unsqueeze at batch size and then stack
        obs = torch.stack(obs, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)

        auto_obs_pred, latents = self.world_model.autoencoder(obs)

        autoencoder_loss = self.world_model.autoencoder.loss(obs, auto_obs_pred)

        







        # for i in range(len(its)):
        #     # Get the data for this iteration
        #     obs_i = obs[i*self.steps_per_ep:(i+1)*self.steps_per_ep]
        #     actions_i = actions[i*self.steps_per_ep:(i+1)*self.steps_per_ep]
        #     rewards_i = rewards[i*self.steps_per_ep:(i+1)*self.steps_per_ep]

        #     hat_obs_i = 
