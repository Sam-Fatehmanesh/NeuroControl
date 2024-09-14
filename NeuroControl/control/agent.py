import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from NeuroControl.models.world import NeuralWorldModel
from NeuroControl.models.actor import NeuralControlActor
from NeuroControl.custom_functions.laprop import LaProp
from NeuroControl.custom_functions.utils import *
from NeuroControl.custom_functions.agc import AGC
#from nfnets.agc import AGC

import pdb


class NeuralAgent(nn.Module):
    def __init__(self, num_neurons, frames_per_step, state_latent_size, steps_per_ep, env, image_latent_size_sqrt=20):
        super(NeuralAgent, self).__init__()

        self.action_dims = env.action_dims


        self.steps_per_ep = steps_per_ep
        self.seq_size = frames_per_step
        self.state_latent_size = state_latent_size
        self.num_neurons = num_neurons
        self.frames_per_step = frames_per_step

        self.image_n = 96
        self.reward_dims = (5,)
        self.reward_value_exp_bin_count = 41
        #self.action_dims = action_dims

        self.image_latent_size_sqrt = image_latent_size_sqrt

        self.seq_obs_latent = self.image_latent_size_sqrt**2 * self.seq_size

        # Initialize the world and actor models
        print("Initializing world and actor models.")
        self.world_model = NeuralWorldModel(frames_per_step, self.action_dims, self.image_n, state_latent_size, self.image_latent_size_sqrt)
        self.actor_model = NeuralControlActor(state_latent_size + self.image_latent_size_sqrt**2, state_latent_size, self.action_dims)

        # Loss function and optimizer
        print("Setting up optimizers.")
        linear_layer_names = [name for name, module in self.world_model.named_modules() if isinstance(module, nn.Linear)]
        self.optimizer_w = LaProp(self.world_model.parameters(), eps=1e-20, lr=4e-5)
        self.optimizer_w = AGC(self.world_model.parameters(), self.optimizer_w, model=self.world_model, ignore_agc=linear_layer_names)
        self.optimizer_a = LaProp(self.actor_model.parameters())
    
    def print_module_names(self):
        for name, module in self.world_model.named_modules():
            print(f"Module name: {name}, Type: {type(module).__name__}")

    def act(self, state):
        return self.actor_model(state.detach())
    
    def act_learn_forward(self, state):
        return self.actor_model(state)
    
    def world_model_pre_train_forward(self, obs_list, actions_list, rewards_list):
        total_loss = torch.zeros(1)

        batch_length = obs_list.shape[1] // self.frames_per_step
        batch_size = obs_list.shape[0]

        hidden_state = torch.zeros(batch_size, self.state_latent_size)

        decoded_obs_list = []

        for i in range(batch_length):
            # Takes 
            obs = obs_list[:, self.frames_per_step*i:self.frames_per_step *(i+1) ]
            actions = actions_list[:, self.frames_per_step*i:self.frames_per_step*(i+1)]
            rewards = rewards_list[:, self.frames_per_step*i:self.frames_per_step*(i+1)]

            # Forward pass through the world model
            #pdb.set_trace()
            decoded_obs, pred_next_obs_lat, obs_lats, hidden_state, predicted_rewards_logits, predicted_rewards_logits_ema, obs_lats_dist, pred_obs_lat_dist = self.world_model.forward(obs, actions, hidden_state)

            decoded_obs_list.append(decoded_obs)

            #total_loss += representation_loss + reward_prediction_loss + (kl_loss)
        decoded_obs_list = torch.stack(decoded_obs_list, dim = 0)
        return decoded_obs_list


    def pre_training_loss(self, obs_list, actions_list, rewards_list, all_losses = False):
        total_loss = torch.zeros(1)
        reward_predictor_loss = torch.zeros(1)
        decoder_representation_loss = torch.zeros(1)
        dynamics_encoder_kl_loss = torch.zeros(1)

        batch_length = obs_list.shape[1] // self.frames_per_step
        batch_size = obs_list.shape[0]

        hidden_state = torch.zeros(batch_size, self.state_latent_size)

        
        for i in range(batch_length):
            #pdb.set_trace()
            # Takes 
            obs = obs_list[:, self.frames_per_step*i:self.frames_per_step *(i+1) ]
            actions = actions_list[:, self.frames_per_step*i:self.frames_per_step*(i+1)]
            rewards = rewards_list[:, self.frames_per_step*i:self.frames_per_step*(i+1)]

            # Forward pass through the world model
            #pdb.set_trace()
            decoded_obs, pred_obs_lat, obs_lats, hidden_state, predicted_rewards_logits, predicted_rewards_logits_ema, obs_lats_dist, pred_obs_lat_dist = self.world_model.forward(obs, actions, hidden_state)
            

            # Compute the loss

            predicted_rewards_logits = predicted_rewards_logits.view(batch_size*self.seq_size, self.reward_value_exp_bin_count)
            predicted_rewards_logits_ema = predicted_rewards_logits_ema.view(batch_size*self.seq_size, self.reward_value_exp_bin_count)

            rewards = torch.reshape(rewards, shape=(batch_size*self.seq_size,))
            #pdb.set_trace()
            twohotloss, predicted_rewards = twohot_symexp_loss(predicted_rewards_logits, rewards, num_bins=self.reward_value_exp_bin_count)
            reward_predictor_ema_reg_loss = F.cross_entropy(predicted_rewards_logits, torch.softmax(predicted_rewards_logits_ema, dim=1), reduction="mean")
            #pdb.set_trace()
            reward_prediction_loss = (twohotloss + reward_predictor_ema_reg_loss)
            if i == 0:
                reward_prediction_loss *= 0

            reward_predictor_loss += reward_prediction_loss
            
            mse_rewards_loss = F.mse_loss(predicted_rewards, rewards)
            
            representation_loss = F.binary_cross_entropy(decoded_obs, obs) #F.mse_loss(obs, decoded_obs)# * 16
            decoder_representation_loss += representation_loss
            #reward_prediction_loss = (symlogMSE(predicted_rewards, rewards) + symlogMSE(predicted_rewards_ema, predicted_rewards))
            #kl_loss = kl_divergence_with_free_bits(pred_obs_lat.detach(), obs_lats) + kl_divergence_with_free_bits(pred_obs_lat, obs_lats.detach()) 
            kl_loss = kl_divergence_with_free_bits(obs_lats_dist.detach(), pred_obs_lat_dist, batch_size) + 0.1 * kl_divergence_with_free_bits(obs_lats_dist, pred_obs_lat_dist.detach(), batch_size) 
            dynamics_encoder_kl_loss += kl_loss

            
                
            #total_loss += torch.flatten(representation_loss) + torch.flatten(reward_prediction_loss) + torch.flatten(kl_loss)
        total_loss = reward_predictor_loss + decoder_representation_loss + dynamics_encoder_kl_loss
        if all_losses:
            return total_loss, decoder_representation_loss, reward_predictor_loss, dynamics_encoder_kl_loss, mse_rewards_loss
        return total_loss



    def predict_image_latents(self, steps, lat_0, actions=None):
        
        #pdb.set_trace()
        batch_dim = lat_0.shape[0]

        pred_image_latents = torch.empty((batch_dim, 0, self.seq_obs_latent))

        saved_h_states = torch.empty((batch_dim, 0, self.state_latent_size))

        latent = lat_0

        h_state = torch.zeros(batch_dim, self.state_latent_size)

        for i in range(steps):
            #torch.rand(batch_dim, self.seq_size, *self.action_dims)
            if actions == None:
                action = torch.rand(batch_dim, self.seq_size, *self.action_dims)
            else:
                action = actions[:, self.frames_per_step*i:self.frames_per_step*(i+1)]
            #pdb.set_trace()
            # print("############")
            # print(latent.size())
            # print(h_state.size())
            # print(action.size())
            h_state = self.world_model.seq_model.forward(latent, h_state, action)
            #print(h_state.size())
            latent, latent_distribution = self.world_model.rep_model.forward(h_state)
            saved_h_states = torch.cat((saved_h_states, h_state.unsqueeze(1)), dim=1)
            pred_image_latents = torch.cat((pred_image_latents, latent.unsqueeze(1)), dim=1)

        return pred_image_latents, saved_h_states

    def predict_obs(self, latents, h_states):
        batch_dim = latents.shape[0]

        current_seq_length = latents.shape[1]

        #pdb.set_trace()

        #latents = latents.view(batch_dim*current_seq_length*self.seq_size, self.image_latent_size_sqrt**2)
        #pdb.set_trace()
        obs = self.world_model.autoencoder.decode(latents, h_states)
        #obs = obs.view(batch_dim * , 8, self.image_n, self.image_n)

        return obs
        
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    # Saves a text file containing the model turned into a string
    def save_str_file_arch(self, path):
        with open(path, 'w') as f:
            f.write(str(self))


    def update_critic_ema_model(self):
        self.world_model.critic.update_ema()

    