import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np
from NeuroControl.models.world import NeuralWorldModel
from NeuroControl.models.actor import NeuralControlActor
from NeuroControl.custom_functions.laprop import LaProp
from NeuroControl.custom_functions.utils import *
from NeuroControl.custom_functions.agc import AGC
from NeuroControl.models.critic import NeuralControlCritic
import pdb
from tqdm import tqdm


class NeuralAgent(nn.Module):
    def __init__(self, num_neurons, frames_per_step, h_state_latent_size, image_latent_size_sqrt=20):
        super(NeuralAgent, self).__init__()

        self.action_dims = (frames_per_step, 5)


        self.seq_size = frames_per_step
        self.h_state_latent_size = h_state_latent_size
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
        self.world_model = NeuralWorldModel(frames_per_step, self.action_dims, self.image_n, h_state_latent_size, self.image_latent_size_sqrt, reward_prediction_logits_num=self.reward_value_exp_bin_count)
        self.actor_model = NeuralControlActor(h_state_latent_size + self.seq_obs_latent, h_state_latent_size*2, self.action_dims)
        self.critic_model = NeuralControlCritic(h_state_latent_size + self.seq_obs_latent, self.frames_per_step, 1, reward_prediction_logits_num=self.reward_value_exp_bin_count)

        # Loss function and optimizer
        print("Setting up optimizers.")
        # Combine all parameters
        all_params = list(self.world_model.parameters()) + list(self.actor_model.parameters()) + list(self.critic_model.parameters())

        # Single shared optimizer
        print("Setting up shared optimizer.")
        self.optimizer = LaProp(all_params, eps=1e-20, lr=4e-5)
        
        # Apply AGC to all linear layers
        #linear_layer_names = [name for name, module in self.named_modules() if isinstance(module, nn.Linear)]
        self.optimizer = AGC(all_params, self.optimizer, model=None, ignore_agc=None)


        self.dreamed_rewards = []

    def act_from_state(self, state):
        action_sample, action_dist = self.actor_model(state.detach())
        return action_sample, action_dist

    def act_sample_from_hidden_state_and_obs(self, hidden_state, obs_lat):
        state = torch.cat((hidden_state, obs_lat), dim=1)
        action_sample, action_dist = self.actor_model(state.detach())
        return action_sample
    
    def state_and_obslat_from_obs(self, obs, action, hidden_state):
        decoded_obs, pred_next_obs_lat, obs_lats, hidden_state, predicted_rewards_logits, predicted_rewards_logits_ema, obs_lats_dist, pred_obs_lat_dist = self.world_model.forward(obs, action, hidden_state)
        return hidden_state, obs_lats
        

    def world_model_pre_train_forward(self, obs_list, actions_list, rewards_list):
        total_loss = torch.zeros(1)

        batch_length = obs_list.shape[1] // self.frames_per_step
        batch_size = obs_list.shape[0]

        hidden_state = torch.zeros(batch_size, self.h_state_latent_size)

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


    def replay_training_loss(self, obs_list, actions_list, rewards_list, all_losses = False):
        world_total_loss = torch.zeros(1)
        reward_predictor_loss = torch.zeros(1)
        decoder_representation_loss = torch.zeros(1)
        dynamics_encoder_kl_loss = torch.zeros(1)

        batch_length = obs_list.shape[1] // self.frames_per_step
        batch_size = obs_list.shape[0]

        hidden_state = torch.zeros(batch_size, self.h_state_latent_size)

        init_obs_lat = None
        init_h_state = None


        model_states = torch.zeros(batch_size, 1, self.h_state_latent_size + self.seq_obs_latent)
        
        for i in range(batch_length):
            #pdb.set_trace()
            # Takes 
            obs = obs_list[:, self.frames_per_step*i:self.frames_per_step *(i+1) ]
            actions = actions_list[:, self.frames_per_step*i:self.frames_per_step*(i+1)]
            rewards = rewards_list[:, self.frames_per_step*i:self.frames_per_step*(i+1)]

            # Forward pass through the world model
            #pdb.set_trace()
            decoded_obs, pred_obs_lat, obs_lats, hidden_state, predicted_rewards_logits, predicted_rewards_logits_ema, obs_lats_dist, pred_obs_lat_dist = self.world_model.forward(obs, actions, hidden_state)
            model_states = torch.cat( (model_states, torch.cat( (hidden_state.unsqueeze(1), obs_lats.unsqueeze(1)), dim = -1 ) ), dim = 1)
            
            if init_h_state is None:
                init_h_state = hidden_state
                init_obs_lat = obs_lats

            # Compute the loss

            predicted_rewards_logits = predicted_rewards_logits.view(batch_size*self.seq_size, self.reward_value_exp_bin_count)
            predicted_rewards_logits_ema = predicted_rewards_logits_ema.view(batch_size*self.seq_size, self.reward_value_exp_bin_count)

            rewards = torch.reshape(rewards, shape=(batch_size*self.seq_size,))
            #pdb.set_trace()
            twohotloss, predicted_rewards = twohot_symexp_loss(predicted_rewards_logits, rewards, num_bins=self.reward_value_exp_bin_count)
            reward_predictor_ema_reg_loss = F.cross_entropy(predicted_rewards_logits, torch.softmax(predicted_rewards_logits_ema, dim=1), reduction="mean")
            #pdb.set_trace()
            reward_prediction_loss = (twohotloss + reward_predictor_ema_reg_loss)

            reward_predictor_loss += reward_prediction_loss
            
            mse_rewards_loss = F.mse_loss(predicted_rewards, rewards)
            
            representation_loss = F.binary_cross_entropy(decoded_obs, obs) #F.mse_loss(obs, decoded_obs)# * 16
            decoder_representation_loss += representation_loss
            #reward_prediction_loss = (symlogMSE(predicted_rewards, rewards) + symlogMSE(predicted_rewards_ema, predicted_rewards))
            #kl_loss = kl_divergence_with_free_bits(pred_obs_lat.detach(), obs_lats) + kl_divergence_with_free_bits(pred_obs_lat, obs_lats.detach()) 
            kl_loss = kl_divergence_with_free_bits(obs_lats_dist.detach(), pred_obs_lat_dist, batch_size) + 0.1 * kl_divergence_with_free_bits(obs_lats_dist, pred_obs_lat_dist.detach(), batch_size) 
            dynamics_encoder_kl_loss += kl_loss

            
                
            #total_loss += torch.flatten(representation_loss) + torch.flatten(reward_prediction_loss) + torch.flatten(kl_loss)

        model_states = model_states[:, 1:, :]
        predicted_value_logits, predicted_value_logits_ema = self.critic_model(model_states.detach().reshape(batch_size*batch_length, -1))
        
        predicted_value_logits = torch.squeeze(predicted_value_logits, dim=1)
        predicted_value_logits_ema = torch.squeeze(predicted_value_logits_ema, dim=1)

        predicted_value = logits_to_reward(predicted_value_logits, self.reward_value_exp_bin_count)
        predicted_value_ema = logits_to_reward(predicted_value_logits_ema, self.reward_value_exp_bin_count)

        predicted_value = predicted_value.reshape(batch_size, batch_length)
        predicted_value_ema = predicted_value_ema.reshape(batch_size, batch_length)

        reward_sums = torch.sum(rewards_list.view(batch_size, self.frames_per_step, batch_length), dim=1).view(batch_size, batch_length)
        returns = self.compute_returns(reward_sums, predicted_value.detach())


        twohotloss, predicted_value = twohot_symexp_loss(predicted_value_logits.reshape(-1, self.reward_value_exp_bin_count), 
                                                        returns.reshape(-1).detach(), 
                                                        num_bins=self.reward_value_exp_bin_count)

        critic_ema_reg_loss = F.cross_entropy(predicted_value_logits.reshape(-1, self.reward_value_exp_bin_count), 
                                            torch.softmax(predicted_value_logits_ema.reshape(-1, self.reward_value_exp_bin_count), dim=1), 
                                            reduction="mean")

        critic_loss = (twohotloss + critic_ema_reg_loss)



        tqdm.write("$$$$$$$$$$$$$$$$$$$$$$$")
        tqdm.write(str(predicted_rewards))
        tqdm.write(str(rewards))

        
        world_total_loss = reward_predictor_loss + decoder_representation_loss + dynamics_encoder_kl_loss
        if all_losses:
            return world_total_loss, decoder_representation_loss, reward_predictor_loss, dynamics_encoder_kl_loss, mse_rewards_loss, critic_loss, (init_obs_lat, init_h_state)
        return world_total_loss, critic_loss, (init_obs_lat, init_h_state)



    def predict_image_latents(self, steps, lat_0, actions=None):
        
        #pdb.set_trace()
        batch_dim = lat_0.shape[0]

        pred_image_latents = torch.empty((batch_dim, 0, self.seq_obs_latent))

        saved_h_states = torch.empty((batch_dim, 0, self.h_state_latent_size))

        latent = lat_0

        h_state = torch.zeros(batch_dim, self.h_state_latent_size)

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

        
    def update_models(self, world_loss, actor_loss, critic_loss):
        self.optimizer.zero_grad()
        total_loss = world_loss + actor_loss + critic_loss
        total_loss.backward()
        self.optimizer.step()
        
        self.world_model.reward_model.update_ema()
        self.critic_model.update_ema()


    def imagine_ahead(self, initial_h_state, initial_obs_latent, initial_reward, horizon):
        device = initial_h_state.device
        batch_size = initial_h_state.shape[0]
        
        #h_states = [initial_h_state]
        #obs_latents = [initial_obs_latent]
        next_h_state = initial_h_state
        next_obs_latent = initial_obs_latent


        model_states = [torch.cat((next_h_state, next_obs_latent), dim=1)]
        actions = [torch.zeros((batch_size, *self.action_dims))]
        action_distributions = [torch.zeros((batch_size, *self.action_dims))]

        total_action_entropy = torch.zeros(1).to(device)
    
        rewards = [initial_reward]
        
        for _ in range(horizon):
            
            action_sample, action_distribution = self.act_from_state(model_states[-1])
            # Sanity testing by replacing action from model with a random action based on a random softmax dist turned into a 1-hot vector sample
            # action_distribution = F.softmax(torch.rand((batch_size, *self.action_dims)), dim=-1).to(device)
            # not_one_hot_sample = torch.multinomial(action_distribution.view(batch_size*self.frames_per_step, self.action_dims[1]), num_samples=1).view(batch_size, self.frames_per_step)
            # action_sample  = F.one_hot(not_one_hot_sample, num_classes=self.action_dims[1]).float().to(device)

            #pdb.set_trace()

            with torch.no_grad():
                next_h_state, next_obs_latent, predicted_reward = self.world_model.imagine_forward(next_h_state, next_obs_latent, actions[-1])
            
            model_states.append(torch.cat((next_h_state, next_obs_latent), dim=1))
            actions.append(action_sample)
            action_distributions.append(action_distribution)
            rewards.append(predicted_reward)
            
        
            
        #pdb.set_trace()
        return torch.stack(model_states, dim=1), torch.stack(actions, dim=1), torch.stack(rewards, dim=1), torch.stack(action_distributions, dim=1)


    def compute_returns(self, rewards, values, discount=0.997, lambda_=0.95):
        # rewards shape: [batch_size, horizon]
        # values shape: [batch_size, horizon]

        
        batch_size, horizon = rewards.shape
        returns = torch.zeros_like(rewards)
        next_return = values[:, -1]  # Initialize with the last value estimate
        
        for t in reversed(range(horizon)):
            returns[:, t] = rewards[:, t] + discount * (
                (1 - lambda_) * values[:, t] + lambda_ * next_return
            )
            next_return = returns[:, t]
        
        return returns


    def imaginary_training(self, initial_h_state, initial_obs_latent, initial_reward, horizon=3, repeat=1, batch_multiplier=1):
        batch_size = initial_h_state.shape[0]
        device = initial_h_state.device

        # Hyperparameters
        discount = 0.997
        lambda_ = 0.95
        eta = 3e-4  # Entropy scale

        B_critic_imagine_loss_factor = 0.3

        batch_length = horizon + 1

        critic_loss = torch.zeros(1).to(device)
        actor_loss = torch.zeros(1).to(device)
        total_predicted_return = torch.zeros(1).to(device)

        imagined_probable_action = None

        for _ in range(repeat):

            # Imagine ahead
            model_states, action_samples, predicted_reward, action_distributions = self.imagine_ahead(initial_h_state, initial_obs_latent, initial_reward, horizon)

            #pdb.set_trace()
            predicted_reward = torch.sum(predicted_reward, dim=2).view(batch_size, batch_length)

            model_states = model_states.reshape(batch_size * batch_length, -1)

            # Compute values
            predicted_value_logits, predicted_value_logits_ema = self.critic_model(model_states.detach())
            predicted_value_logits = torch.squeeze(predicted_value_logits, dim=1)
            predicted_value_logits_ema = torch.squeeze(predicted_value_logits_ema, dim=1)

            predicted_value = logits_to_reward(predicted_value_logits, self.reward_value_exp_bin_count)
            predicted_value_ema = logits_to_reward(predicted_value_logits_ema, self.reward_value_exp_bin_count)

            predicted_value = predicted_value.reshape(batch_size, batch_length)
            predicted_value_ema = predicted_value_ema.reshape(batch_size, batch_length)

            

            # Compute returns
            returns = self.compute_returns(predicted_reward, predicted_value.detach(), discount, lambda_)
            #returns = returns.view(batch_size*batch_length)


            twohotloss, predicted_values = twohot_symexp_loss(predicted_value_logits.reshape(-1, self.reward_value_exp_bin_count), 
                                                            returns.reshape(-1).detach(), 
                                                            num_bins=self.reward_value_exp_bin_count)

            critic_ema_reg_loss = F.cross_entropy(predicted_value_logits.reshape(-1, self.reward_value_exp_bin_count), 
                                                torch.softmax(predicted_value_logits_ema.reshape(-1, self.reward_value_exp_bin_count), dim=1), 
                                                reduction="mean")

            critic_loss += (twohotloss + critic_ema_reg_loss) * B_critic_imagine_loss_factor


            # Actor loss
            # Normalize and divide by max(range, 1)
            returns_95 = torch.quantile(returns, 0.95, dim=None, keepdim=False)
            returns_05 = torch.quantile(returns, 0.05, dim=None, keepdim=False)

            normalized_returns = (returns) / max(returns_95-returns_05, torch.tensor(1.0))

            # Compute policy loss using Reinforce estimator
            policy_loss = -(normalized_returns[:, 1:].unsqueeze(-1).unsqueeze(-1).expand(batch_size, horizon, *self.action_dims).detach() * torch.log(action_distributions[:, 1:])).sum()
            policy_loss /= batch_size

            # Compute entropy bonus
            entropy = self.actor_model.entropy(action_distributions[:, 1:])

            # Combine policy loss and entropy bonus
            actor_loss += policy_loss - eta * entropy


            total_predicted_return += predicted_reward.sum() / batch_size


            imagined_probable_action = action_samples.sum(dim=tuple(range(action_samples.dim() - 1))).argmax()



        critic_loss = torch.squeeze(critic_loss)

        return actor_loss, critic_loss, total_predicted_return, imagined_probable_action