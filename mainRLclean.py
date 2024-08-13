import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nest
from tqdm import tqdm
import cv2
import os
from matplotlib import pyplot as plt
import csv
import time
from datetime import datetime

from NeuroControl.SNNSimenv.snnenv import snnEnv
from NeuroControl.SNNSimenv.synthCI import create_video
from NeuroControl.models.world import WorldModelMamba
from NeuroControl.models.actor import NeuralControlActor

class NeuralControlAgent:
    def __init__(self, snn_params, neuron_params, rl_params, seed=42):
        self.snn_params = snn_params
        self.neuron_params = neuron_params
        self.rl_params = rl_params
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.setup_environment()
        self.setup_models()
        self.setup_optimizers()

        self.ep_scores = []
        self.losses_w = []
        self.losses_a = []
        self.losses_c = []
        self.sim_step_speed_times = []

    def setup_environment(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        nest.set_verbosity(30)

        self.env = snnEnv(snn_params=self.snn_params, 
                          neuron_params=self.neuron_params, 
                          rl_params=self.rl_params, 
                          snn_filename=None,
                          apply_optical_error=False)
        
        self.action_size = int(self.snn_params["num_neurons_stimulated"] * self.env.step_action_observsation_simulation_time)
        self.action_dims = (self.snn_params["num_neurons_stimulated"], self.env.step_action_observsation_simulation_time)

    def setup_models(self):
        image_n = 280
        num_frames_per_step = self.snn_params["step_action_observsation_simulation_time"]
        latent_size = int(32**2)
        self.state_size = latent_size

        self.world_model = WorldModelMamba(image_n, num_frames_per_step, latent_size, self.action_size).to(self.device)
        self.actor_model = NeuralControlActor(self.state_size, latent_size, self.action_dims).to(self.device)

    def setup_optimizers(self):
        self.optimizer_w = optim.Adam(self.world_model.parameters(), lr=0.0001)
        self.optimizer_a = optim.Adam(self.actor_model.parameters(), lr=0.00000001)

    def train(self, num_episodes):
        with tqdm(total=num_episodes, desc="Episodes") as pbar:
            for episode in range(num_episodes):
                self.run_episode(episode, num_episodes, pbar)

    def run_episode(self, episode, num_episodes, pbar):
        pretraining = episode < num_episodes // 2

        first_obs, _ = self.env.reset()
        first_obs = torch.tensor(np.array(first_obs, dtype=np.float32)).unsqueeze(0).to(self.device) / 255.0
        
        prev_obs = first_obs
        model_state = torch.rand(self.state_size).to(self.device)
        total_true_reward = np.zeros((1))
        action_distributions = torch.empty((0, *self.action_dims)).to(self.device)

        total_loss_w = torch.zeros((1)).to(self.device)
        predicted_total_rewards = torch.tensor([]).view(0, 2).to(self.device)
        predicted_total_rewards_train = torch.tensor([]).view(0, 2).to(self.device)

        for step in range(self.rl_params["steps_per_ep"]):
            t1 = time.time()
            action, predicted_obs, true_obs, reward, model_state = self.step(prev_obs, model_state, total_true_reward, pretraining, step)
            
            action_distributions = torch.cat((action_distributions, action.unsqueeze(0)), dim=0)
            total_true_reward += reward

            loss = self.world_model.pred_loss(predicted_obs, true_obs)
            total_loss_w += loss

            prev_obs = true_obs
            t2 = time.time()
            self.sim_step_speed_times.append(t2-t1)

        self.update_models(total_loss_w, predicted_total_rewards, predicted_total_rewards_train, action_distributions, total_true_reward, pretraining)
        
        self.ep_scores.append(total_true_reward)
        pbar.update(1)
        self.print_episode_info(total_loss_w, total_true_reward, action_distributions)

    def step(self, prev_obs, model_state, total_true_reward, pretraining, step):
        steps_left = self.rl_params["steps_per_ep"] - step
        action = self.actor_model(model_state.detach())
        if pretraining:
            action = torch.log(torch.rand(action.shape).to(self.device))

        action_sample = torch.bernoulli(torch.exp(action)).detach().cpu().numpy()

        prev_obs = torch.transpose(prev_obs, 0, 1)
        steps_left_tensor = int(steps_left) * torch.ones((1), dtype=torch.float32).to(self.device)
        current_reward_total_tensor = int(total_true_reward[0]) * torch.ones((1), dtype=torch.float32).to(self.device)

        predicted_obs, model_state, _, _ = self.world_model(False, prev_obs, action.detach(), model_state, steps_left_tensor, current_reward_total_tensor)

        true_obs, reward, done, _ = self.env.step(action_sample)
        reward /= self.rl_params["steps_per_ep"]

        true_obs = torch.tensor(np.array(true_obs, dtype=np.float32)).unsqueeze(0).to(self.device) / 255.0
        predicted_obs = torch.transpose(predicted_obs, 0, 1)

        return action, predicted_obs, true_obs, reward, model_state

    def update_models(self, total_loss_w, predicted_total_rewards, predicted_total_rewards_train, action_distributions, total_true_reward, pretraining):
        actor_model_loss_factor = 1
        actor_entropy_loss_factor = 0
        critic_model_loss_factor = 32

        action_entropy_sum = self.actor_model.entropy(action_distributions)
        actor_entropy_loss_term = actor_entropy_loss_factor * action_entropy_sum
        
        if not pretraining:
            predicted_total_rewards_lower_bound = torch.min(predicted_total_rewards, dim=1)[0]
            actor_policy_loss = torch.sum(predicted_total_rewards_lower_bound)
            actor_model_loss = -actor_model_loss_factor * (actor_policy_loss + actor_entropy_loss_term)
            
            self.optimizer_a.zero_grad()
            actor_model_loss.backward(retain_graph=True)
            self.optimizer_a.step()
            
            self.losses_a.append(actor_model_loss.item())

        total_true_reward_tensor = torch.ones_like(predicted_total_rewards_train) * torch.tensor(total_true_reward, dtype=torch.float).to(self.device)
        world_critic_model_loss = critic_model_loss_factor * torch.sum(self.world_model.critic_loss_func(predicted_total_rewards_train, total_true_reward_tensor))
        
        loss_w = total_loss_w + world_critic_model_loss
        self.losses_w.append(total_loss_w.item())
        self.losses_c.append(world_critic_model_loss.item())

        self.optimizer_w.zero_grad()
        loss_w.backward()
        self.optimizer_w.step()

    def print_episode_info(self, total_loss_w, total_true_reward, action_distributions):
        tqdm.write("#######################################")
        tqdm.write(f"World Model Loss: {total_loss_w.item()}")
        tqdm.write(f"Current Episode Total Reward: {total_true_reward}")
        tqdm.write("Middle action")
        mid_ep_action_index = len(action_distributions) // 2
        tqdm.write(str(action_distributions[mid_ep_action_index]))

    def generate_example_prediction_video(self, folder_name):
        self.world_model.eval()
        self.actor_model.eval()
        predictions = []

        first_obs, _ = self.env.reset()
        first_obs = torch.tensor(np.array(first_obs, dtype=np.float32)).unsqueeze(0).to(self.device) / 255.0
        
        prev_obs = first_obs
        model_state = torch.rand(self.state_size).to(self.device)
        total_true_reward = np.zeros((1))

        for step in range(self.rl_params["steps_per_ep"]):
            action, predicted_obs, true_obs, reward, model_state = self.step(prev_obs, model_state, total_true_reward, False, step)

            predicted_obs_np = predicted_obs.detach().cpu().numpy()
            predicted_obs_np = np.squeeze(predicted_obs_np, axis=1)
            predictions.extend([frame for frame in predicted_obs_np])

            total_true_reward += reward
            prev_obs = true_obs

        save_prediction_video(predictions, folder_name)

    def render_true_simulation(self, folder_name):
        sim_steps_rendering = self.snn_params["step_action_observsation_simulation_time"] * self.rl_params["steps_per_ep"]
        self.env.render(sim_steps_rendering, save_dir=folder_name)

def save_prediction_video(predictions, folder_name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 1.0
    frame_size = (280, 280)
    filename = folder_name + "predicted_simulation.mp4"
    out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

    for frame in predictions:
        frame *= 255.0
        frame = frame.squeeze()
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()

def ema(x, alpha):
    b = x.copy()
    ema = b[0]
    for i in range(1, len(b)):
        ema = alpha * b[i] + (1 - alpha) * ema
        b[i] = ema
    return b

def save_results(agent, folder_name):
    plot_rewards(agent.ep_scores, folder_name)
    plot_true_rewards(agent.env.reward_record, folder_name)
    plot_losses(agent.losses_w, agent.losses_a, agent.losses_c, folder_name)
    save_losses_csv(agent.losses_w, agent.losses_a, agent.losses_c, agent.ep_scores, folder_name)
    plot_sim_step_speed(agent.sim_step_speed_times, folder_name)

def plot_rewards(ep_scores, folder_name):
    plt.figure()
    plt.plot(ep_scores)
    plt.plot(ema(ep_scores, 1/6))
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.savefig(folder_name + "total_reward_graph.png")
    plt.close()

def plot_true_rewards(true_rewards, folder_name):
    plt.figure()
    plt.plot(true_rewards)
    plt.plot(ema(true_rewards, 1/6))
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("Score at Each Step")
    plt.savefig(folder_name + "step_score_graph.png")
    plt.close()

def plot_losses(losses_w, losses_a, losses_c, folder_name):
    for loss_type, losses in [("world", losses_w), ("actor", losses_a), ("critic", losses_c)]:
        plt.figure()
        plt.plot(losses)
        if loss_type == "critic":
            plt.plot(ema(losses, 1/6))
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title(f"{loss_type.capitalize()} Model Loss over Time")
        plt.savefig(folder_name + f"{loss_type}_loss_graph.png")
        plt.close()

def save_losses_csv(losses_w, losses_a, losses_c, ep_scores, folder_name):
    with open(folder_name + "losses.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "World Model Loss", "Actor Model Loss", "Critic Model Loss", "Env True Ep Score"])
        for idx, loss_w, loss_a, loss_c, score in zip(range(len(losses_w)), losses_w, losses_a, losses_c, ep_scores):
            writer.writerow([idx, loss_w, loss_a, loss_c, score])

def plot_sim_step_speed(sim_step_speed_times, folder_name):
    plt.figure()
    plt.plot(sim_step_speed_times[1:])
    plt.xlabel("Step")
    plt.ylabel("Time (s)")
    plt.title("Simulation Step Speed over Time")
    plt.savefig(folder_name + "sim_step_speed_graph.png")
    plt.close()

def save_parameters(agent, folder_name):
    with open(folder_name + 'params.txt', 'w') as f:
        f.write(str(agent.world_model))
        f.write('\n\n')
        f.write("snn_params\n")
        f.write(str(agent.snn_params))
        f.write('\n\n')
        f.write("neuron_params\n")
        f.write(str(agent.neuron_params))
        f.write('\n\n')
        f.write("rl_params\n")
        f.write(str(agent.rl_params))
        f.write('\n\n')
        f.write("seed\n")
        f.write(str(agent.seed))

def main():
    # Setup folders
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
    folder_name = f'experiments/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
    os.makedirs(folder_name)

    # Initialize parameters (snn_params, neuron_params, rl_params)
    # ... (your parameter initialization code here)

    agent = NeuralControlAgent(snn_params, neuron_params, rl_params)
    
    save_parameters(agent, folder_name)

    num_episodes = 8096
    agent.train(num_episodes)
    
